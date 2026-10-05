"""
Веб-API ANPR: список распознанных номеров, поиск, CSV, live-события (SSE), снимки.
Видео браузер получает напрямую от MediaMTX (WebRTC/WHEP или HLS) через nginx.
"""
import asyncio
import csv
import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

EVENTS = Path(os.getenv("EVENTS_FILE", "/data/events.jsonl"))
SNAPS = Path(os.getenv("SNAPS_DIR", "/data/snaps"))
STATIC = Path(__file__).parent / "static"
# Камеры: path в MediaMTX -> заголовок. "raw" — исходный поток, "anpr" — с разметкой
CAMERAS = json.loads(os.getenv("CAMERAS",
    '[{"id":"cam1","title":"Въезд КПП-1","raw":"cam1","anpr":"anpr"}]'))

LAT2CYR = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")
SNAPS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ANPR RU")


def load_events():
    if not EVENTS.exists():
        return []
    out = []
    with EVENTS.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def plate_matcher(q: str):
    """Поиск по подстроке, латиница -> кириллица, '?' или '*' = любой символ."""
    q = re.sub(r"\s", "", q.upper()).translate(LAT2CYR)
    if not q:
        return lambda p: True
    rx = re.compile(re.escape(q).replace(r"\?", ".").replace(r"\*", ".*"))
    return lambda p: bool(rx.search(p))


def filtered(q, camera, date_from, date_to):
    m = plate_matcher(q or "")
    for e in reversed(load_events()):
        if camera and e.get("camera") != camera:
            continue
        if date_from and e["ts"] < date_from:
            continue
        if date_to and e["ts"] > date_to:
            continue
        if m(e["plate"]):
            yield e


@app.get("/api/config")
def config():
    return {"cameras": CAMERAS}


@app.get("/api/events")
def events(q: str = "", camera: str = "", date_from: str = "", date_to: str = "",
           limit: int = Query(200, le=5000)):
    res = []
    for e in filtered(q, camera, date_from, date_to):
        res.append(e)
        if len(res) >= limit:
            break
    return res


@app.get("/api/stats")
def stats():
    ev = load_events()
    hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")
    return {"total": len(ev),
            "today": sum(e["ts"].startswith(today) for e in ev),
            "last_hour": sum(e["ts"] >= hour_ago for e in ev),
            "unique_today": len({e["plate"] for e in ev if e["ts"].startswith(today)})}


@app.get("/api/events.csv")
def events_csv(q: str = "", camera: str = "", date_from: str = "", date_to: str = ""):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["Время", "Камера", "Номер", "Трек", "Голосов"])
    for e in filtered(q, camera, date_from, date_to):
        w.writerow([e["ts"], e.get("camera", ""), e["plate"], e.get("track", ""), e.get("votes", "")])
    data = "\ufeff" + buf.getvalue()  # BOM, чтобы Excel открыл кириллицу
    return StreamingResponse(iter([data]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=plates.csv"})


@app.get("/api/sse")
async def sse():
    """Хвост events.jsonl как Server-Sent Events. Переживает ротацию файла."""
    async def gen():
        f, ino = None, None
        idle = 0
        while True:
            if f is None and EVENTS.exists():
                f = EVENTS.open(encoding="utf-8"); f.seek(0, os.SEEK_END); ino = os.fstat(f.fileno()).st_ino
            line = f.readline() if f else ""
            if line.endswith("\n"):
                idle = 0
                yield f"event: plate\ndata: {line.strip()}\n\n"
                continue
            await asyncio.sleep(0.5); idle += 1
            if idle % 30 == 0:
                yield ": ping\n\n"                     # keepalive для прокси
                if f and EVENTS.exists() and EVENTS.stat().st_ino != ino:
                    f.close(); f = EVENTS.open(encoding="utf-8"); ino = os.fstat(f.fileno()).st_ino
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/snaps", StaticFiles(directory=SNAPS), name="snaps")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
