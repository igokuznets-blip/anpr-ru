"""
Веб-API ANPR: камеры и их состояние, журнал номеров, поиск, CSV, live-события (SSE), снимки.
Синхронизирует пути MediaMTX с cameras.yml через MediaMTX Control API.
Видео браузер получает от MediaMTX (WebRTC/WHEP или HLS) через nginx.
"""
from __future__ import annotations

import asyncio
import contextlib
import csv
import glob
import io
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))  # локальный запуск из репозитория
from cameras import Camera, load_cameras, parse_cameras  # noqa: E402

log = logging.getLogger("web")
EVENTS = Path(os.getenv("EVENTS_FILE", "/data/events.jsonl"))
SNAPS = Path(os.getenv("SNAPS_DIR", "/data/snaps"))
CAMERAS_FILE = os.getenv("CAMERAS_FILE", "/config/cameras.yml")
STATUS_GLOB = os.getenv("STATUS_GLOB", "/data/status-*.json")
MEDIAMTX_API = os.getenv("MEDIAMTX_API", "").rstrip("/")   # http://mediamtx:9997
SYNC_INTERVAL = float(os.getenv("MEDIAMTX_SYNC_INTERVAL", "15"))
STALE_SEC = 15
STATIC = Path(__file__).parent / "static"
LAT2CYR = str.maketrans("ABEKMHOPCTYX", "АВЕКМНОРСТУХ")
SNAPS.mkdir(parents=True, exist_ok=True)


# ---------- камеры (перечитываются при изменении cameras.yml) ----------
_cache: dict = {"mtime": None, "cams": [], "error": None}


def get_cameras(include_disabled: bool = False) -> list[Camera]:
    p = Path(CAMERAS_FILE)
    if p.is_file():
        mtime = p.stat().st_mtime
        if mtime != _cache["mtime"]:
            try:
                _cache.update(cams=load_cameras(p, include_disabled=True), error=None)
            except Exception as e:  # noqa: BLE001 — оставляем последнюю рабочую конфигурацию
                log.error("cameras.yml: %s", e)
                _cache["error"] = str(e)
            _cache["mtime"] = mtime
        cams = _cache["cams"]
    else:  # запасной вариант: JSON в переменной CAMERAS
        cams = parse_cameras({"cameras": json.loads(os.getenv("CAMERAS", '[{"id":"cam1","title":"Камера 1"}]'))},
                             include_disabled=True)
    return cams if include_disabled else [c for c in cams if c.enabled]


# ---------- MediaMTX Control API ----------
def mtx_request(method: str, path: str, body: dict | None = None, timeout: float = 3):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(MEDIAMTX_API + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def desired_path_conf(cam: Camera) -> dict:
    if cam.url == "publisher":
        return {"source": "publisher"}
    return {"source": cam.url, "rtspTransport": "tcp", "sourceOnDemand": False}


def sync_mediamtx(cams: list[Camera], request=mtx_request) -> dict:
    """Привести пути MediaMTX к cameras.yml: добавить, обновить, удалить. Идемпотентно."""
    current = {it["name"]: it for it in (request("GET", "/v3/config/paths/list") or {}).get("items", [])}
    want = {c.id: desired_path_conf(c) for c in cams}
    done = {"added": [], "updated": [], "deleted": []}
    for name, conf in want.items():
        if name not in current:
            request("POST", f"/v3/config/paths/add/{name}", conf)
            done["added"].append(name)
        elif any(current[name].get(k) != v for k, v in conf.items()):
            request("PATCH", f"/v3/config/paths/patch/{name}", conf)
            done["updated"].append(name)
    for name in current:
        # служебные пути из mediamtx.yml (all_others, regexp "~...") не трогаем
        if name not in want and name != "all_others" and not name.startswith("~"):
            request("DELETE", f"/v3/config/paths/delete/{name}")
            done["deleted"].append(name)
    return done


def mtx_paths_state() -> dict:
    if not MEDIAMTX_API:
        return {}
    try:
        items = (mtx_request("GET", "/v3/paths/list?itemsPerPage=1000") or {}).get("items", [])
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        log.debug("mediamtx недоступен: %s", e)
        return {}
    return {it["name"]: {"ready": bool(it.get("ready")), "readers": len(it.get("readers") or []),
                         "bytes": it.get("bytesReceived", 0)} for it in items}


async def sync_loop():
    while True:
        try:
            res = await asyncio.to_thread(sync_mediamtx, get_cameras())
            if any(res.values()):
                log.info("mediamtx sync: %s", res)
        except Exception as e:  # noqa: BLE001
            log.warning("mediamtx sync: %s", e)
        await asyncio.sleep(SYNC_INTERVAL)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(sync_loop()) if MEDIAMTX_API else None
    yield
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="ANPR RU", lifespan=lifespan)


# ---------- события ----------
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
    """Поиск по подстроке, латиница -> кириллица, '?' = любой символ, '*' = любая последовательность."""
    q = re.sub(r"\s", "", q.upper()).translate(LAT2CYR)
    if not q:
        return lambda p: True
    rx = re.compile(re.escape(q).replace(r"\?", ".").replace(r"\*", ".*"))
    return lambda p: bool(rx.search(p))


def parse_camera_filter(camera: str) -> set[str] | None:
    ids = {c.strip() for c in camera.split(",") if c.strip()}
    return ids or None


def filtered(q, camera, date_from, date_to):
    m, cams = plate_matcher(q or ""), parse_camera_filter(camera or "")
    for e in reversed(load_events()):
        if cams and e.get("camera") not in cams:
            continue
        if date_from and e["ts"] < date_from:
            continue
        if date_to and e["ts"] > date_to:
            continue
        if m(e["plate"]):
            yield e


def anpr_status() -> dict:
    out = {}
    for path in glob.glob(STATUS_GLOB):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fresh = time.time() - data.get("updated", 0) < STALE_SEC
        for cid, st in (data.get("cameras") or {}).items():
            out[cid] = {**st, "shard": data.get("shard"), "fresh": fresh}
    return out


# ---------- API ----------
@app.get("/api/config")
def config():
    return {"cameras": [c.public() for c in get_cameras()], "config_error": _cache["error"]}


@app.get("/api/cameras")
def cameras_status():
    """Состояние камер: MediaMTX (исходный поток / поток с разметкой) + сервис anpr."""
    mtx, st = mtx_paths_state(), anpr_status()
    now = time.time()
    res = []
    for c in get_cameras():
        s = st.get(c.id) or {}
        working = bool(s.get("fresh") and s.get("alive") and s.get("last_frame") and now - s["last_frame"] < 10)
        res.append({**c.public(),
                    "source": mtx.get(c.raw_path, {"ready": None}),
                    "output": mtx.get(c.anpr_path, {"ready": None}),
                    "processing": {"working": working, "fps": s.get("fps"), "plates": s.get("plates"),
                             "shard": s.get("shard"), "connected": s.get("connected"), "error": s.get("error")}})
    return res


@app.post("/api/cameras/sync")
def cameras_sync():
    if not MEDIAMTX_API:
        raise HTTPException(400, "MEDIAMTX_API не задан")
    try:
        return sync_mediamtx(get_cameras())
    except urllib.error.URLError as e:
        raise HTTPException(502, f"mediamtx: {e}") from e


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
def stats(camera: str = ""):
    cams = parse_camera_filter(camera)
    ev = [e for e in load_events() if not cams or e.get("camera") in cams]
    hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    today = datetime.now().strftime("%Y-%m-%d")
    by_cam: dict[str, int] = {}
    for e in ev:
        if e["ts"].startswith(today):
            by_cam[e.get("camera", "")] = by_cam.get(e.get("camera", ""), 0) + 1
    return {"total": len(ev),
            "today": sum(e["ts"].startswith(today) for e in ev),
            "last_hour": sum(e["ts"] >= hour_ago for e in ev),
            "unique_today": len({e["plate"] for e in ev if e["ts"].startswith(today)}),
            "today_by_camera": by_cam}


@app.get("/api/events.csv")
def events_csv(q: str = "", camera: str = "", date_from: str = "", date_to: str = ""):
    titles = {c.id: c.title for c in get_cameras(include_disabled=True)}
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(["Время", "Камера", "Название камеры", "Номер", "Трек", "Голосов"])
    for e in filtered(q, camera, date_from, date_to):
        cid = e.get("camera", "")
        w.writerow([e["ts"], cid, titles.get(cid, ""), e["plate"], e.get("track", ""), e.get("votes", "")])
    data = "\ufeff" + buf.getvalue()  # BOM, чтобы Excel открыл кириллицу
    return StreamingResponse(iter([data]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=plates.csv"})


@app.get("/api/sse")
async def sse():
    """Хвост events.jsonl как Server-Sent Events. Переживает ротацию файла."""
    async def gen():
        f, ino, idle = None, None, 0
        while True:
            if f is None and EVENTS.exists():
                f = EVENTS.open(encoding="utf-8")
                f.seek(0, os.SEEK_END)
                ino = os.fstat(f.fileno()).st_ino
            line = f.readline() if f else ""
            if line.endswith("\n"):
                idle = 0
                yield f"event: plate\ndata: {line.strip()}\n\n"
                continue
            await asyncio.sleep(0.5)
            idle += 1
            if idle % 30 == 0:
                yield ": ping\n\n"                     # keepalive для прокси
                if f and EVENTS.exists() and EVENTS.stat().st_ino != ino:
                    f.close()
                    f = EVENTS.open(encoding="utf-8")
                    ino = os.fstat(f.fileno()).st_ino
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/snaps", StaticFiles(directory=SNAPS), name="snaps")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
