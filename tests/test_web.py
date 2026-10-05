import importlib
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "common"))

EVENTS = [("cam1", "А123ВС77"), ("cam2", "М777ММ199"), ("cam1", "Е005КХ50"), ("cam2", "Т456УР777")]
CAMERAS_YML = """
cameras:
  - {id: cam1, title: Въезд, url: "rtsp://admin:secret@10.0.0.1/s"}
  - {id: cam2, title: Выезд, url: publisher}
  - {id: cam3, title: Off, enabled: false}
"""


@pytest.fixture()
def web(tmp_path, monkeypatch):
    ev = tmp_path / "events.jsonl"
    with ev.open("w", encoding="utf-8") as f:
        for i, (cam, p) in enumerate(EVENTS):
            f.write(json.dumps({"id": f"{cam}-{i}", "ts": f"2026-01-01T10:0{i}:00", "camera": cam,
                                "track": i, "plate": p, "votes": 3}, ensure_ascii=False) + "\n")
    (tmp_path / "cameras.yml").write_text(CAMERAS_YML, encoding="utf-8")
    (tmp_path / "status-main.json").write_text(json.dumps({"shard": "main", "updated": time.time(), "cameras": {
        "cam1": {"fps": 14.2, "plates": 5, "last_frame": time.time(), "connected": True, "alive": True}}}))
    monkeypatch.setenv("EVENTS_FILE", str(ev))
    monkeypatch.setenv("SNAPS_DIR", str(tmp_path / "snaps"))
    monkeypatch.setenv("CAMERAS_FILE", str(tmp_path / "cameras.yml"))
    monkeypatch.setenv("STATUS_GLOB", str(tmp_path / "status-*.json"))
    monkeypatch.delenv("MEDIAMTX_API", raising=False)
    import app
    importlib.reload(app)
    app._cache.update(mtime=None, cams=[], error=None)
    return app, TestClient(app.app), tmp_path


def test_config_hides_urls_and_disabled(web):
    _, c, _ = web
    r = c.get("/api/config").json()
    assert [x["id"] for x in r["cameras"]] == ["cam1", "cam2"]
    assert "secret" not in json.dumps(r)
    assert r["cameras"][0]["anpr"] == "cam1_anpr"


def test_config_reload_and_error(web):
    _, c, tmp = web
    p = tmp / "cameras.yml"
    p.write_text("cameras:\n  - {id: cam9, title: Новая}\n", encoding="utf-8")
    import os
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert [x["id"] for x in c.get("/api/config").json()["cameras"]] == ["cam9"]
    p.write_text("cameras:\n  - {id: 'BAD ID'}\n", encoding="utf-8")
    os.utime(p, (time.time() + 10, time.time() + 10))
    r = c.get("/api/config").json()
    assert [x["id"] for x in r["cameras"]] == ["cam9"] and r["config_error"]  # остаётся прежняя


def test_cameras_status(web):
    _, c, _ = web
    st = {x["id"]: x for x in c.get("/api/cameras").json()}
    assert st["cam1"]["processing"]["working"] is True and st["cam1"]["processing"]["fps"] == 14.2
    assert st["cam1"]["anpr"] == "cam1_anpr"
    assert st["cam2"]["processing"]["working"] is False


def test_events_order_and_limit(web):
    _, c, _ = web
    assert [e["plate"] for e in c.get("/api/events?limit=2").json()] == ["Т456УР777", "Е005КХ50"]


@pytest.mark.parametrize("params,expected", [
    ({"q": "a123"}, ["А123ВС77"]),
    ({"q": "?7?7*"}, ["Т456УР777", "М777ММ199"]),
    ({"camera": "cam1"}, ["Е005КХ50", "А123ВС77"]),
    ({"camera": "cam1,cam2", "q": "77"}, ["Т456УР777", "М777ММ199", "А123ВС77"]),
])
def test_search(web, params, expected):
    _, c, _ = web
    assert [e["plate"] for e in c.get("/api/events", params=params).json()] == expected


def test_stats_and_csv(web):
    _, c, _ = web
    assert c.get("/api/stats").json()["total"] == 4
    assert c.get("/api/stats?camera=cam2").json()["total"] == 2
    body = c.get("/api/events.csv?camera=cam2").content.decode("utf-8-sig")
    assert body.splitlines()[0].startswith("Время;Камера;Название камеры;Номер")
    assert "Выезд" in body and "А123ВС77" not in body


def test_sync_mediamtx(web):
    app, _, _ = web
    calls = []
    current = {"items": [
        {"name": "all_others", "source": "publisher"},
        {"name": "cam2", "source": "rtsp://old"},                 # изменился
        {"name": "old_cam", "source": "rtsp://x"},                # удалён из конфига
    ]}

    def fake(method, path, body=None):
        calls.append((method, path, body))
        return current if path == "/v3/config/paths/list" else None

    res = app.sync_mediamtx(app.get_cameras(), request=fake)
    assert res == {"added": ["cam1"], "updated": ["cam2"], "deleted": ["old_cam"]}
    assert ("POST", "/v3/config/paths/add/cam1",
            {"source": "rtsp://admin:secret@10.0.0.1/s", "rtspTransport": "tcp", "sourceOnDemand": False}) in calls
    assert ("PATCH", "/v3/config/paths/patch/cam2", {"source": "publisher"}) in calls
    assert not any("all_others" in p for _, p, _ in calls[1:])


def test_index(web):
    _, c, _ = web
    assert "Сетка" in c.get("/").text
