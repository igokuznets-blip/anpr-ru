import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

PLATES = ["А123ВС77", "М777ММ199", "Е005КХ50", "Т456УР777"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    ev = tmp_path / "events.jsonl"
    with ev.open("w", encoding="utf-8") as f:
        for i, p in enumerate(PLATES):
            f.write(json.dumps({"id": f"cam1-{i}", "ts": f"2026-01-01T10:0{i}:00", "camera": "cam1",
                                "track": i, "plate": p, "votes": 3}, ensure_ascii=False) + "\n")
    monkeypatch.setenv("EVENTS_FILE", str(ev))
    monkeypatch.setenv("SNAPS_DIR", str(tmp_path / "snaps"))
    import app
    importlib.reload(app)
    return TestClient(app.app)


def test_config(client):
    assert client.get("/api/config").json()["cameras"][0]["id"] == "cam1"


def test_events_order_and_limit(client):
    r = client.get("/api/events?limit=2").json()
    assert [e["plate"] for e in r] == ["Т456УР777", "Е005КХ50"]  # новые сверху


@pytest.mark.parametrize("q,expected", [
    ("a123", ["А123ВС77"]),             # латиница в запросе
    ("?7?7*", ["Т456УР777", "М777ММ199"]),
    ("кх", ["Е005КХ50"]),
    ("", PLATES[::-1]),
])
def test_search(client, q, expected):
    assert [e["plate"] for e in client.get("/api/events", params={"q": q}).json()] == expected


def test_csv(client):
    r = client.get("/api/events.csv")
    assert r.status_code == 200
    body = r.content.decode("utf-8-sig")
    assert body.splitlines()[0].startswith("Время;Камера;Номер")
    assert "М777ММ199" in body


def test_stats(client):
    assert client.get("/api/stats").json()["total"] == len(PLATES)


def test_index(client):
    assert "ANPR" in client.get("/").text
