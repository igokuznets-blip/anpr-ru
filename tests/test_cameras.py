import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from cameras import load_cameras, parse_cameras, parse_only, point_in_roi  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_is_valid():
    cams = load_cameras(ROOT / "cameras.example.yml", include_disabled=True)
    assert [c.id for c in cams] == ["cam1", "cam2", "parking"]
    assert [c.id for c in load_cameras(ROOT / "cameras.example.yml")] == ["cam1"]  # только enabled


def test_defaults_env_and_paths(monkeypatch):
    monkeypatch.setenv("CAM_A", "rtsp://u:p@10.0.0.1/s")
    cams = parse_cameras({"defaults": {"fps": 10}, "cameras": [
        {"id": "a", "url": "${CAM_A}"},
        {"id": "b", "url": "${CAM_B:-publisher}", "fps": 25, "title": "Б"},
    ]})
    a, b = cams
    assert (a.url, a.fps, a.title) == ("rtsp://u:p@10.0.0.1/s", 10, "a")
    assert (b.url, b.fps, b.title) == ("publisher", 25, "Б")
    assert (a.raw_path, a.anpr_path) == ("a", "a_anpr")


def test_public_hides_url():
    pub = parse_cameras({"cameras": [{"id": "c1", "url": "rtsp://admin:secret@cam"}]})[0].public()
    assert "url" not in pub and "secret" not in str(pub)
    assert pub["raw"] == "c1" and pub["anpr"] == "c1_anpr"


@pytest.mark.parametrize("cams,msg", [
    ([{"id": "Cam 1"}], "id"),
    ([{"id": "a"}, {"id": "a"}], "повторяющийся"),
    ([{"id": "x_anpr"}], "_anpr"),
    ([{"id": "a", "roi": [[0, 0], [1, 1]]}], "roi"),
    ([{"id": "a", "roi": [[0, 0], [2, 0], [1, 1]]}], "roi"),
    ([{"id": "a", "foo": 1}], "неизвестные"),
])
def test_validation(cams, msg):
    with pytest.raises(ValueError, match=msg):
        parse_cameras({"cameras": cams})


def test_only_filter():
    data = {"cameras": [{"id": "a"}, {"id": "b"}, {"id": "c", "enabled": False}]}
    assert [c.id for c in parse_cameras(data, only=["b"])] == ["b"]
    with pytest.raises(ValueError, match="не найдены"):
        parse_cameras(data, only=["c"])
    assert parse_only(" a, b ,,") == ["a", "b"] and parse_only("") is None


def test_dir_instead_of_file(tmp_path):
    with pytest.raises(ValueError, match="каталог"):
        load_cameras(tmp_path)


def test_point_in_roi():
    roi = [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]
    assert point_in_roi(0.5, 0.8, roi)
    assert not point_in_roi(0.5, 0.2, roi)
    assert not point_in_roi(0.05, 0.8, roi)
    assert point_in_roi(0.5, 0.2, None)
