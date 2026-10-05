"""
Конфигурация камер (cameras.yml) — общий модуль для сервисов anpr и web.

cameras.yml:
  defaults:                 # необязательно, применяется ко всем камерам
    fps: 15
  cameras:
    - id: cam1              # [a-z0-9_-], используется как путь MediaMTX
      title: Въезд КПП-1
      url: ${CAM1_URL}      # RTSP камеры, ${VAR} / ${VAR:-default} из окружения; "publisher" — поток публикуют извне
      roi: [[0.1,0.4],[0.9,0.4],[0.9,1],[0.1,1]]   # необязательно: полигон зоны распознавания (доли кадра)
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
ANPR_SUFFIX = "_anpr"
FIELDS = {"id", "title", "url", "enabled", "roi", "fps", "min_vehicle_width", "min_votes"}


@dataclass
class Camera:
    id: str
    title: str
    url: str = "publisher"
    enabled: bool = True
    roi: list[list[float]] | None = None
    fps: int | None = None
    min_vehicle_width: int | None = None
    min_votes: int | None = None

    @property
    def raw_path(self) -> str:
        """Путь MediaMTX с исходным потоком."""
        return self.id

    @property
    def anpr_path(self) -> str:
        """Путь MediaMTX с потоком с разметкой."""
        return self.id + ANPR_SUFFIX

    def public(self) -> dict:
        """Описание для браузера — без URL камеры (там логин/пароль)."""
        d = asdict(self)
        d.pop("url")
        d.update(raw=self.raw_path, anpr=self.anpr_path)
        return d


def expand_env(value):
    if isinstance(value, str):
        return ENV_RE.sub(lambda m: os.environ.get(m[1], m[2] if m[2] is not None else ""), value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


def _validate_roi(cam_id: str, roi):
    if roi is None:
        return None
    ok = isinstance(roi, list) and len(roi) >= 3 and all(
        isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in p)
        for p in roi)
    if not ok:
        raise ValueError(f"камера {cam_id}: roi — список из ≥3 точек [x, y] в долях кадра 0..1")
    return [[float(x), float(y)] for x, y in roi]


def parse_cameras(data: dict, only: list[str] | None = None, include_disabled: bool = False) -> list[Camera]:
    data = expand_env(data or {})
    defaults = data.get("defaults") or {}
    cams, seen = [], set()
    for i, raw in enumerate(data.get("cameras") or []):
        c = {**defaults, **(raw or {})}
        cid = str(c.get("id", ""))
        if not ID_RE.match(cid):
            raise ValueError(f"камера #{i + 1}: id '{cid}' — только a-z, 0-9, '_' и '-', до 32 символов")
        if cid.endswith(ANPR_SUFFIX):
            raise ValueError(f"камера {cid}: id не должен оканчиваться на '{ANPR_SUFFIX}'")
        if cid in seen:
            raise ValueError(f"камера {cid}: повторяющийся id")
        unknown = set(c) - FIELDS
        if unknown:
            raise ValueError(f"камера {cid}: неизвестные поля {sorted(unknown)}")
        seen.add(cid)
        cam = Camera(id=cid, title=str(c.get("title") or cid), url=str(c.get("url") or "publisher"),
                     enabled=bool(c.get("enabled", True)), roi=_validate_roi(cid, c.get("roi")),
                     fps=c.get("fps"), min_vehicle_width=c.get("min_vehicle_width"), min_votes=c.get("min_votes"))
        if cam.enabled or include_disabled:
            cams.append(cam)
    if only:
        missing = set(only) - {c.id for c in cams}
        if missing:
            raise ValueError(f"камеры не найдены или отключены: {sorted(missing)}")
        cams = [c for c in cams if c.id in only]
    return cams


def load_cameras(path: str | Path, only: list[str] | None = None, include_disabled: bool = False) -> list[Camera]:
    p = Path(path)
    if p.is_dir():
        raise ValueError(f"{p} — каталог. Создайте файл: cp cameras.example.yml cameras.yml")
    return parse_cameras(yaml.safe_load(p.read_text(encoding="utf-8")), only, include_disabled)


def parse_only(value: str | None) -> list[str] | None:
    items = [s.strip() for s in (value or "").split(",") if s.strip()]
    return items or None


def point_in_roi(x: float, y: float, roi: list[list[float]] | None) -> bool:
    """Точка (в долях кадра) внутри полигона? Нет ROI — весь кадр."""
    if not roi:
        return True
    inside = False
    j = len(roi) - 1
    for i in range(len(roi)):
        xi, yi = roi[i]
        xj, yj = roi[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
