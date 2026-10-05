#!/usr/bin/env python3
"""
ANPR RU: детекция ТС (YOLO) -> детекция номера (YOLO) -> OCR (EasyOCR или CRNN)
-> голосование по треку -> отрисовка рамок -> RTSP (MediaMTX) + events.jsonl + снимки.

Несколько камер (cameras.yml): на каждую камеру — свой поток чтения, свой детектор ТС с трекером
и свой выходной RTSP; модель номера и OCR общие (одна копия в памяти GPU).

  # все включённые камеры из cameras.yml
  python anpr.py --cameras-file cameras.yml --source-template rtsp://mediamtx:8554/{id} \
                 --out-template rtsp://mediamtx:8554/{id}_anpr
  # шардирование: этот экземпляр обрабатывает только часть камер
  python anpr.py --cameras-file cameras.yml --only cam1,cam2 --shard gpu0
  # одна камера / файл без cameras.yml
  python anpr.py --source samples/demo.mp4 --show --device cpu

Все параметры можно задать переменными окружения ANPR_<ПАРАМЕТР> (например ANPR_DEVICE=cpu).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from plate_rules import LAT_LETTERS, RU_LETTERS, normalize_plate, pretty

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))  # локальный запуск из репозитория
from cameras import Camera, load_cameras, parse_only, point_in_roi  # noqa: E402

log = logging.getLogger("anpr")
VEH_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
STOP = threading.Event()


# ---------- Неблокирующее чтение потока (всегда свежий кадр) ----------
class Reader:
    def __init__(self, src: str, name: str):
        self.src, self.name = src, name
        self.live = src.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.frame, self.seq, self.connected, self.ended = None, 0, False, False
        self.cap = None
        threading.Thread(target=self._loop, daemon=True, name=f"reader-{name}").start()

    def _open(self):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)

    def _loop(self):
        self._open()
        while not STOP.is_set():
            ok, f = self.cap.read()
            if not ok:
                self.connected = False
                if self.live:
                    log.warning("[%s] поток недоступен, переподключение через 3 с", self.name)
                    STOP.wait(3)
                    self.cap.release()
                    self._open()
                    continue
                self.ended = True
                break
            self.frame, self.connected = f, True
            self.seq += 1
            if not self.live:
                time.sleep(1 / (self.cap.get(cv2.CAP_PROP_FPS) or 25))
        self.cap.release()

    def read(self):
        return self.seq, (None if self.frame is None else self.frame.copy())


# ---------- OCR-бэкенды ----------
class EasyOCRBackend:
    def __init__(self, gpu: bool):
        import easyocr
        self.r = easyocr.Reader(["ru"], gpu=gpu, verbose=False)

    def __call__(self, plate_bgr) -> str:
        pc = cv2.resize(plate_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY))
        return "".join(self.r.readtext(g, detail=0, paragraph=True,
                                       allowlist=RU_LETTERS + LAT_LETTERS + "0123456789"))


class CRNNBackend:
    def __init__(self, path: str, device: str):
        import torch

        from crnn import CRNN, greedy_decode, preprocess
        self.torch, self.decode, self.pre = torch, greedy_decode, preprocess
        self.dev = "cpu" if device == "cpu" else f"cuda:{device}"
        self.m = CRNN().to(self.dev).eval()
        self.m.load_state_dict(torch.load(path, map_location=self.dev))

    def __call__(self, plate_bgr) -> str:
        with self.torch.no_grad():
            x = self.pre(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)).to(self.dev)
            return self.decode(self.m(x))[0]


class PlateRecognizer:
    """Общие для всех камер детектор номера + OCR. Вызовы сериализуются блокировкой."""

    def __init__(self, a):
        self.model = YOLO(a.plate_model)
        self.ocr = CRNNBackend(a.crnn_model, a.device) if a.ocr == "crnn" else EasyOCRBackend(a.device != "cpu")
        self.device = a.device
        self.lock = threading.Lock()

    def detect(self, crop):
        with self.lock:
            return self.model(crop, conf=0.4, device=self.device, verbose=False)[0].boxes.xyxy.int().tolist()

    def read(self, plate_img) -> str:
        with self.lock:
            return self.ocr(plate_img)


class EventSink:
    """events.jsonl + снимки; одна запись на подтверждённый номер."""

    def __init__(self, events_path: str, snaps_dir: str):
        os.makedirs(snaps_dir, exist_ok=True)
        os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
        self.f = open(events_path, "a", encoding="utf-8")
        self.snaps = snaps_dir
        self.lock = threading.Lock()

    def emit(self, cam: Camera, tid: int, plate: str, votes: int, bbox, plate_img, car_img):
        eid = f"{cam.id}-{tid}-{int(time.time() * 1000)}"
        cv2.imwrite(os.path.join(self.snaps, f"{eid}_plate.jpg"), plate_img)
        cv2.imwrite(os.path.join(self.snaps, f"{eid}_car.jpg"), car_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ev = {"id": eid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "camera": cam.id, "track": tid,
              "plate": plate, "votes": votes, "bbox": bbox,
              "plate_img": f"{eid}_plate.jpg", "car_img": f"{eid}_car.jpg"}
        with self.lock:
            self.f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self.f.flush()
        log.info("[%s] номер %s (трек %s, голосов %s)", cam.id, plate, tid, votes)


# ---------- Выход в RTSP через ffmpeg -> MediaMTX ----------
def ffmpeg_writer(url: str, w: int, h: int, fps: int):
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
           "-g", str(fps * 2), "-pix_fmt", "yuv420p", "-f", "rtsp", "-rtsp_transport", "tcp", url]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def load_font(size=22):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ---------- Обработчик одной камеры ----------
class CameraWorker(threading.Thread):
    def __init__(self, cam: Camera, src: str, out: str | None, a, plates: PlateRecognizer, sink: EventSink, font):
        super().__init__(daemon=True, name=f"cam-{cam.id}")
        self.cam, self.src, self.out, self.a = cam, src, out, a
        self.plates, self.sink, self.font = plates, sink, font
        self.fps_out = cam.fps or a.fps
        self.min_votes = cam.min_votes or a.min_votes
        self.min_w = cam.min_vehicle_width or a.min_vehicle_width
        self.vehicles = YOLO(a.vehicle_model)        # отдельный экземпляр = отдельный трекер
        self.reader = Reader(src, cam.id)
        self.view = None                             # последний кадр с разметкой (для --show)
        self.stats = {"fps": 0.0, "plates": 0, "last_frame": None, "started": time.time(), "error": None}

    def status(self) -> dict:
        return {**self.stats, "connected": self.reader.connected, "alive": self.is_alive()}

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001 — камера не должна ронять остальные
            log.exception("[%s] ошибка обработчика", self.cam.id)
            self.stats["error"] = str(e)

    def _run(self):
        votes: dict[int, Counter] = defaultdict(Counter)
        confirmed: dict[int, str] = {}
        last_seen: dict[int, float] = {}
        writer, last_seq, ema = None, -1, None
        log.info("[%s] старт: %s -> %s", self.cam.id, self.src, self.out or "—")

        while not STOP.is_set():
            seq, frame = self.reader.read()
            if frame is None or seq == last_seq:
                if self.reader.ended:
                    break
                time.sleep(0.005)
                continue
            last_seq = seq
            t0 = time.time()
            h, w = frame.shape[:2]
            if self.out and writer is None:
                writer = ffmpeg_writer(self.out, w, h, self.fps_out)
            roi = self.cam.roi
            if roi:
                pts = np.array([[int(x * w), int(y * h)] for x, y in roi], np.int32)
                cv2.polylines(frame, [pts], True, (255, 255, 0), 1)

            res = self.vehicles.track(frame, persist=True, classes=VEH_CLASSES, conf=0.35,
                                      tracker="bytetrack.yaml", device=self.a.device, verbose=False)[0]
            labels = []
            if res.boxes.id is not None:
                for (x1, y1, x2, y2), tid in zip(res.boxes.xyxy.int().tolist(), res.boxes.id.int().tolist()):
                    last_seen[tid] = t0
                    in_roi = point_in_roi((x1 + x2) / 2 / w, y2 / h, roi)   # нижняя середина рамки ТС
                    crop = frame[y1:y2, x1:x2].copy()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0) if in_roi else (128, 128, 128), 2)
                    if in_roi and tid not in confirmed and crop.size and (x2 - x1) > self.min_w:
                        self._read_plate(frame, crop, x1, y1, x2, y2, tid, votes, confirmed)
                    text = pretty(confirmed[tid]) if tid in confirmed else f"#{tid}"
                    labels.append(((x1, y1), text, (0, 200, 0) if tid in confirmed else (0, 160, 255)))

            for tid in [t for t, ts in last_seen.items() if t0 - ts > 10]:
                last_seen.pop(tid)
                votes.pop(tid, None)
                confirmed.pop(tid, None)

            dt = max(time.time() - t0, 1e-3)
            ema = 1 / dt if ema is None else 0.9 * ema + 0.1 / dt
            self.stats.update(fps=round(ema, 1), last_frame=time.time())
            frame = self._draw_labels(frame, labels, f"{self.cam.title} · {ema:.1f} FPS")

            if writer:
                try:
                    writer.stdin.write(frame.tobytes())
                except (BrokenPipeError, ValueError):
                    log.warning("[%s] ffmpeg завершился, перезапуск выхода", self.cam.id)
                    writer = None
            if self.a.show:
                self.view = frame

        if writer:
            writer.stdin.close()
            writer.wait()

    def _read_plate(self, frame, crop, x1, y1, x2, y2, tid, votes, confirmed):
        for px1, py1, px2, py2 in self.plates.detect(crop)[:1]:
            cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 255, 0), 2)
            plate_img = crop[py1:py2, px1:px2]
            if plate_img.size == 0:
                continue
            p = normalize_plate(self.plates.read(plate_img))
            if not p:
                continue
            votes[tid][p] += 1
            best, n = votes[tid].most_common(1)[0]
            if n >= self.min_votes:
                confirmed[tid] = best
                self.stats["plates"] += 1
                self.sink.emit(self.cam, tid, best, n, [x1, y1, x2, y2], plate_img, crop)

    def _draw_labels(self, frame, labels, header):
        # кириллица через PIL (cv2.putText её не поддерживает)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for (x, y), text, color in labels:
            y = max(y - 28, 0)
            tb = d.textbbox((x, y), text, font=self.font)
            d.rectangle((tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2), fill=(0, 0, 0))
            d.text((x, y), text, font=self.font, fill=color[::-1])
        tb = d.textbbox((10, 10), header, font=self.font)
        d.rectangle((tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2), fill=(0, 0, 0))
        d.text((10, 10), header, font=self.font, fill=(255, 255, 0))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def write_status(path: str, shard: str, workers: list[CameraWorker]):
    """data/status-<shard>.json — читается веб-интерфейсом."""
    data = {"shard": shard, "updated": time.time(), "cameras": {w.cam.id: w.status() for w in workers}}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def parse_args():
    env = lambda k, d=None: os.getenv(f"ANPR_{k}", d)  # noqa: E731
    ap = argparse.ArgumentParser(description="ANPR RU — распознавание номеров РФ в видеопотоках")
    g = ap.add_argument_group("камеры")
    g.add_argument("--cameras-file", default=env("CAMERAS_FILE"), help="cameras.yml")
    g.add_argument("--only", default=env("ONLY"), help="обрабатывать только эти id через запятую (шард)")
    g.add_argument("--source-template", default=env("SOURCE_TEMPLATE", "{url}"),
                   help="откуда читать: {url} — адрес камеры напрямую, или rtsp://mediamtx:8554/{id}")
    g.add_argument("--out-template", default=env("OUT_TEMPLATE"), help="rtsp://mediamtx:8554/{id}_anpr")
    g.add_argument("--source", default=env("SOURCE"), help="одна камера: RTSP URL или файл (без cameras.yml)")
    g.add_argument("--out", default=env("OUT"), help="выход для --source")
    g.add_argument("--camera", default=env("CAMERA", "cam1"), help="id для --source")
    ap.add_argument("--vehicle-model", default=env("VEHICLE_MODEL", "yolov8n.pt"))
    ap.add_argument("--plate-model", default=env("PLATE_MODEL", "models/plate.pt"))
    ap.add_argument("--ocr", choices=["easyocr", "crnn"], default=env("OCR", "easyocr"))
    ap.add_argument("--crnn-model", default=env("CRNN_MODEL", "models/crnn.pt"))
    ap.add_argument("--show", action="store_true", help="окна OpenCV")
    ap.add_argument("--events", default=env("EVENTS", "data/events.jsonl"))
    ap.add_argument("--snaps", default=env("SNAPS", "data/snaps"))
    ap.add_argument("--status-dir", default=env("STATUS_DIR", "data"))
    ap.add_argument("--shard", default=env("SHARD", "main"), help="имя экземпляра (файл статуса)")
    ap.add_argument("--fps", type=int, default=int(env("FPS", 15)))
    ap.add_argument("--device", default=env("DEVICE", "cpu"), help="cpu или индекс GPU")
    ap.add_argument("--min-votes", type=int, default=int(env("MIN_VOTES", 3)))
    ap.add_argument("--min-vehicle-width", type=int, default=int(env("MIN_VEHICLE_WIDTH", 80)))
    a = ap.parse_args()
    if not a.source and not a.cameras_file:
        ap.error("укажите --cameras-file (ANPR_CAMERAS_FILE) или --source (ANPR_SOURCE)")
    return a


def resolve_cameras(a) -> list[tuple[Camera, str, str | None]]:
    if a.source:
        return [(Camera(id=a.camera, title=a.camera, url=a.source), a.source, a.out)]
    cams = load_cameras(a.cameras_file, only=parse_only(a.only))
    if not cams:
        raise SystemExit("в cameras.yml нет включённых камер")
    out = []
    for c in cams:
        src = a.source_template.format(id=c.id, url=c.url)
        if src == "publisher":
            raise SystemExit(f"камера {c.id}: url=publisher — читайте через MediaMTX "
                             f"(--source-template rtsp://mediamtx:8554/{{id}})")
        out.append((c, src, a.out_template.format(id=c.id) if a.out_template else None))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(threadName)s %(message)s")
    a = parse_args()
    cams = resolve_cameras(a)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: STOP.set())

    log.info("камер: %d (%s), ocr=%s, device=%s, shard=%s",
             len(cams), ", ".join(c.id for c, _, _ in cams), a.ocr, a.device, a.shard)
    plates, sink, font = PlateRecognizer(a), EventSink(a.events, a.snaps), load_font()
    workers = [CameraWorker(c, src, out, a, plates, sink, font) for c, src, out in cams]
    for w in workers:
        w.start()

    os.makedirs(a.status_dir, exist_ok=True)
    status_path = os.path.join(a.status_dir, f"status-{a.shard}.json")
    next_status = 0.0
    while not STOP.is_set() and any(w.is_alive() for w in workers):
        if time.time() >= next_status:
            write_status(status_path, a.shard, workers)
            next_status = time.time() + 2
        if a.show:  # GUI только из главного потока
            for w in workers:
                if w.view is not None:
                    cv2.imshow(f"ANPR {w.cam.id}", w.view)
            if cv2.waitKey(30) == 27:
                STOP.set()
        else:
            STOP.wait(0.5)

    STOP.set()
    for w in workers:
        w.join(timeout=5)
    write_status(status_path, a.shard, workers)


if __name__ == "__main__":
    main()
