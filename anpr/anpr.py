#!/usr/bin/env python3
"""
ANPR RU: детекция ТС (YOLO) -> детекция номера (YOLO) -> OCR (EasyOCR или CRNN)
-> голосование по треку -> отрисовка рамок -> RTSP (MediaMTX) + events.jsonl + снимки.

Примеры:
  python anpr.py --source rtsp://mediamtx:8554/cam1 --plate-model models/plate.pt --out rtsp://mediamtx:8554/anpr
  python anpr.py --source samples/demo.mp4 --plate-model models/plate.pt --show --device cpu
Все параметры можно задать переменными окружения ANPR_<ПАРАМЕТР> (например ANPR_DEVICE=cpu).
"""
import argparse
import json
import logging
import os
import subprocess
import threading
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from plate_rules import LAT_LETTERS, RU_LETTERS, normalize_plate, pretty

log = logging.getLogger("anpr")
VEH_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]


# ---------- Неблокирующее чтение потока (всегда свежий кадр) ----------
class Reader:
    def __init__(self, src: str):
        self.src = src
        self.live = src.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.frame, self.stopped = None, False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                if self.live:
                    log.warning("поток недоступен, переподключение: %s", self.src)
                    time.sleep(2)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                    continue
                self.stopped = True
                break
            self.frame = f
            if not self.live:
                time.sleep(1 / (self.cap.get(cv2.CAP_PROP_FPS) or 25))

    def read(self):
        return None if self.frame is None else self.frame.copy()


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


def parse_args():
    env = lambda k, d=None: os.getenv(f"ANPR_{k}", d)  # noqa: E731
    ap = argparse.ArgumentParser(description="ANPR RU — распознавание номеров РФ в видеопотоке")
    ap.add_argument("--source", default=env("SOURCE"), help="RTSP URL или видеофайл")
    ap.add_argument("--vehicle-model", default=env("VEHICLE_MODEL", "yolov8n.pt"))
    ap.add_argument("--plate-model", default=env("PLATE_MODEL", "models/plate.pt"))
    ap.add_argument("--ocr", choices=["easyocr", "crnn"], default=env("OCR", "easyocr"))
    ap.add_argument("--crnn-model", default=env("CRNN_MODEL", "models/crnn.pt"))
    ap.add_argument("--out", default=env("OUT"), help="rtsp://mediamtx:8554/anpr")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--camera", default=env("CAMERA", "cam1"))
    ap.add_argument("--events", default=env("EVENTS", "data/events.jsonl"))
    ap.add_argument("--snaps", default=env("SNAPS", "data/snaps"))
    ap.add_argument("--fps", type=int, default=int(env("FPS", 15)))
    ap.add_argument("--device", default=env("DEVICE", "cpu"), help="cpu или индекс GPU")
    ap.add_argument("--min-votes", type=int, default=int(env("MIN_VOTES", 3)))
    ap.add_argument("--min-vehicle-width", type=int, default=int(env("MIN_VEHICLE_WIDTH", 80)))
    a = ap.parse_args()
    if not a.source:
        ap.error("укажите --source или ANPR_SOURCE")
    return a


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    veh, plate = YOLO(a.vehicle_model), YOLO(a.plate_model)
    ocr = CRNNBackend(a.crnn_model, a.device) if a.ocr == "crnn" else EasyOCRBackend(a.device != "cpu")
    font = load_font()

    votes: dict[int, Counter] = defaultdict(Counter)
    confirmed: dict[int, str] = {}
    last_seen: dict[int, float] = {}
    os.makedirs(a.snaps, exist_ok=True)
    os.makedirs(os.path.dirname(a.events) or ".", exist_ok=True)
    events = open(a.events, "a", encoding="utf-8")
    reader, writer = Reader(a.source), None
    log.info("старт: source=%s camera=%s ocr=%s device=%s", a.source, a.camera, a.ocr, a.device)

    while True:
        frame = reader.read()
        if frame is None:
            if reader.stopped:
                break
            time.sleep(0.01)
            continue
        t0 = time.time()
        h, w = frame.shape[:2]
        if a.out and writer is None:
            writer = ffmpeg_writer(a.out, w, h, a.fps)

        res = veh.track(frame, persist=True, classes=VEH_CLASSES, conf=0.35,
                        tracker="bytetrack.yaml", device=a.device, verbose=False)[0]
        labels = []

        if res.boxes.id is not None:
            for (x1, y1, x2, y2), tid in zip(res.boxes.xyxy.int().tolist(), res.boxes.id.int().tolist()):
                last_seen[tid] = t0
                crop = frame[y1:y2, x1:x2].copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
                if tid not in confirmed and crop.size and (x2 - x1) > a.min_vehicle_width:
                    pr = plate(crop, conf=0.4, device=a.device, verbose=False)[0]
                    for px1, py1, px2, py2 in pr.boxes.xyxy.int().tolist()[:1]:
                        cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 255, 0), 2)
                        plate_img = crop[py1:py2, px1:px2]
                        if plate_img.size == 0:
                            continue
                        p = normalize_plate(ocr(plate_img))
                        if not p:
                            continue
                        votes[tid][p] += 1
                        best, n = votes[tid].most_common(1)[0]
                        if n >= a.min_votes:
                            confirmed[tid] = best
                            eid = f"{a.camera}-{tid}-{int(t0 * 1000)}"
                            cv2.imwrite(os.path.join(a.snaps, f"{eid}_plate.jpg"), plate_img)
                            cv2.imwrite(os.path.join(a.snaps, f"{eid}_car.jpg"), crop,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                            ev = {"id": eid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "camera": a.camera,
                                  "track": tid, "plate": best, "votes": n, "bbox": [x1, y1, x2, y2],
                                  "plate_img": f"{eid}_plate.jpg", "car_img": f"{eid}_car.jpg"}
                            events.write(json.dumps(ev, ensure_ascii=False) + "\n")
                            events.flush()
                            log.info("номер %s (трек %s, голосов %s)", best, tid, n)
                text = pretty(confirmed[tid]) if tid in confirmed else f"#{tid}"
                labels.append(((x1, y1), text, (0, 200, 0) if tid in confirmed else (0, 160, 255)))

        for tid in [t for t, ts in last_seen.items() if t0 - ts > 10]:
            last_seen.pop(tid)
            votes.pop(tid, None)
            confirmed.pop(tid, None)

        # подписи кириллицей через PIL (cv2.putText не поддерживает кириллицу)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for (x, y), text, color in labels:
            y = max(y - 28, 0)
            tb = d.textbbox((x, y), text, font=font)
            d.rectangle((tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2), fill=(0, 0, 0))
            d.text((x, y), text, font=font, fill=color[::-1])
        d.text((10, 10), f"{1 / max(time.time() - t0, 1e-3):.1f} FPS", font=font, fill=(255, 255, 0))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if writer:
            try:
                writer.stdin.write(frame.tobytes())
            except BrokenPipeError:
                log.warning("ffmpeg завершился, перезапуск выхода")
                writer = None
        if a.show:
            cv2.imshow("ANPR RU", frame)
            if cv2.waitKey(1) == 27:
                break

    reader.stopped = True
    if writer:
        writer.stdin.close()
        writer.wait()


if __name__ == "__main__":
    main()
