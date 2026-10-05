#!/usr/bin/env python3
"""
Обучение CRNN+CTC на датасете номеров (имя файла = номер латиницей) -> models/crnn.pt

  python tools/prepare_datasets.py ocr
  python tools/train_crnn.py --data datasets/ocr --epochs 30 --device cuda
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "anpr"))
from crnn import CRNN, IMG_H, IMG_W, greedy_decode  # noqa: E402
from plate_rules import OCR_ALPHABET  # noqa: E402

CHAR2IDX = {c: i + 1 for i, c in enumerate(OCR_ALPHABET)}  # 0 = blank


class PlateDS(Dataset):
    def __init__(self, folder: Path, augment: bool):
        self.files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        label = p.stem.split("_")[0]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            h, w = img.shape
            m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), random.uniform(0.9, 1.05))
            img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.6, 1.4), beta=random.uniform(-40, 40))
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        x = torch.from_numpy((img - 0.5) / 0.5)[None]
        y = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return x, y, label


def collate(b):
    xs, ys, labels = zip(*b)
    return torch.stack(xs), torch.cat(ys), torch.tensor([len(y) for y in ys]), list(labels)


def evaluate(model, dl, dev):
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for x, _, _, labels in dl:
            pred = greedy_decode(model(x.to(dev)))
            ok += sum(p == t for p, t in zip(pred, labels))
            n += len(labels)
    return ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/ocr", type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="models/crnn.pt")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    tr = DataLoader(PlateDS(a.data / "train", True), a.batch, shuffle=True,
                    num_workers=a.workers, collate_fn=collate, drop_last=True)
    va = DataLoader(PlateDS(a.data / "val", False), a.batch, num_workers=a.workers, collate_fn=collate)
    model = CRNN().to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tr))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    best = -1.0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y, ylen, _ in tr:
            x = x.to(a.device)
            logp = model(x).log_softmax(-1).permute(1, 0, 2)  # T,B,C
            tlen = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp, y, tlen, ylen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            sched.step()
            total += loss.item()
        acc = evaluate(model, va, a.device)
        print(f"эпоха {ep:3d}  loss {total / len(tr):.4f}  точность номера (val) {acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), a.out)
    test = a.data / "test"
    if test.exists():
        model.load_state_dict(torch.load(a.out, map_location=a.device))
        te = DataLoader(PlateDS(test, False), a.batch, collate_fn=collate)
        print(f"test: точность {evaluate(model, te, a.device):.4f}")
    print(f"готово: {a.out} (лучшая val {best:.4f})")


if __name__ == "__main__":
    main()
