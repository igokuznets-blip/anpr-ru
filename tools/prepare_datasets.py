#!/usr/bin/env python3
"""
Скачивает и готовит открытые датасеты номеров РФ (Hugging Face, CC BY 4.0, основаны на AUTO.RIA / Nomeroff Net):

  plates  — AY000554/Car_plate_detecting_dataset (~25.6K изобр., YOLO)  -> datasets/plates + plates.yaml
  ocr     — AY000554/Car_plate_OCR_dataset      (~45.5K номеров)       -> datasets/ocr/{train,val,test}

  python tools/prepare_datasets.py plates
  python tools/prepare_datasets.py ocr
  python tools/prepare_datasets.py all --out datasets
"""
import argparse
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
REPOS = {"plates": "AY000554/Car_plate_detecting_dataset", "ocr": "AY000554/Car_plate_OCR_dataset"}
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OCR_RE = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$")


def download(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  есть: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  скачивание {url}")

    def hook(n, bs, total):
        if total > 0:
            sys.stdout.write(f"\r  {min(n * bs, total) / 1e6:8.1f} / {total / 1e6:.1f} МБ")
            sys.stdout.flush()

    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dst)
    print()


def fetch_split(repo: str, split: str, cache: Path) -> Path:
    z = cache / repo.replace("/", "__") / f"{split}.zip"
    download(HF.format(repo=repo, file=f"{split}.zip"), z)
    ex = z.with_suffix("")
    if not ex.exists():
        print(f"  распаковка {z.name}")
        with zipfile.ZipFile(z) as f:
            f.extractall(ex)
    return ex


def prepare_plates(out: Path, cache: Path):
    root = out / "plates"
    for split in SPLITS:
        src = fetch_split(REPOS["plates"], split, cache)
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        n = 0
        for img in src.rglob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = next((p for p in [img.with_suffix(".txt"),
                                    Path(str(img.with_suffix(".txt")).replace("/images/", "/labels/"))]
                        if p.exists()), None)
            if lbl is None:
                continue
            shutil.copy2(img, root / split / "images" / img.name)
            # все классы -> 0 (plate)
            lines = [" ".join(["0"] + ln.split()[1:]) for ln in lbl.read_text().splitlines() if ln.strip()]
            (root / split / "labels" / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
            n += 1
        print(f"  {split}: {n} изображений")
    yaml = out / "plates.yaml"
    yaml.write_text(f"path: {root.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                    f"names:\n  0: plate\n")
    print(f"готово: {yaml}")


def prepare_ocr(out: Path, cache: Path):
    root = out / "ocr"
    for split in SPLITS:
        src = fetch_split(REPOS["ocr"], split, cache)
        dst = root / split
        dst.mkdir(parents=True, exist_ok=True)
        n = bad = 0
        for img in src.rglob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            label = img.stem.split("_")[0].upper()
            if not OCR_RE.match(label):
                bad += 1
                continue
            name = f"{label}{img.suffix.lower()}"
            k = 1
            while (dst / name).exists():
                name = f"{label}_{k}{img.suffix.lower()}"
                k += 1
            shutil.copy2(img, dst / name)
            n += 1
        print(f"  {split}: {n} номеров (пропущено {bad})")
    print(f"готово: {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plates", "ocr", "all"])
    ap.add_argument("--out", default="datasets", type=Path)
    ap.add_argument("--cache", default="datasets/.cache", type=Path)
    a = ap.parse_args()
    if a.what in ("plates", "all"):
        print("== детекция номеров")
        prepare_plates(a.out, a.cache)
    if a.what in ("ocr", "all"):
        print("== OCR номеров")
        prepare_ocr(a.out, a.cache)


if __name__ == "__main__":
    main()
