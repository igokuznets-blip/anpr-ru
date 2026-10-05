#!/usr/bin/env bash
# Обучение YOLO-детектора номерной пластины -> models/plate.pt
#   tools/train_plate.sh [model=yolov8s.pt] [epochs=100] [imgsz=640] [device=0]
set -euo pipefail
MODEL=${1:-yolov8s.pt}; EPOCHS=${2:-100}; IMGSZ=${3:-640}; DEVICE=${4:-0}
cd "$(dirname "$0")/.."
[ -f datasets/plates.yaml ] || python tools/prepare_datasets.py plates
yolo detect train data=datasets/plates.yaml model="$MODEL" epochs="$EPOCHS" imgsz="$IMGSZ" \
     device="$DEVICE" batch=-1 patience=20 project=runs name=plate exist_ok=True \
     hsv_v=0.5 degrees=5 perspective=0.0005 mosaic=1.0
mkdir -p models && cp runs/plate/weights/best.pt models/plate.pt
yolo detect val model=models/plate.pt data=datasets/plates.yaml split=test device="$DEVICE"
echo "готово: models/plate.pt"
