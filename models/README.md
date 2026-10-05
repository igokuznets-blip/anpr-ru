# Модели

Веса не хранятся в git. Положите сюда:

| Файл | Что это | Как получить |
|---|---|---|
| `plate.pt` | YOLO-детектор номерной пластины (1 класс) | `make train-plate` или `tools/train_plate.sh` |
| `crnn.pt` | CRNN для OCR (опционально, при `ANPR_OCR=crnn`) | `make train-ocr` или `tools/train_crnn.py` |

`yolov8n.pt` (детектор ТС, COCO) скачивается автоматически при сборке образа.
