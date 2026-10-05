# ANPR RU — распознавание номеров РФ в видеопотоке

[![CI](https://github.com/OWNER/anpr-ru/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/anpr-ru/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)

Видеоаналитика транспортного потока: детекция автомобилей, поиск и распознавание российских номерных знаков,
отрисовка рамок поверх видео и веб-интерфейс с живым просмотром камеры и журналом распознанных номеров.

![Веб-интерфейс](docs/screenshot.png)

## Возможности

- Детекция ТС (YOLOv8, COCO: легковые, мотоциклы, автобусы, грузовики) и трекинг ByteTrack
- Детекция номерной пластины отдельной YOLO-моделью внутри рамки ТС
- OCR: EasyOCR «из коробки» или собственная CRNN+CTC (скрипт обучения в комплекте)
- Нормализация под формат РФ `А123ВС77` / `А123ВС777`: латиница → кириллица, исправление путаницы `О/0`, `В/8` по позиции
- Голосование по треку — номер подтверждается после N одинаковых чтений
- Поток с разметкой публикуется в MediaMTX → смотреть в браузере (WebRTC / HLS) или VLC (RTSP)
- Веб-интерфейс: живое видео, переключение «с разметкой / оригинал», live-лента номеров (SSE), снимки,
  поиск по маске (`?12?ВС*`, латиница допускается), выгрузка CSV
- Работает в изолированной сети: веса и `hls.js` скачиваются на этапе сборки образов

## Архитектура

```
камера ──RTSP──► MediaMTX (cam1) ──► anpr ──RTSP──► MediaMTX (anpr) ──WebRTC/HLS──┐
                                      │                                            ▼
                                      └─ data/events.jsonl + data/snaps ─► web ─► nginx :8080 ─► браузер
```

| Сервис | Назначение |
|---|---|
| `mediamtx` | Забирает поток камеры один раз, раздаёт в ANPR и браузер (RTSP / WebRTC / HLS) |
| `anpr` | YOLO + OCR, рисует рамки, публикует поток `anpr`, пишет события и снимки |
| `web` | FastAPI: журнал номеров, поиск, CSV, SSE, статика интерфейса |
| `nginx` | Единая точка входа: `/` → web, `/webrtc/` и `/hls/` → MediaMTX |
| `demo` | (профиль `demo`) крутит `samples/demo.mp4` вместо камеры |

## Быстрый старт

```bash
git clone https://github.com/OWNER/anpr-ru.git && cd anpr-ru
cp .env.example .env            # укажите CAM1_URL и WEBRTC_HOST
# положите модель детекции номера в models/plate.pt (см. «Обучение»)
docker compose up -d --build    # CPU
# или GPU (NVIDIA Container Toolkit):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Интерфейс: `http://<хост>:8080` · RTSP с разметкой: `rtsp://<хост>:8554/anpr`

Демо без камеры: положите любое видео с машинами в `samples/demo.mp4` и выполните `make demo`.

Без Docker:

```bash
pip install -r anpr/requirements.txt
python anpr/anpr.py --source samples/demo.mp4 --plate-model models/plate.pt --show --device cpu
```

## Настройка

Переменные `.env`:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CAM1_URL` | — | RTSP-адрес камеры |
| `CAM1_TITLE` | `Камера 1` | Название в интерфейсе |
| `WEBRTC_HOST` | `127.0.0.1` | IP хоста, видимый браузерам (обязательно для WebRTC) |
| `HTTP_PORT` / `RTSP_PORT` | `8080` / `8554` | Внешние порты |
| `ANPR_OCR` | `easyocr` | `easyocr` или `crnn` |
| `ANPR_DEVICE` | `cpu` | `cpu` или индекс GPU |
| `ANPR_FPS` | `15` | FPS выходного потока |
| `ANPR_MIN_VOTES` | `3` | Сколько одинаковых чтений нужно для подтверждения |

Все параметры `anpr.py` доступны как `--ключ` или `ANPR_<КЛЮЧ>` (`python anpr/anpr.py --help`).

Порты: `8080/tcp` — интерфейс, `8554/tcp` — RTSP, `8189/udp` — медиа WebRTC. Если UDP закрыт,
переключите плеер на HLS (задержка ~1–2 с).

### Несколько камер

1. Добавьте пути `cam2`, `anpr2` в `mediamtx.yml` (или `MTX_PATHS_CAM2_SOURCE`).
2. Скопируйте сервис `anpr` в `docker-compose.yml` с `ANPR_SOURCE=.../cam2`, `ANPR_OUT=.../anpr2`, `ANPR_CAMERA=cam2`.
3. Добавьте камеру в `CAMERAS` сервиса `web`: `{"id":"cam2","title":"...","raw":"cam2","anpr":"anpr2"}`.

## API

| Метод | Описание |
|---|---|
| `GET /api/config` | Список камер |
| `GET /api/events?q=&camera=&date_from=&date_to=&limit=` | Журнал номеров (новые сверху) |
| `GET /api/events.csv` | То же в CSV (`;`, UTF-8 BOM для Excel) |
| `GET /api/stats` | Счётчики: за час, сегодня, уникальных, всего |
| `GET /api/sse` | Server-Sent Events, событие `plate` на каждый новый номер |
| `GET /snaps/<file>` | Снимки номера и ТС |

Формат события (`data/events.jsonl`):

```json
{"id":"cam1-12-1759671234567","ts":"2026-10-05T18:47:00","camera":"cam1","track":12,
 "plate":"А123ВС77","votes":3,"bbox":[410,220,780,560],
 "plate_img":"cam1-12-1759671234567_plate.jpg","car_img":"cam1-12-1759671234567_car.jpg"}
```

## Обучение

Используются открытые датасеты на основе AUTO.RIA / Nomeroff Net (CC BY 4.0):

| Датасет | Объём | Для чего |
|---|---|---|
| [Car_plate_detecting_dataset](https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset) | 25 632 изобр., YOLO | Детекция номера → `models/plate.pt` |
| [Car_plate_OCR_dataset](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset) | 45 514 номеров | OCR (CRNN) → `models/crnn.pt` |

```bash
pip install ultralytics torch opencv-python-headless
make datasets        # скачать и подготовить оба датасета в datasets/
make train-plate     # YOLOv8s, 100 эпох -> models/plate.pt
make train-ocr       # CRNN+CTC, 30 эпох -> models/crnn.pt, затем ANPR_OCR=crnn
```

Совет: дообучите детектор на 300–1000 кадрах с ваших камер (ракурс, ночь, ИК) — это даёт наибольший прирост.

Для детектора ТС на ракурсе дорожных камер можно дополнительно использовать
[UA-DETRAC](https://ar5iv.labs.arxiv.org/html/1511.04136) или [MIO-TCD](https://tcd.miovision.com/).

## Разработка

```bash
pip install -r requirements-dev.txt
make test    # pytest: правила номеров и API
make lint    # ruff
```

CI (GitHub Actions) запускает линтер и тесты, собирает образы `web` и `anpr` и публикует их в
GitHub Container Registry: `ghcr.io/OWNER/anpr-ru/web`, `ghcr.io/OWNER/anpr-ru/anpr`
(ветка `main` и теги `v*`).

## Структура

```
anpr/            сервис распознавания (anpr.py, plate_rules.py, crnn.py, Dockerfile)
web/             FastAPI + веб-интерфейс (static/index.html)
nginx/           reverse proxy
tools/           подготовка датасетов и обучение
tests/           pytest
models/          веса (не в git)
data/            события и снимки (не в git)
samples/         демо-видео (не в git)
```

## Лицензия

[AGPL-3.0](LICENSE) — проект использует [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
Для закрытого коммерческого использования нужна лицензия Ultralytics Enterprise либо замена детектора.
Датасеты распространяются под CC BY 4.0 — сохраняйте указание авторства.

Распознавание номеров — обработка персональных данных: соблюдайте требования 152-ФЗ при эксплуатации.
