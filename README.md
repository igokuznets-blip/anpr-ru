# ANPR RU — распознавание номеров РФ в видеопотоке

[![CI](https://github.com/OWNER/anpr-ru/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/anpr-ru/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)

Видеоаналитика транспортного потока: детекция автомобилей, поиск и распознавание российских номерных знаков,
отрисовка рамок поверх видео и веб-интерфейс с живым просмотром камер (одна или сеткой) и журналом распознанных номеров.
Поддерживается любое число камер: один файл `cameras.yml`, шардирование по GPU/хостам.

![Веб-интерфейс](docs/screenshot.png)

## Возможности

- Несколько камер из `cameras.yml`: свой трекер и выходной поток на камеру, общие модели номера/OCR,
  зона распознавания (ROI) и параметры для каждой камеры, шардирование между экземплярами
- Детекция ТС (YOLOv8, COCO: легковые, мотоциклы, автобусы, грузовики) и трекинг ByteTrack
- Детекция номерной пластины отдельной YOLO-моделью внутри рамки ТС
- OCR: EasyOCR «из коробки» или собственная CRNN+CTC (скрипт обучения в комплекте)
- Нормализация под формат РФ `А123ВС77` / `А123ВС777`: латиница → кириллица, исправление путаницы `О/0`, `В/8` по позиции
- Голосование по треку — номер подтверждается после N одинаковых чтений
- Поток с разметкой публикуется в MediaMTX → смотреть в браузере (WebRTC / HLS) или VLC (RTSP)
- Веб-интерфейс: одна камера или сетка всех камер, «с разметкой / оригинал», состояние каждой камеры
  (сигнал, FPS, экземпляр anpr), live-лента номеров (SSE) с фильтром по камере, снимки,
  поиск по маске (`?12?ВС*`, латиница допускается), выгрузка CSV
- Пути MediaMTX создаются автоматически из `cameras.yml` (Control API), правки подхватываются без перезапуска
- Работает в изолированной сети: веса и `hls.js` скачиваются на этапе сборки образов

## Архитектура

```
камеры ─RTSP─► MediaMTX <id> ─► anpr (поток на камеру) ─RTSP─► MediaMTX <id>_anpr ─WebRTC/HLS─┐
                  ▲               │                                                            ▼
   cameras.yml ─► web ─(API)──────┘── events.jsonl, snaps/, status-<shard>.json ─► web ─► nginx :8080 ─► браузер
```

| Сервис | Назначение |
|---|---|
| `mediamtx` | Забирает поток каждой камеры один раз, раздаёт в ANPR и браузер (RTSP / WebRTC / HLS) |
| `anpr` | YOLO + OCR для всех (или части — шард) камер, публикует `<id>_anpr`, пишет события, снимки и статус |
| `web` | FastAPI: камеры и их состояние, журнал номеров, поиск, CSV, SSE; синхронизирует пути MediaMTX с `cameras.yml` |
| `nginx` | Единая точка входа: `/` → web, `/webrtc/` и `/hls/` → MediaMTX |
| `demo` | (профиль `demo`) крутит `samples/demo.mp4` вместо камеры |

## Быстрый старт

```bash
git clone https://github.com/OWNER/anpr-ru.git && cd anpr-ru
make env                        # создаст .env и cameras.yml из шаблонов
# в cameras.yml опишите камеры, в .env — CAMx_URL и WEBRTC_HOST
# положите модель детекции номера в models/plate.pt (см. «Обучение»)
docker compose up -d --build    # CPU
# или GPU (NVIDIA Container Toolkit):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Интерфейс: `http://<хост>:8080` (сетка: `http://<хост>:8080/#view=grid`) ·
RTSP: `rtsp://<хост>:8554/<id>` — исходный, `rtsp://<хост>:8554/<id>_anpr` — с разметкой

Демо без камеры: положите любое видео с машинами в `samples/demo.mp4` и выполните `make demo`
(у `cam1` в `cameras.yml` должен быть `url: publisher` — так по умолчанию, если `CAM1_URL` не задан).

Без Docker:

```bash
pip install -r anpr/requirements.txt
# одна камера / файл
python anpr/anpr.py --source samples/demo.mp4 --plate-model models/plate.pt --show --device cpu
# все камеры из cameras.yml напрямую, без MediaMTX
python anpr/anpr.py --cameras-file cameras.yml --plate-model models/plate.pt --show
```

## Камеры

Все камеры описываются в `cameras.yml` (шаблон — [`cameras.example.yml`](cameras.example.yml); файл в git не попадает):

```yaml
defaults:                  # для всех камер, можно переопределить у каждой
  fps: 15
  min_votes: 3
  min_vehicle_width: 80

cameras:
  - id: cam1               # a-z, 0-9, _ и -; это же путь MediaMTX
    title: Въезд КПП-1
    url: ${CAM1_URL}       # подстановка из .env; ${VAR:-по_умолчанию}; publisher — поток публикуют извне
    roi: [[0.05, 0.35], [0.95, 0.35], [0.95, 1.0], [0.05, 1.0]]   # зона распознавания, доли кадра
  - id: cam2
    title: Выезд КПП-1
    url: rtsp://user:pass@192.168.1.11:554/stream1
    enabled: false         # временно отключить
```

| Поле | Описание |
|---|---|
| `id` | Уникальный идентификатор, путь MediaMTX (`<id>` и `<id>_anpr`) |
| `title` | Название в интерфейсе и CSV |
| `url` | RTSP камеры или `publisher` |
| `enabled` | `false` — камера не создаётся и не обрабатывается |
| `roi` | Полигон зоны распознавания: номер читается, только если нижняя середина рамки ТС внутри |
| `fps`, `min_votes`, `min_vehicle_width` | Параметры распознавания для камеры |

Что происходит при изменении `cameras.yml`:

- **web** перечитывает файл автоматически: интерфейс и пути MediaMTX обновляются в течение ~15 с
  (при ошибке в файле остаётся прежняя конфигурация, ошибка показывается в интерфейсе);
- **anpr** нужно перезапустить: `make restart-anpr`.

Проверить файл: `make check-cameras`. Состояние камер: `make cameras` или `GET /api/cameras`.

### Масштабирование

Один экземпляр `anpr` обрабатывает все включённые камеры: на каждую камеру — свой поток чтения,
свой детектор ТС с трекером и свой выходной поток; модель номера и OCR загружаются один раз.
Когда одной GPU не хватает, раздайте камеры нескольким экземплярам (шардам) через `ANPR_ONLY`:

```bash
cp docker-compose.shards.example.yml docker-compose.shards.yml   # распределите id камер по шардам
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.shards.yml up -d
```

Каждая камера должна попасть ровно в один шард; в интерфейсе видно, какой экземпляр её обрабатывает.
Ориентир: ~6–10 камер по 15 FPS на GPU уровня RTX 3060, на CPU — 1–2 камеры по 5 FPS.
Шарды на других хостах подключаются к тому же MediaMTX и общему каталогу `data/` (NFS),
либо пишут события в свой `data/` с отдельным web.

## Настройка

Переменные `.env`:

| Переменная | По умолчанию | Описание |
|---|---|---|
| `CAM1_URL`, `CAM2_URL`, … | — | Адреса камер для подстановки в `cameras.yml` |
| `WEBRTC_HOST` | `127.0.0.1` | IP хоста, видимый браузерам (обязательно для WebRTC) |
| `HTTP_PORT` / `RTSP_PORT` | `8080` / `8554` | Внешние порты |
| `ANPR_OCR` | `easyocr` | `easyocr` или `crnn` |
| `ANPR_DEVICE` | `cpu` | `cpu` или индекс GPU |
| `ANPR_FPS` | `15` | FPS выходного потока (если не задан в `cameras.yml`) |
| `ANPR_MIN_VOTES` | `3` | Сколько одинаковых чтений нужно для подтверждения |
| `ANPR_ONLY` | — | Обрабатывать только эти камеры (`cam1,cam2`) — для шардов |

Все параметры `anpr.py` доступны как `--ключ` или `ANPR_<КЛЮЧ>` (`python anpr/anpr.py --help`).

Порты: `8080/tcp` — интерфейс, `8554/tcp` — RTSP, `8189/udp` — медиа WebRTC. Если UDP закрыт,
переключите плеер на HLS (задержка ~1–2 с). Метрики MediaMTX для Prometheus — `http://mediamtx:9998/metrics`
внутри docker-сети (API `:9997` и метрики наружу не публикуются).

## API

| Метод | Описание |
|---|---|
| `GET /api/config` | Список включённых камер (без URL и паролей) |
| `GET /api/cameras` | Состояние камер: исходный поток и поток с разметкой в MediaMTX, FPS, номера, шард, ошибки |
| `POST /api/cameras/sync` | Немедленно синхронизировать пути MediaMTX с `cameras.yml` |
| `GET /api/events?q=&camera=cam1,cam2&date_from=&date_to=&limit=` | Журнал номеров (новые сверху) |
| `GET /api/events.csv` | То же в CSV (`;`, UTF-8 BOM для Excel), с названием камеры |
| `GET /api/stats?camera=` | Счётчики: за час, сегодня, уникальных, всего, сегодня по камерам |
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
make test    # pytest: правила номеров, cameras.yml, API, синхронизация MediaMTX
make lint    # ruff
```

CI (GitHub Actions) запускает линтер и тесты, собирает образы `web` и `anpr` и публикует их в
GitHub Container Registry: `ghcr.io/OWNER/anpr-ru/web`, `ghcr.io/OWNER/anpr-ru/anpr`
(ветка `main` и теги `v*`).

## Структура

```
anpr/            сервис распознавания (anpr.py, plate_rules.py, crnn.py, Dockerfile)
common/          cameras.py — разбор и проверка cameras.yml (общий для anpr и web)
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
