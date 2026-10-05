.PHONY: help env up up-gpu demo down logs test lint datasets train-plate train-ocr cameras check-cameras restart-anpr

help:            ## список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

env:             ## создать .env и cameras.yml из шаблонов
	@test -f .env || (cp .env.example .env && echo ".env создан — отредактируйте его")
	@test -f cameras.yml || (cp cameras.example.yml cameras.yml && echo "cameras.yml создан — опишите камеры")

up: env          ## запустить (CPU)
	docker compose up -d --build

up-gpu: env      ## запустить с NVIDIA GPU
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

demo: env        ## демо на samples/demo.mp4 без камеры
	docker compose --profile demo up -d --build

down:            ## остановить
	docker compose --profile demo down

logs:            ## логи
	docker compose logs -f --tail=100

test:            ## тесты
	python -m pytest -q

lint:            ## линтер
	ruff check .

datasets:        ## скачать и подготовить датасеты
	python tools/prepare_datasets.py all

train-plate:     ## обучить детектор номера -> models/plate.pt
	tools/train_plate.sh

train-ocr:       ## обучить CRNN -> models/crnn.pt
	python tools/prepare_datasets.py ocr && python tools/train_crnn.py

cameras:         ## состояние камер (через запущенный web)
	@curl -s localhost:$${HTTP_PORT:-8080}/api/cameras | python3 -m json.tool

check-cameras:   ## проверить cameras.yml
	@python3 -c "import sys; sys.path.insert(0,'common'); from cameras import load_cameras; \
	cs=load_cameras('cameras.yml', include_disabled=True); \
	[print(f\"  {'+' if c.enabled else '-'} {c.id:12} {c.title}\") for c in cs]; print(f'ok: {len(cs)} камер')"

restart-anpr:    ## перезапустить распознавание (после правки cameras.yml)
	docker compose restart anpr
