.PHONY: help env up up-gpu demo down logs test lint datasets train-plate train-ocr

help:            ## список команд
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

env:             ## создать .env из шаблона
	@test -f .env || cp .env.example .env && echo ".env готов — отредактируйте его"

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
