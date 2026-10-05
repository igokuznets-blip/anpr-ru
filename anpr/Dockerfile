# Сборка из корня репозитория: docker build -f anpr/Dockerfile .
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fonts-dejavu-core libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY anpr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Веса скачиваются на этапе сборки — контейнер работает в изолированной сети
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
 && python -c "import easyocr; easyocr.Reader(['ru'], gpu=False)"
COPY anpr/*.py common/cameras.py ./
ENTRYPOINT ["python", "anpr.py"]
