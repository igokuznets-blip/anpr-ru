# Сборка из корня репозитория: docker build -f web/Dockerfile .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY web/app.py common/cameras.py ./
COPY web/static ./static
# hls.js кладём локально — интерфейс работает без интернета
ADD https://cdn.jsdelivr.net/npm/hls.js@1.5.17/dist/hls.min.js ./static/hls.min.js
RUN chmod 644 ./static/hls.min.js
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/config')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
