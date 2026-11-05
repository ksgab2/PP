FROM python:3.11-slim

# OpenCV headless 실행에 필요한 최소 시스템 라이브러리 + 한글 폰트
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY main.py .
COPY best.pt .
COPY static ./static

# Ultralytics가 자동 다운로드/캐시하는 경로를 컨테이너 내부로 고정
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
