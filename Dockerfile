# =============================================
# SensifyHR Mülakat Analiz Sistemi v3.0 (FAZ-3)
# Docker Image (varsayılan: CPU). GPU için host CUDA + uygun base image gerekir.
# =============================================

FROM python:3.10-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python ortam ayarları
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Varsayılan performans ayarları (isteğe göre docker run -e ile override edilebilir)
ENV SENSIFYHR_FACE_TARGET_FPS=5
ENV SENSIFYHR_STT_BACKEND=faster-whisper
ENV SENSIFYHR_FW_COMPUTE_TYPE=int8_float16
# Prewarm default kapalı (container startup süresini uzatabilir)
ENV SENSIFYHR_PREWARM_MODELS=0

# GPU kullanımı: boş bırak = otomatik tespit, "" = CPU only
# ENV CUDA_VISIBLE_DEVICES=""

# Gereksinimler (lock dosyası ile)
COPY requirements.lock.sonn.txt ./requirements.lock.sonn.txt
RUN sed -i '/^packaging @/c\packaging' requirements.lock.sonn.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        -r requirements.lock.sonn.txt

# Ek paketler (lock dosyasında olmayabilir)
RUN pip install --no-cache-dir ollama 2>/dev/null || true

# Uygulama kodları
COPY . .

# Gerekli klasörler
RUN mkdir -p temp_uploads uploads reports

# MediaPipe model dosyasını indir (weights/ altına)
RUN python -c "import urllib.request; urllib.request.urlretrieve( \
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', \
    'weights/face_landmarker.task')" 2>/dev/null || echo "Model indirme atlandı (offline build)"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
