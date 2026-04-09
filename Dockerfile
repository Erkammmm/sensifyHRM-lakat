# =============================================
# SensifyHR Mülakat Analiz Sistemi v3.x
# Docker Image — CPU (GPU versiyonu için requirements.txt + CUDA tabanlı base image kullanın)
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

# CPU modu: CUDA devre dışı
ENV CUDA_VISIBLE_DEVICES=""

# Tüm CPU çekirdeklerini kullan (8 çekirdek)
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV NUMEXPR_NUM_THREADS=8
ENV OPENBLAS_NUM_THREADS=8
ENV TORCH_NUM_THREADS=8

# Gereksinimler
COPY requirements.cpu.txt ./requirements.cpu.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        -r requirements.cpu.txt

# Uygulama kodları
COPY . .

# Gerekli klasörler
RUN mkdir -p temp_uploads reports

EXPOSE 8090

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "2"]
