# SensifyHR Mülakat Analiz Sistemi - Docker Image
FROM python:3.10-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# CPU-only çalıştırma (Py-Feat / Torch için)
ENV CUDA_VISIBLE_DEVICES=""

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Geçici dosyalar için dizin oluştur
RUN mkdir -p temp_uploads

# Port
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
