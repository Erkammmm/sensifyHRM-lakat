#!/bin/bash
# Hızlı Test Scripti - Linux/Mac
# Kullanım: ./test_quick.sh test_video.mp4

echo "========================================"
echo "SensifyHR Mülakat Analiz Sistemi"
echo "Hızlı Test Scripti"
echo "========================================"
echo ""

# Anaconda environment'ı aktif et
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch_env

# Video dosyası kontrolü
if [ -z "$1" ]; then
    echo "HATA: Video dosyası belirtilmedi!"
    echo "Kullanım: ./test_quick.sh test_video.mp4"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "HATA: Video dosyası bulunamadı: $1"
    exit 1
fi

echo "Test videosu: $1"
echo ""
echo "Test başlatılıyor..."
echo ""

# Test scriptini çalıştır
python test_example.py "$1"

echo ""
echo "========================================"
echo "Test tamamlandı!"
echo "========================================"
