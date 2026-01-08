# SensifyHR Mülakat Analiz Sistemi

Online iş görüşmelerinde aday davranışlarını analiz eden AI motoru.

## Özellikler

- **Yüz Duygu Analizi**: DeepFace kullanarak 7 temel duygu tespiti
- **Göz Teması Analizi**: MediaPipe ile göz teması ve bakış yönü analizi
- **Gerçek Zamanlı İşleme**: Arka planda video analizi
- **Yapılandırılmış Raporlar**: JSON formatında detaylı analiz raporları
- **RESTful API**: FastAPI ile modern API arayüzü

## Kurulum

### Gereksinimler

- Python 3.10+
- FFmpeg
- CUDA (opsiyonel, GPU desteği için)

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd sensifyHRMülakay
```

2. **Virtual environment oluşturun:**
```bash
python -m venv torch_env
torch_env\Scripts\activate  # Windows
# veya
source torch_env/bin/activate  # Linux/Mac
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **FFmpeg'i yükleyin:**
- Windows: [FFmpeg indir](https://ffmpeg.org/download.html) ve PATH'e ekleyin
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

## Kullanım

### Hızlı Test

1. **Test videosunu proje klasörüne koyun** (örn: `test_video.mp4`)

2. **Anaconda Prompt'ta:**
```bash
cd C:\Users\cetki\Desktop\sensifyHRMülakay
conda activate torch_env
python test_example.py test_video.mp4
```

3. **Veya Windows'ta hızlı test:**
```bash
test_quick.bat test_video.mp4
```

### API'yi Başlatma

```bash
conda activate torch_env
uvicorn api.main:app --reload
```

API `http://localhost:8000` adresinde çalışacaktır.

### API Endpoints

#### 1. Video Analizi
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@interview_video.mp4"
```

#### 2. Analiz Durumu
```bash
curl "http://localhost:8000/status/{interview_id}"
```

#### 3. Sağlık Kontrolü
```bash
curl "http://localhost:8000/health"
```

### Docker ile Çalıştırma

```bash
docker build -t sensifyhr-interview .
docker run -p 8000:8000 sensifyhr-interview
```

## Proje Yapısı

```
sensifyHRMülakay/
├── src/
│   ├── video_processor.py      # Video işleme
│   ├── emotion_analyzer.py     # Duygu analizi
│   ├── eye_contact_analyzer.py # Göz teması analizi
│   └── pipeline.py             # Ana pipeline
├── api/
│   └── main.py                 # FastAPI endpoints
├── requirements.txt
├── Dockerfile
└── README.md
```

## Rapor Formatı

Analiz sonuçları JSON formatında döner:

```json
{
  "interview_id": "uuid",
  "duration_seconds": 600,
  "emotion_analysis": {...},
  "eye_contact_analysis": {...},
  "overall_assessment": {...}
}
```

Detaylı format için `ARCHITECTURE.md` dosyasına bakın.

## Geliştirme

Detaylı geliştirme raporu için `PROJECT_REPORT.md` dosyasına bakın.

## Lisans

Ticari kullanım için lisans gerekebilir.
