# SensifyHR Mülakat Analiz Sistemi

Online iş görüşmelerinde aday davranışlarını analiz eden AI motoru.

## Özellikler

- **Yüz ve Gaze Analizi**: MediaPipe Face Mesh ile 478 landmark + iris tracking
  - Ham özellikler: Ağız genişliği/yüksekliği, göz açıklığı, kaş mesafesi, çene açıklığı
  - Gaze yönü: Left, Right, Up, Down, Center (iris tabanlı)
  - Kalman filtreleme ile gürültü azaltma
- **Ses Analizi**: Librosa ile ham ses özellikleri
  - Speech rate, RMS energy, Pitch (f0), Silence ratio
  - Spectral centroid, Zero crossing rate
- **Py-Feat Analizi**: Duygu, kafa pozisyonu ve landmark çıktıları (CPU)
  - Frame bazlı ham veriler rapora eklenir
  - Terminal/Gemini özeti: duygu dağılımları, yüz tespit oranı, pose istatistikleri
- **Frame Özetleme**: Detaylı frame analizlerini özetleyen modül
- **Rapor Oluşturma**: JSON, HTML ve PDF formatında detaylı raporlar
- **RESTful API**: FastAPI ile modern API arayüzü

## Kurulum

### Gereksinimler

- Python 3.10+
- FFmpeg

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd sensifyHRMülakay
```

2. **Virtual environment oluşturun:**
```bash
python -m venv sensifyhr
sensifyhr\Scripts\activate  # Windows
# veya
source sensifyhr/bin/activate  # Linux/Mac
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

2. **Terminal'de:**
```bash
python test_example.py test_video.mp4
```

3. **Canlı kamera testi:**
```bash
python test_webcam_mediapipe.py
```

### API'yi Başlatma

```bash
python api/main.py
```

veya

```bash
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

#### 4. API Dokümantasyonu
Tarayıcıda açın: `http://localhost:8000/docs`

## Proje Yapısı

```
sensifyHRMülakay/
├── src/
│   ├── video_processor.py           # Video işleme
│   ├── mediapipe_face_gaze_analyzer.py  # MediaPipe yüz + gaze analizi
│   ├── voice_analyzer.py            # Ses analizi (Librosa)
│   ├── frame_summarizer.py          # Frame analiz özetleme
│   ├── pyfeat_analyzer.py           # Py-Feat analizi (duygu/pose/landmark)
│   ├── pyfeat_summarizer.py         # Py-Feat özetleme
│   ├── report_generator.py          # Rapor oluşturma (JSON, HTML, PDF)
│   └── pipeline.py                  # Ana pipeline
├── api/
│   └── main.py                      # FastAPI endpoints
├── reports/                          # Analiz raporları
├── uploads/                          # Yüklenen videolar
├── requirements.txt
└── README.md
```

## Rapor Formatı

Analiz sonuçları JSON formatında döner:

```json
{
  "interview_id": "uuid",
  "duration_seconds": 600,
  "frame_summary": {
    "general_statistics": {...},
    "gaze_summary": {...},
    "cognitive_load_score": {...},
    "emotion_change_points": [...]
  },
  "voice_analysis": {
    "raw_voice_features": {
      "speech_rate": {...},
      "rms_energy": {...},
      "pitch_f0": {...},
      "silence_ratio": 0.2
    }
  },
  "pyfeat_summary": {
    "emotion_distribution": {...},
    "pose_summary": {...},
    "face_detection_rate": 100.0,
    "general_statistics": {...}
  },
  "report_files": {
    "json": "reports/report_xxx.json",
    "html": "reports/report_xxx.html",
    "pdf": "reports/report_xxx.pdf"
  }
}
```

Detaylı format ve teknik bilgiler için `PROJE_DOKUMANTASYONU.md` dosyasına bakın.

## Teknoloji Stack

- **MediaPipe**: Yüz landmark ve iris tracking
- **Librosa**: Ses analizi
- **Py-Feat**: Duygu, pose ve landmark analizi
- **FastAPI**: REST API
- **Plotly**: Veri görselleştirme
- **Jinja2**: HTML rapor şablonları
- **xhtml2pdf**: PDF oluşturma

## Lisans

Ticari kullanım için lisans gerekebilir.
