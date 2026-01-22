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
  - Zaman pencereli örnekleme (performans için)
  - Frame bazlı ham veriler rapora eklenir (seçili frame'ler)
  - Terminal/Gemini özeti: duygu dağılımları, yüz tespit oranı, pose istatistikleri
- **Frame Özetleme**: Detaylı frame analizlerini özetleyen modül
- **Rapor Oluşturma**: JSON ve HTML formatında detaylı raporlar
- **RESTful API**: FastAPI ile modern API arayüzü
- **Dayanıklılık**: Yüz yoksa hata atmaz, NaN/inf veriler JSON uyumlu hale getirilir

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

**Gemini entegrasyonu**: `/analyze` çağrısı sonrasında Gemini otomatik çalışır ve
çıktı `ai_analysis` alanına yazılır (ham veriler gönderilmez).
Gerekli env: `GEMINI_API_KEY` (opsiyonel: `GEMINI_MODEL`).

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
│   ├── report_generator.py          # Rapor oluşturma (JSON, HTML)
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
    "html": "reports/report_xxx.html"
  }
}
```

Not: HTML grafikler `frame_analysis` ve `voice_analysis` ham serileri üzerinden Python/Matplotlib ile üretilir (uzun videolarda örneklenir).

Detaylı format ve teknik bilgiler için `PROJE_DOKUMANTASYONU.md` dosyasına bakın.

## Performans Ayarları (Py-Feat)

`src/pipeline.py` içinde ayarlanabilir:
- `window_size_seconds`
- `speech_sampling_seconds`
- `silence_sampling_seconds`
- `batch_size`

## İK Raporu (Gemini için)

- **Ham veri gönderme**: `frame_analysis` ve Py-Feat frame detayları Gemini'ye gönderilmez.
- **Özet üzerinden yorum**: `frame_summary`, `voice_analysis` ve `pyfeat_summary` kullanılır.
- **Prompt dosyası**: `src/prompt.txt`

Önerilen prompt şablonu:
```
Sen, video mülakatlar üzerinden aday değerlendirmesi yapan
kıdemli bir İK analisti ve davranış bilimcisin.

Sana verilen veriler:
- Adayın video mülakatı boyunca çıkarılmış
  multimodal analiz çıktılarıdır.
- Bu çıktılar; ses, konuşma akıcılığı, yüz ifadeleri,
  bakış yönü, duygusal durum ve zaman içindeki değişimleri içerir.
- Veriler sayısal ve objektiftir, ancak yorumlanmaya ihtiyaç duyar.

Görevin:
Bu multimodal çıktıları bir İK uzmanı bakış açısıyla,
kısa, net ve yorumlayıcı bir rapora dönüştürmek.

Aşağıdaki başlıklar altında yaz:

1) Genel İzlenim
   - Özellikle ilk 10 saniyeye odaklan
   - Aday ilk bakışta nasıl algılanıyor?

2) İletişim ve Akıcılık
   - Konuşma temposu
   - Duraksamalar
   - Artikülasyon ve ritim

3) Duygusal Durum
   - Stres / rahatlık sinyalleri
   - Duygusal stabilite
   - Zaman içindeki değişim

4) Güven ve Beden Dili
   - Göz teması
   - Yüz açıklığı
   - Genel beden dili tutarlılığı

5) Olası Risk Sinyalleri
   - Aşırı stres
   - Kaçamak bakış
   - Tutarsızlıklar
   (Varsa belirt, yoksa “belirgin risk sinyali yok” de)

6) Genel Değerlendirme
   - Bu aday mülakatta nasıl bir izlenim bırakır?
   - Kısa, profesyonel bir İK yorumu ile bitir.

Yazım kuralları:
- Teknik terimleri sadeleştir
- İK uzmanına hitap et
- Abartma, varsayım yapma
- Kısa paragraflar kullan
- Yorum yap, veri tekrar etme
```

## Teknoloji Stack

- **MediaPipe**: Yüz landmark ve iris tracking
- **Librosa**: Ses analizi
- **Py-Feat**: Duygu, pose ve landmark analizi
- **FastAPI**: REST API
- **Matplotlib/Seaborn**: Veri görselleştirme
- **Jinja2**: HTML rapor şablonları
- **Google GenAI**: Gemini istemcisi

## Lisans

Ticari kullanım için lisans gerekebilir.
