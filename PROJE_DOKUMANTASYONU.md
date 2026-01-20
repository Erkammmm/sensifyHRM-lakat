# SensifyHR Mülakat Analiz Sistemi - Proje Dokümantasyonu

## 📋 Proje Özeti

SensifyHR Mülakat Analiz Sistemi, video tabanlı mülakat kayıtlarını analiz ederek adayların davranışsal özelliklerini değerlendiren bir sistemdir. Sistem üç ana bileşenden oluşur:

1. **Yüz ve Gaze Analizi**: MediaPipe Face Mesh ile 478 landmark + iris tracking
2. **Ses Analizi**: Librosa ile ham ses özellikleri analizi
3. **Py-Feat Analizi**: Duygu, kafa pozisyonu ve landmark analizi (CPU)
4. **Rapor Oluşturma**: JSON, HTML ve PDF formatında detaylı raporlar

---

## 📁 Proje Yapısı

```
sensifyHRMülakay/
├── src/                                  # Ana kaynak kod klasörü
│   ├── __init__.py
│   ├── video_processor.py                # Video işleme (frame/audio extraction)
│   ├── mediapipe_face_gaze_analyzer.py   # MediaPipe yüz + gaze analizi
│   ├── voice_analyzer.py                 # Ses analizi (Librosa - raw features)
│   ├── frame_summarizer.py               # Frame analiz özetleme
│   ├── pyfeat_analyzer.py                # Py-Feat analizi (duygu/pose/landmark)
│   ├── pyfeat_summarizer.py              # Py-Feat özetleme
│   ├── report_generator.py               # Rapor oluşturma (JSON, HTML, PDF)
│   └── pipeline.py                       # Ana pipeline (tüm modülleri koordine eder)
├── api/                                  # FastAPI REST API
│   └── main.py
├── reports/                              # Analiz raporları (JSON, HTML, PDF)
├── uploads/                              # Yüklenen videolar
├── test_example.py                       # Test scripti (video analizi için)
├── test_webcam_mediapipe.py              # Canlı kamera testi
├── requirements.txt                      # Python bağımlılıkları
└── README.md                             # Genel bilgiler
```

---

## 🔧 Kullanılan Teknolojiler ve Modeller

### 1. Yüz ve Gaze Analizi (Face & Gaze Analysis)

**Kütüphane**: MediaPipe Face Mesh  
**Versiyon**: 0.10.9  
**Model**: Face Mesh with Iris (refine_landmarks=True)

**Özellikler**:
- **478 Facial Landmarks**: Yüzün detaylı geometrik noktaları
- **Iris Tracking**: Her iki göz için iris merkez noktaları (4 nokta/göz)
- **Real-time Processing**: Yüksek FPS ile gerçek zamanlı işleme

**Ham Özellikler (Raw Features)**:
- **mouth_width_norm**: Normalize edilmiş ağız genişliği (0-1)
- **mouth_height_norm**: Normalize edilmiş ağız yüksekliği (0-1)
- **eye_opening_norm**: Normalize edilmiş göz açıklığı (0-1)
- **brow_distance_norm**: Normalize edilmiş kaş-göz mesafesi (0-1)
- **jaw_open_norm**: Normalize edilmiş çene açıklığı (0-1)

**Gaze Yönü (Iris Tabanlı)**:
- **Left**: Sola bakış
- **Right**: Sağa bakış
- **Up**: Yukarı bakış
- **Down**: Aşağı bakış
- **Center**: Merkez (kameraya bakış)

**Gaze Hesaplama Yöntemi**:
1. İlk 2 saniyede iris pozisyonları toplanır (baseline calibration)
2. Baseline'dan standart sapma hesaplanır
3. Sonraki frame'lerde iris pozisyonu baseline'a göre normalize edilir
4. Kalman filtresi ile gürültü azaltılır
5. Threshold'lara göre yön belirlenir

**Kalman Filtreleme**:
- Gaze x ve y koordinatları için ayrı Kalman filtreleri
- Ham özellikler için de Kalman filtreleme (opsiyonel)
- Gürültü azaltma ve daha stabil sonuçlar

**Frame Skip Optimizasyonu**:
- Varsayılan: Her 3 frame'de bir analiz (frame_skip=3)
- Performans için optimize edilmiş
- Tüm frame'ler için timestamp senkronizasyonu

**Modül**: `src/mediapipe_face_gaze_analyzer.py`

---

### 2. Ses Analizi (Voice Analysis)

**Kütüphane**: Librosa  
**Versiyon**: 0.10.0+

**Ham Özellikler (Raw Voice Features)**:
- **speech_rate**: Konuşma hızı (relative_index_0_100)
- **rms_energy**: RMS enerjisi (mean, std, min, max)
- **pitch_f0**: Temel frekans (mean, std, min, max) - Hz cinsinden
- **pause_durations**: Duraklama süreleri listesi
- **silence_ratio**: Sessizlik oranı (0-1)
- **spectral_centroid**: Spektral ağırlık merkezi
- **zero_crossing_rate**: Sıfır geçiş oranı

**Veri Ön İşleme**:
- Ses dosyası yükleme (librosa.load)
- Resample (16 kHz)
- Mono conversion
- Frame-based feature extraction

**Modül**: `src/voice_analyzer.py`

---

### 3. Py-Feat Analizi (Emotion & Pose)

**Kütüphane**: py-feat  
**Çalışma Modu**: CPU (hız öncelikli)

**Ham Çıktılar**:
- **emotions**: Duygu skorları
- **pose**: Pitch, Yaw, Roll
- **landmarks**: Yüz landmark koordinatları
- **facebox / aus**: Ham yüz kutusu ve AU benzeri veriler

**Özet Bileşenleri**:
- **Emotion Distribution**: Ortalama, varyans, baskın duygu
- **Face Detection Rate**: Yüz tespit oranı
- **Pose Summary**: Pitch/Yaw/Roll mean/std/min/max

**Modül**: `src/pyfeat_analyzer.py`, `src/pyfeat_summarizer.py`

---

### 4. Frame Özetleme (Frame Summarization)

**Amaç**: Detaylı frame analizlerini özetleyerek daha yönetilebilir hale getirmek.

**Özet Bileşenleri**:
- **General Statistics**: Ortalama özellikler, gaze center yüzdesi
- **Gaze Summary**: Her yön için frame sayısı ve yüzdesi
- **Cognitive Load Score**: Thinking/Reading tespiti (jaw_open + gaze_direction)
- **Emotion Change Points**: Özelliklerdeki radikal değişimler (30%+ değişim)

**Modül**: `src/frame_summarizer.py`

---

### 5. Rapor Oluşturma (Report Generation)

**Formatlar**:
- **JSON**: Ham ve özetlenmiş veriler
- **HTML**: Jinja2 template ile görsel rapor
- **PDF**: xhtml2pdf ile HTML'den PDF oluşturma

**Görselleştirmeler**:
- **Time-series Grafikler**: Ağız ve göz hareketleri zaman içinde
- **Gaze Distribution**: Pasta grafiği ile gaze yönleri dağılımı
**Not**: Grafikler `frame_analysis` (ham frame verisi) üzerinden Python/Plotly ile üretilir.

**Modül**: `src/report_generator.py`

---

## 🔄 Pipeline Yapısı

### Ana Pipeline (`src/pipeline.py`)

**Akış**:
1. **Video İşleme**: Frame'ler ve ses çıkarılır
2. **MediaPipe Analizi**: Her 3 frame'de bir yüz + gaze analizi
3. **Ses Analizi**: Librosa ile tüm ses dosyası analiz edilir
4. **Py-Feat Analizi**: Video üzerinde duygu/pose/landmark analizi
5. **Frame Özetleme**: Detaylı frame analizleri özetlenir
6. **Rapor Oluşturma**: JSON, HTML ve PDF formatında raporlar kaydedilir

**Performans Optimizasyonları**:
- Frame sampling: Her 3 frame'de bir analiz (frame_skip=3)
- Lazy loading: MediaPipe ilk kullanımda başlatılır
- Kalman filtreleme: Gürültü azaltma ve daha stabil sonuçlar

---

## 📊 Çıktı Formatı (JSON Rapor)

```json
{
  "interview_id": "uuid",
  "duration_seconds": 56.62,
  "frame_summary": {
    "general_statistics": {
      "total_frames_analyzed": 415,
      "average_features": {
        "mouth_width_norm": 0.937,
        "mouth_height_norm": 0.034,
        "eye_opening_norm": 0.068,
        "brow_distance_norm": 0.415,
        "jaw_open_norm": 0.669
      },
      "gaze_center_percentage": 88.2
    },
    "gaze_summary": {
      "direction_counts": {
        "center": 75,
        "down": 10,
        "left": 0,
        "right": 0,
        "up": 0
      },
      "direction_percentages": {
        "center": 88.2,
        "down": 11.8
      }
    },
    "cognitive_load_score": {
      "thinking_reading_count": 10,
      "thinking_reading_percentage": 11.8,
      "total_speaking_frames": 85,
      "speaking_percentage": 100.0
    },
    "emotion_change_points": [
      {
        "timestamp": 0.10,
        "feature": "mouth_height_norm",
        "change_percentage": 91.3,
        "previous_value": 0.005,
        "current_value": 0.009
      }
    ]
  },
  "voice_analysis": {
    "raw_voice_features": {
      "speech_rate": {
        "value": 8.39,
        "unit": "relative_index_0_100"
      },
      "rms_energy": {
        "mean": 0.064711,
        "std": 0.012345,
        "min": 0.001234,
        "max": 0.123456
      },
      "pitch_f0": {
        "mean": 240.38,
        "std": 12.34,
        "min": 200.0,
        "max": 280.0
      },
      "silence_ratio": 0.201,
      "duration_seconds": 13.80
    }
  },
  "pyfeat_analysis": {
    "frame_analysis": [
      {
        "frame_index": 0,
        "timestamp": 0.0,
        "faces": [
          {
            "face_id": 0,
            "emotions": {...},
            "pose": {"Pitch": -2.1, "Yaw": 1.4, "Roll": 0.7},
            "landmarks": [...]
          }
        ]
      }
    ],
    "summary": {
      "emotion_distribution": {...},
      "pose_summary": {...},
      "face_detection_rate": 100.0,
      "general_statistics": {...}
    },
    "metadata": {...}
  },
  "report_files": {
    "json": "reports/report_xxx.json",
    "html": "reports/report_xxx.html",
    "pdf": "reports/report_xxx.pdf"
  }
}
```

---

## 🧠 İK Raporlama (Gemini Prompt)

- **Ham veri gönderilmez**: `frame_analysis` ve Py-Feat frame detayları Gemini'ye dahil edilmez.
- **Sadece özet kullanılır**: `frame_summary`, `voice_analysis`, `pyfeat_summary`.

**Prompt şablonu**:
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

---

## 🛠️ Kullanılan Yöntemler

### Veri Ön İşleme
- **Yüz Tespiti**: MediaPipe Face Mesh (478 landmark + iris)
- **Normalizasyon**: Landmark koordinatları normalize edilir (0-1 aralığı)
- **Frame Skip**: Performans için her 3 frame'de bir analiz

### Model Inference
- **MediaPipe**: C++ backend, Python wrapper
- **Librosa**: NumPy-based signal processing

### Hiperparametreler
- **Frame Skip**: 3 (her 3 frame'de bir analiz)
- **Gaze Baseline Calibration**: İlk 2 saniye
- **Gaze Threshold**: Baseline'dan standart sapma bazlı
- **Kalman Filter**: Process variance=5e-2, Measurement variance=1e-1
- **Audio Sample Rate**: 16 kHz

---

## 📦 Bağımlılıklar

**Ana Kütüphaneler**:
- `mediapipe==0.10.9`: Yüz landmark ve iris tracking
- `protobuf==3.20.3`: MediaPipe uyumluluğu için sabit versiyon
- `librosa>=0.10.0`: Ses analizi
- `py-feat>=0.7.0`: Duygu, pose, landmark analizi
- `opencv-python>=4.8.0`: Video işleme
- `numpy>=1.24.0,<2.0.0`: Numerik işlemler
- `scipy>=1.10.0,<1.11.0`: Py-Feat uyumluluğu
- `fastapi>=0.104.0`: REST API
- `google-genai>=0.6.0`: Gemini istemcisi
- `plotly>=5.18.0`: Veri görselleştirme
- `jinja2>=3.1.2`: HTML şablonları
- `xhtml2pdf>=0.2.11`: PDF oluşturma

**Tam liste**: `requirements.txt`

---

## 🚀 Kullanım

### Video Analizi

```bash
python test_example.py video_dosyasi.mp4
```

**Çıktı**: `reports/report_<uuid>.json`, `reports/report_<uuid>.html`, `reports/report_<uuid>.pdf`

### Canlı Kamera Testi

```bash
python test_webcam_mediapipe.py
```

### API Kullanımı

```bash
# API'yi başlat
python api/main.py

# Video yükle ve analiz et
curl -X POST "http://localhost:8000/analyze" -F "file=@video.mp4"
```

**Gemini entegrasyonu**: `/analyze` çağrısı sonrası otomatik çalışır ve
çıktı `ai_analysis` alanına yazılır (ham veriler gönderilmez).
**Gerekli env**: `GEMINI_API_KEY` (opsiyonel: `GEMINI_MODEL`)
**Prompt dosyası**: `src/prompt.txt` (yoksa `src/prompt`)

---

## 📝 Notlar

- **Performans**: MediaPipe frame sampling (her 3 frame), Py-Feat CPU analiz
- **Kalman Filtreleme**: Gürültü azaltma için kullanılır, daha stabil sonuçlar verir
- **Baseline Calibration**: Gaze yönü için ilk 2 saniyede otomatik kalibrasyon yapılır
- **Raporlar**: Tüm raporlar `reports/` klasörüne kaydedilir
- **Uploads**: Yüklenen videolar `uploads/` klasörüne kaydedilir

---

## 🔍 Teknik Detaylar

### MediaPipe Face Mesh
- **Landmarks**: 478 nokta (yüz + iris)
- **Iris Landmarks**: 468-471 (sol), 473-476 (sağ)
- **Input**: RGB görüntü
- **Output**: Normalize edilmiş landmark koordinatları (0-1)
- **Inference Time**: ~10-20ms per frame (CPU)

### Gaze Hesaplama
- **Method**: Iris center position relative to eye box
- **Baseline**: İlk 2 saniyede toplanan iris pozisyonları
- **Normalization**: Baseline'a göre normalize edilir
- **Direction Classification**: Threshold bazlı (baseline std kullanılır)

### Librosa
- **Sample Rate**: 16 kHz
- **Frame Length**: 2048 samples
- **Hop Length**: 512 samples
- **Processing Time**: Real-time'da çalışır

---

## 📚 Referanslar

- **MediaPipe**: https://mediapipe.dev/
- **Librosa**: https://librosa.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Plotly**: https://plotly.com/python/

---

**Son Güncelleme**: 2026-01-15
