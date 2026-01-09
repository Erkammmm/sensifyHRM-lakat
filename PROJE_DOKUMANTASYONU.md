# SensifyHR Mülakat Analiz Sistemi - Proje Dokümantasyonu

## 📋 Proje Özeti

SensifyHR Mülakat Analiz Sistemi, video tabanlı mülakat kayıtlarını analiz ederek adayların davranışsal özelliklerini değerlendiren bir sistemdir. Sistem üç ana bileşenden oluşur:

1. **Duygu Analizi**: DeepFace ile yüz ifadelerinden yaş, cinsiyet, duygu ve ırk tespiti
2. **Göz Teması Analizi**: MobileGaze pre-trained modeli ile bakış yönü tespiti (camera/left/right/up/down)
3. **Ses Analizi**: Librosa ile ses özelliklerinden stres, kaygı, heyecan ve konuşma kalitesi analizi

---

## 📁 Proje Yapısı

```
sensifyHRMülakay/
├── src/                          # Ana kaynak kod klasörü
│   ├── __init__.py
│   ├── video_processor.py        # Video işleme (frame/audio extraction)
│   ├── emotion_analyzer.py      # Duygu analizi (DeepFace)
│   ├── eye_contact_analyzer.py  # Göz teması analizi (MobileGaze)
│   ├── voice_analyzer.py        # Ses analizi (Librosa)
│   ├── behavioral_analyzer.py   # Davranışsal analiz (rule-based fusion)
│   ├── pipeline.py              # Ana pipeline (tüm modülleri koordine eder)
│   ├── gaze_estimation_model.py # MobileGaze model wrapper
│   └── models/                  # Gaze estimation model mimarileri
│       ├── mobilenet.py         # MobileNetV2 architecture
│       ├── mobileone.py         # MobileOne architecture
│       └── resnet.py            # ResNet architectures
├── api/                          # FastAPI REST API
│   └── main.py
├── reports/                      # Analiz raporları (JSON formatında)
├── weights/                      # Pre-trained model weights
│   └── gaze_estimation/
│       └── mobilenetv2.pt       # MobileGaze model weights
├── test_example.py              # Test scripti (video analizi için)
├── requirements.txt             # Python bağımlılıkları
└── README.md                    # Genel bilgiler
```

---

## 🔧 Kullanılan Teknolojiler ve Modeller

### 1. Duygu Analizi (Emotion Analysis)

**Model**: DeepFace  
**Kütüphane**: `deepface` (Python)  
**Versiyon**: 0.0.79+

**Çıktılar**:
- **Age (Yaş)**: Ortalama, min, max, standart sapma
- **Gender (Cinsiyet)**: Dominant cinsiyet, dağılım, güven skoru
- **Emotion (Duygu)**: 7 kategori (angry, disgust, fear, happy, sad, surprise, neutral)
- **Race (Irk)**: Dominant ırk, dağılım, güven skoru

**Veri Seti**: 
- **FER2013**: DeepFace'in eğitildiği veri seti
  - 35,887 grayscale yüz görüntüsü
  - 7 duygu kategorisi
  - Train/Validation/Test split

**Transfer Learning**: 
- ✅ Pre-trained model kullanımı (DeepFace)
- ❌ Fine-tuning yapılmadı (out-of-the-box kullanım)

**Veri Ön İşleme**:
- Yüz tespiti (otomatik)
- Normalizasyon (ImageNet stats)
- Resize (model gereksinimlerine göre)

**Modül**: `src/emotion_analyzer.py`

---

### 2. Göz Teması Analizi (Eye Contact Analysis)

**Model**: MobileGaze (MobileNetV2 tabanlı)  
**Kaynak**: https://github.com/yakhyo/gaze-estimation  
**Model Tipi**: Pre-trained CNN classification model

**Çıktılar**:
- **Gaze Class (Bakış Sınıfı)**: 
  - `camera`: Kameraya bakıyor (göz teması var)
  - `left`: Sola bakıyor
  - `right`: Sağa bakıyor
  - `up`: Yukarı bakıyor
  - `down`: Aşağı bakıyor
- **Gaze Angles**: Yaw (yatay) ve Pitch (dikey) açıları (derece)
- **Eye Contact Score**: Camera sınıfı oranına göre (0-100%)

**Veri Seti**: 
- Model Gaze360 ve MPIIGaze veri setleri üzerinde eğitilmiş
- Classification approach: Açılar bin'lere ayrılmış (90 bins, -90° ile +90° arası)

**Transfer Learning**: 
- ✅ Pre-trained model kullanımı
- ❌ Fine-tuning yapılmadı

**Veri Ön İşleme**:
- MediaPipe Face Detection ile yüz tespiti
- Yüz crop (bounding box)
- Resize (224x224)
- Normalizasyon (ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Float64 (double) format (model gereksinimi)

**Model Mimarisi**:
- Backbone: MobileNetV2 (hafif, mobil uyumlu)
- Output: 2 fully connected layers (pitch ve yaw için ayrı)
- Classification: 90 bins (her biri ~2 derece genişliğinde)

**Modül**: 
- `src/eye_contact_analyzer.py` (analiz logic)
- `src/gaze_estimation_model.py` (model wrapper)
- `src/models/mobilenet.py` (model architecture)

---

### 3. Ses Analizi (Voice Analysis)

**Kütüphane**: Librosa  
**Versiyon**: 0.10.0+

**Çıktılar**:
- **Pitch (Perde)**:
  - Mean pitch, std pitch, pitch range
  - Pitch variability (coefficient of variation)
  - Yüksek değişkenlik → stres/kaygı göstergesi
- **Energy (Enerji)**:
  - Mean energy, std energy, energy variability
  - Ses seviyesi ve değişkenliği
- **MFCC (Mel-Frequency Cepstral Coefficients)**:
  - 13 MFCC katsayısı (ton kalitesi)
  - Her katsayı için mean ve std
- **Prosodic Features (Prosodik Özellikler)**:
  - Speech rate (konuşma hızı): Zero crossing rate bazlı
  - Pause ratio (duraklama oranı): Düşük enerji bölgeleri
  - Tempo (BPM): Beat tracking
- **Voice Activity Detection (VAD)**:
  - Voice activity ratio: Konuşma süresi / toplam süre
  - Dominance score: Konuşma oranı

**Stres ve Kaygı Hesaplama**:
- **Stres Seviyesi**: Pitch variability + energy variability + pause ratio kombinasyonu
- **Güven Skoru**: Düşük pitch variability + yüksek energy + düşük pause ratio → yüksek güven

**Veri Ön İşleme**:
- Ses dosyası yükleme (librosa.load)
- Resample (16 kHz)
- Mono conversion
- Frame-based feature extraction (2048 frame length, 512 hop length)

**Modül**: `src/voice_analyzer.py`

---

### 4. Davranışsal Analiz (Behavioral Analysis)

**Yaklaşım**: Rule-based fusion (kural tabanlı birleştirme)

**Metrikler**:
- **Suspicion Score**: Gaze kaçırma, aşırı sağ-sol bakış, uzun süre kameradan uzaklaşma
- **Risk Level**: Suspicion score'a göre (low/medium/high)
- **Engagement Score**: Eye contact + voice confidence kombinasyonu
- **Anomalies**: Tespit edilen anormal davranışlar listesi

**Kullanılan Sinyaller**:
- Gaze patterns (camera/left/right/up/down oranları)
- Eye contact consistency
- Voice stress level
- Emotion stability

**Modül**: `src/behavioral_analyzer.py`

---

## 🔄 Pipeline Yapısı

### Ana Pipeline (`src/pipeline.py`)

**Akış**:
1. **Video İşleme**: Frame'ler ve ses çıkarılır
2. **Duygu Analizi**: DeepFace ile her 5 frame'de bir analiz
3. **Göz Teması Analizi**: MobileGaze ile her 5 frame'de bir analiz
4. **Ses Analizi**: Librosa ile tüm ses dosyası analiz edilir
5. **Davranışsal Analiz**: Tüm sonuçlar birleştirilir
6. **Rapor Oluşturma**: JSON formatında rapor kaydedilir

**Performans Optimizasyonları**:
- Frame sampling: Her 5 frame'de bir analiz (performans için)
- Model caching: Modeller bir kez yüklenir, tüm frame'ler için kullanılır
- Lazy loading: Modeller gerektiğinde yüklenir

---

## 📊 Çıktı Formatı (JSON Rapor)

```json
{
  "interview_id": "uuid",
  "duration_seconds": 56.62,
  "emotion_analysis": {
    "dominant_emotion": "neutral",
    "emotion_distribution": {...},
    "age_info": {...},
    "gender_info": {...},
    "race_info": {...}
  },
  "eye_contact_analysis": {
    "average_eye_contact_percentage": 29.5,
    "gaze_patterns": {
      "camera_ratio": 0.295,
      "left_ratio": 0.229,
      "right_ratio": 0.050,
      "up_ratio": 0.025,
      "down_ratio": 0.597
    },
    "average_gaze_angle": 36.19
  },
  "voice_analysis": {
    "stress_level": 0.69,
    "confidence_score": 0.28,
    "speech_rate": 7.20,
    "pause_ratio": 0.20
  },
  "behavioral_analysis": {
    "suspicion_score": 0.13,
    "risk_level": "low",
    "anomalies": [...]
  }
}
```

---

## 🛠️ Kullanılan Yöntemler

### Veri Ön İşleme
- **Yüz Tespiti**: MediaPipe Face Detection (yüz bounding box)
- **Normalizasyon**: ImageNet statistics (mean/std)
- **Resize**: Model gereksinimlerine göre (224x224)
- **Audio Preprocessing**: Resample (16 kHz), mono conversion

### Model Inference
- **DeepFace**: TensorFlow/Keras backend
- **MobileGaze**: PyTorch (float64/double precision)
- **Librosa**: NumPy-based signal processing

### Hiperparametreler
- **Frame Sampling Rate**: 5 (her 5 frame'de bir analiz)
- **Gaze Threshold**: 15° (camera sınıfı için)
- **Eye Contact Threshold**: %60 (zaman penceresi bazlı)
- **Audio Sample Rate**: 16 kHz
- **MFCC Coefficients**: 13

### Transfer Learning
- ✅ Tüm modeller pre-trained kullanılıyor
- ❌ Fine-tuning yapılmadı
- ❌ Custom training yapılmadı

---

## 📦 Bağımlılıklar

**Ana Kütüphaneler**:
- `deepface>=0.0.79`: Duygu analizi
- `mediapipe==0.10.7`: Yüz tespiti
- `torch>=2.0.0`: Gaze estimation modeli
- `torchvision>=0.15.0`: Model utilities
- `librosa>=0.10.0`: Ses analizi
- `opencv-python>=4.8.0`: Video işleme
- `numpy>=1.24.0,<2.0.0`: Numerik işlemler
- `scipy>=1.11.4`: Bilimsel hesaplamalar

**Tam liste**: `requirements.txt`

---

## 🚀 Kullanım

### Video Analizi

```bash
python test_example.py video_dosyasi.mp4
```

**Çıktı**: `reports/report_<uuid>.json`

---

## 📝 Notlar

- **Performans**: Frame sampling (her 5 frame) performans için optimize edilmiştir
- **Model Weights**: MobileGaze weights otomatik indirilir (ilk kullanımda)
- **DeepFace Weights**: DeepFace weights otomatik indirilir (ilk kullanımda)
- **Raporlar**: Tüm raporlar `reports/` klasörüne kaydedilir

---

## 🔍 Teknik Detaylar

### Gaze Estimation Model
- **Architecture**: MobileNetV2
- **Input**: 224x224 RGB face crop
- **Output**: 90-bin classification (pitch ve yaw için ayrı)
- **Precision**: Float64 (double)
- **Inference Time**: ~125ms per frame (CPU)

### DeepFace
- **Backend**: TensorFlow/Keras
- **Models**: VGG-Face, FaceNet512 (DeepFace tarafından seçilir)
- **Inference Time**: ~200-300ms per frame (CPU)

### Librosa
- **Sample Rate**: 16 kHz
- **Frame Length**: 2048 samples
- **Hop Length**: 512 samples
- **Processing Time**: Real-time'da çalışır

---

## 📚 Referanslar

- **DeepFace**: https://github.com/serengil/deepface
- **MobileGaze**: https://github.com/yakhyo/gaze-estimation
- **Librosa**: https://librosa.org/
- **MediaPipe**: https://mediapipe.dev/

---

**Son Güncelleme**: 2026-01-08
