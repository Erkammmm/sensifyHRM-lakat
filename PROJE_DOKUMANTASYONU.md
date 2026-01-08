# SensifyHR Mülakat Analiz Sistemi - Proje Dokümantasyonu

## 📁 Proje Yapısı

```
sensifyHRMülakay/
├── src/                          # Ana kaynak kod klasörü
│   ├── __init__.py
│   ├── video_processor.py        # Video işleme (frame extraction, audio extraction)
│   ├── emotion_analyzer.py      # Duygu analizi (DeepFace)
│   ├── eye_contact_analyzer.py  # Göz teması analizi (MediaPipe)
│   ├── voice_analyzer.py        # Ses analizi (librosa)
│   ├── behavioral_analyzer.py   # Davranışsal analiz (rule-based)
│   └── pipeline.py              # Ana pipeline (tüm modülleri koordine eder)
├── api/                          # FastAPI REST API
│   └── main.py
├── reports/                      # Analiz raporları (JSON)
├── test_example.py              # Test scripti
├── requirements.txt             # Python bağımlılıkları
└── README.md                    # Genel bilgiler
```

---

## 🔧 Kullanılan Teknolojiler ve Modeller

### 1. Duygu Analizi (Emotion Analysis)

**Model**: DeepFace  
**Kütüphane**: `deepface` (Python)  
**Versiyon**: 0.0.79+

**Kullanılan Özellikler**:
- `DeepFace.analyze()` fonksiyonu ile:
  - **Age** (Yaş): Ortalama, min, max, std
  - **Gender** (Cinsiyet): Dominant cinsiyet, dağılım, güven skoru
  - **Emotion** (Duygu): 7 kategori (angry, disgust, fear, happy, sad, surprise, neutral)
  - **Race** (Irk): Dominant ırk, dağılım, güven skoru

**Veri Seti**: 
- FER2013 (DeepFace'in eğitildiği veri seti)
- 35,887 grayscale yüz görüntüsü
- 7 duygu kategorisi

**Transfer Learning**: 
- ✅ Pretrained model kullanımı (DeepFace)
- ❌ Fine-tuning yapılmadı

**Modül**: `src/emotion_analyzer.py`

---

### 2. Göz Teması Analizi (Eye Contact Analysis)

**Model**: MediaPipe Face Mesh  
**Kütüphane**: `mediapipe` (Python)  
**Versiyon**: 0.10.7

**Kullanılan Özellikler**:
- **468 yüz landmark noktası** tespiti
- **Göz merkezi hesaplama**: Sol ve sağ göz landmark'larının ortalaması
- **Head pose estimation**: `cv2.solvePnP` ile 3D baş pozisyonu (pitch, yaw, roll)
- **Gaze direction calculation**: Göz merkezi ile frame merkezi arası mesafe + head pose

**Hesaplama Yöntemi**:
1. MediaPipe Face Mesh ile 468 landmark tespit edilir
2. Sol ve sağ göz merkezleri hesaplanır
3. Göz merkezi ile kamera merkezi (frame merkezi) arası mesafe hesaplanır
4. Head pose (yaw, pitch) dikkate alınarak gaze angle hesaplanır
5. Eye contact score = `max(0, 1.0 - (angle / 25.0))` (0-25 derece arası iyi)

**Modül**: `src/eye_contact_analyzer.py`

**Çıktı Metrikleri**:
- `average_eye_contact_percentage`: Ortalama göz teması yüzdesi
- `consistency_score`: Göz teması tutarlılığı (düşük std = yüksek tutarlılık)
- `gaze_patterns`: Bakış yönü dağılımı (direct, slight_deviation, moderate_deviation, significant_deviation)
- `average_gaze_angle`: Ortalama gaze açısı (derece)

---

### 3. Ses Analizi (Voice Analysis)

**Kütüphane**: `librosa`  
**Versiyon**: 0.10.0+

**Kullanılan Yöntemler**:
- **Pitch (Perde)**: `librosa.piptrack` (50-400 Hz aralığı)
- **Energy (Enerji)**: RMS (Root Mean Square)
- **MFCC**: 13 katsayı (Mel-Frequency Cepstral Coefficients)
- **Prosodic Features**:
  - Zero Crossing Rate (ZCR)
  - Tempo (BPM)
  - Duraklama tespiti (düşük enerji bölgeleri)
  - Konuşma hızı

**Skor Hesaplama**:
- **Stres Skoru**: Pitch değişkenliği (40%) + Energy değişkenliği (30%) + Duraklama (30%)
- **Güven Skoru**: Pitch stabilitesi (30%) + Energy stabilitesi (20%) + Düşük duraklama (30%) + Optimal konuşma hızı (20%)

**Modül**: `src/voice_analyzer.py`

---

### 4. Davranışsal Analiz (Behavioral Analysis)

**Yöntem**: Rule-based Pattern Recognition + Statistical Analysis

**Kullanılan Teknikler**:
1. **Sliding Window Analizi**:
   - 2-5 saniyelik pencereler
   - Her pencere için: eye_contact_mean, gaze_angle_mean, downward_ratio, saccade_rate

2. **Baseline Öğrenme**:
   - İlk 60 saniye (veya video kısa ise ilk pencereler) baseline olarak kabul edilir
   - Sonraki pencereler baseline'dan sapma (z-score) ile değerlendirilir

3. **Okuma/Cheating Şüphesi**:
   - Yüksek downward_ratio (aşağı bakma) + düşük eye_contact = şüphe
   - Sürekli pattern tespiti (arka arkaya 5+ frame)

4. **Korelasyon Analizi**:
   - Duygu-göz teması korelasyonu
   - Ses-davranış korelasyonu

**Modül**: `src/behavioral_analyzer.py`

**Çıktı Metrikleri**:
- `suspicion_score`: Genel şüphe skoru (0-1)
- `risk_level`: Risk seviyesi (low/medium/high)
- `reading_suspicion`: Okuma şüphesi skoru
- `high_deviation_ratio`: Baseline'dan yüksek sapma oranı

---

## 📊 Rapor Yapısı

### emotion_analysis
- `dominant_emotion`: En sık görülen duygu (İngilizce)
- `dominant_emotion_tr`: En sık görülen duygu (Türkçe)
- `emotion_distribution`: Her duygu için ortalama, min, max, std
- `stability_score`: Duygu değişkenliği (0-1, yüksek = stabil)
- `coverage`: Analiz edilen frame oranı
- `age_info`: Ortalama, min, max, std yaş
- `gender_info`: Dominant cinsiyet, dağılım, güven skoru
- `race_info`: Dominant ırk, dağılım, güven skoru

### eye_contact_analysis
- `average_eye_contact_percentage`: Ortalama göz teması yüzdesi
- `consistency_score`: Göz teması tutarlılığı
- `gaze_patterns`: Bakış yönü dağılımı
- `average_gaze_angle`: Ortalama gaze açısı (derece)
- `coverage`: Analiz edilen frame oranı

### voice_analysis
- `stress_level`: Stres skoru (0-1)
- `confidence_score`: Güven skoru (0-1)
- `speech_rate`: Konuşma hızı
- `pause_ratio`: Duraklama oranı
- `mean_pitch`: Ortalama perde
- `pitch_variability`: Perde değişkenliği

### behavioral_analysis
- `suspicion_score`: Genel şüphe skoru (0-1)
- `risk_level`: Risk seviyesi (low/medium/high)
- `reading_suspicion`: Okuma şüphesi skoru
- `high_deviation_ratio`: Baseline'dan yüksek sapma oranı

### overall_assessment
- `engagement_score`: Genel katılım skoru (0-1)
- `recommendations`: Öneriler listesi

---

## 🛠️ Teknoloji Stack

### Backend
- **Python**: 3.10
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server

### Video İşleme
- **OpenCV**: Video okuma, frame extraction
- **FFmpeg**: Audio extraction (16kHz, mono)

### Deep Learning & Face Analysis
- **DeepFace**: Duygu, yaş, cinsiyet, ırk analizi
- **MediaPipe Face Mesh**: Yüz landmark tespiti (468 nokta)
- **TensorFlow**: DeepFace bağımlılığı (2.13.0)

### Ses İşleme
- **librosa**: Ses özellik çıkarma
- **soundfile**: Ses dosyası işleme
- **scipy**: İstatistiksel analiz

### Veri İşleme
- **NumPy**: Sayısal hesaplamalar
- **Pandas**: Veri manipülasyonu
- **PIL/Pillow**: Görüntü işleme

---

## 📈 Veri Akışı

1. **Video Input** → `VideoProcessor.extract_frames()`
2. **Frame'ler** → `EmotionAnalyzer.analyze_frames()` (DeepFace)
3. **Frame'ler** → `EyeContactAnalyzer.analyze_frames()` (MediaPipe)
4. **Video** → `VideoProcessor.extract_audio()` → `VoiceAnalyzer.analyze_audio()` (librosa)
5. **Tüm Sonuçlar** → `BehavioralAnalyzer.analyze_behavior()` (rule-based)
6. **Final Report** → `Pipeline._generate_report()`

---

## 🎯 Kullanım

```bash
# Test
python test_example.py test_video.mp4

# API (opsiyonel)
uvicorn api.main:app --reload
```

---

## 📝 Notlar

- **İlk çalıştırmada**: DeepFace modelleri indirilecek (~1.6GB)
- **Sample rate**: Her 5 frame'de bir analiz (performans için)
- **Raporlar**: `reports/` klasörüne kaydedilir
- **Modeller**: Pretrained (fine-tuning yapılmadı)
