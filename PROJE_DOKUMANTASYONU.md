# SensifyHR FAZ-4 — Proje Dokümantasyonu

*Sunum ve teknik rapor kaynağı — Türkçe, güncel (2026-03)*

---

## 1. Proje Hedefi

### Problem
Online iş mülakatlarında İK uzmanları adayı yalnızca söylediklerine bakarak değerlendiriyor. Ses tonu, yüz ifadesi, göz teması gibi davranışsal sinyaller değerlendirme dışı kalıyor — ya fark edilmiyor, ya da önyargıyla yorumlanıyor.

### Çözüm
SensifyHR, bir video mülakat kaydını otomatik olarak analiz ederek İK uzmanına **karar destekleyici** bir davranışsal rapor sunar. Sistem karar vermez; gözlemler ve yorumlar.

### Hedef Kullanıcı
- İK uzmanları (teknik bilgi gerektirmez — HTML dashboard yeterli)
- Büyük ölçekli işe alım süreçleri (çok sayıda aday, standart değerlendirme ihtiyacı)

### Temel İlke
> Sistem "elenmeli" veya "uygun değil" yazmaz. Zaman damgalı, sinyal destekli gözlemler sunar.

---

## 2. Sistem Mimarisi

### Pipeline Akışı (5 Adım)

```
[Video .mp4]
     │
     ▼
[Adım 1] VideoProcessor
     • ffmpeg ile ses çıkarma (16kHz mono WAV)
     • Video metadata: fps, çözünürlük, süre
     │
     ▼ (Paralel çalışır)
[Adım 2a] TextAnalyzer          [Adım 2b] VoiceAnalyzer       [Adım 2c] FaceAnalyzer
 faster-whisper STT              torchaudio per-second         UniFace: Her 10. kare
 Türkçe segment üretimi          ses özellikleri               DDAMFN duygu + güven
 [{start,end,text},...]          [{rms,f0,konusma_stili}...]   MobileGaze pitch/yaw
                                                                [{timestamp,emotion,...}]
     │                                [Adım 2d] AudioSignalFusion
     │                                 f0+enerji → ses profili
     │                                 Canlı/Kararlı/Dengeli/Sakin/Gergin
     ▼
[Adım 3] ContextualAggregator
     • Tüm sinyalleri zaman bazlı hizala
     • Gaze offset normalizasyonu (median bias düzeltmesi)
     • build_segment_signal_packages(): her STT segmenti için sinyal paketi
     • build_smart_blocks(): doğal sessizlik sınırlı paragraf blokları
     │
     ▼
[Adım 4] LLM (Gemini → Ollama)
     • 5 sinyal kaynağı + paragraf blokları prompt'a giriyor
     • 7 bölümlü Türkçe davranışsal analiz
     │
     ▼
[Adım 5] ReportGenerator
     • HTML dashboard (Chart.js + Jinja2)
     • JSON rapor (makine tarafından okunabilir)
```

### 4 Sinyal Kaynağı

| Kaynak | Modül | Zaman Çözünürlüğü |
|--------|-------|-------------------|
| Yüz duygusu + güven | FaceAnalyzer | Her 10. kare (~0.33s) |
| Göz bakış yönü | FaceAnalyzer (MobileGaze) | Her 10. kare |
| Ses özellikleri | VoiceAnalyzer | 1 saniyelik pencereler |
| Ses profili / valence | AudioSignalFusion | 3 saniyelik pencereler |

---

## 3. Kullanılan Teknolojiler

| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| PyTorch | 2.5.1+cu121 | Derin öğrenme altyapısı (GPU) |
| torchaudio | 2.5.1+cu121 | Ses sinyal analizi (GPU-native) |
| torchvision | 0.20.1+cu121 | Görüntü işleme |
| UniFace DDAMFN | 3.0.0 | Yüz duygu analizi (7 sınıf, AffectNet7) |
| UniFace MobileGaze | 3.0.0 | Göz bakış tahmini (pitch/yaw derece) |
| UniFace RetinaFace | 3.0.0 | Yüz tespiti |
| faster-whisper | 1.2.1 | Türkçe konuşmadan metne (STT), CTranslate2 |
| Gemini 2.5 Pro/Flash | google-generativeai | Birincil LLM (bulut) |
| Gemma3:12b | Ollama (yerel) | Yedek LLM (offline, veri gizliliği) |
| FastAPI | 0.128.1 | REST API |
| Uvicorn | 0.40.0 | ASGI sunucu |
| Jinja2 | 3.1.6 | HTML rapor template |
| Matplotlib | 3.10.8 | Timeline grafikleri |
| Chart.js | 4.4.0 | Dashboard interaktif grafikleri |
| Python | 3.10 | Dil |
| CUDA | 12.1 | GPU hızlandırma |

---

## 4. Sinyal Kaynakları — Detay

### Yüz Duygusu (DDAMFN AffectNet7)
- **Ne ölçüyor:** Anlık yüz ifadesini 7 sınıfa sınıflandırır (Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral)
- **Nasıl hesaplanıyor:** DDAMFN derin öğrenme modeli, yüz crop üzerinden softmax olasılık dağılımı üretir
- **Güvenilirlik notu:** emotion_confidence < 0.55 → "Neutral" sayılır. AffectNet7'de Happy/Surprise karışıklığı düzeltilmiştir.

### Göz Bakışı (MobileGaze)
- **Ne ölçüyor:** Göz bakış yönünü pitch (yukarı/aşağı) ve yaw (sağ/sol) derece cinsinden ölçer
- **Nasıl hesaplanıyor:** MobileGaze ResNet18, yüz crop üzerinden bakış vektörü tahmin eder
- **Güvenilirlik notu:** Kamera yerleşimine göre sistematik bias oluşabilir. Video geneli median offset ile normalize edilir. center/up/down/left/right eşiği: |pitch| > 20°, |yaw| > 22°

### Ses Profili (torchaudio kural sistemi)
- **Ne ölçüyor:** Konuşma enerjisi ve pitch varyasyonundan ses davranış profili çıkarır
- **Nasıl hesaplanıyor:** Her 3 saniyelik ses parçasında rms_dbfs ve f0_std hesaplanır; kural tablosuna göre 5 profile eşlenir
- **Güvenilirlik notu:** Dil bağımsızdır. Absolute yorumdan kaçınılmalı — kişi ve konuşma stiline göre "Kararlı" farklı anlamlar taşıyabilir. SER modeli (wav2vec2 tabanlı) Türkçe için güvenilir sonuç vermediğinden kaldırılmıştır.

### Konuşma Metni (faster-whisper)
- **Ne ölçüyor:** Söylenen kelimeleri ve zaman damgalarını çıkarır
- **Nasıl hesaplanıyor:** Systran/faster-whisper-large-v3-turbo, CTranslate2 optimizasyonlu, CUDA destekli
- **Güvenilirlik notu:** Türkçe için çok yüksek doğruluk. Arka plan gürültüsünde hata payı artar.

---

## 5. FAZ-3'ten FAZ-4'e Ne Değişti

| Bileşen | FAZ-3 | FAZ-4 | Neden Değiştirildi |
|---------|-------|-------|-------------------|
| Yüz analizi | MediaPipe FaceLandmarker + blendshape kuralları | UniFace DDAMFN + MobileGaze | MediaPipe kural tabanlı → güvenilmez; DDAMFN gerçek sınıflandırıcı |
| Ses özellikleri | librosa (CPU) | torchaudio (GPU-native) | Performans + CUDA entegrasyonu |
| Ses duygu modeli | HuBERT SER (SeaBenSea) → firdhokk → ehcalabres | torchaudio f0+enerji kural sistemi | SER modelleri Türkçe için güvenilmez (sakin konuşma → angry/sad etiket) |
| Konuşma yapısı | 3-35 saniye rastgele thought units | Doğal sessizlik bazlı paragraf blokları (50-200 kelime) | LLM için anlamlı içerik birimi |
| LLM prompt | 4 sinyal kaynağı, 5 çıktı bölümü | 5 sinyal kaynağı, 7 çıktı bölümü | Zaman bloğu içerik analizi eklendi |
| Bakış etiketleri | String ("Ekrana Bakıyor") | Derece bazlı (pitch_deg, yaw_deg) + normalize | Sayısal sinyal → güvenilir eşik uygulanabilir |

---

## 6. Dashboard Bölümleri

### KPI Kartları
Mülakatın özeti — 4 metrik kart:
- **Baskın Duygu:** En sık görülen yüz ifadesi (renk kodlu)
- **Kamera Teması:** Kameraya bakılan süre yüzdesi
- **Konuşma Güveni:** Ortalama konuşma netliği ve sürekliliği
- **Stres Skoru:** Yüz duygusu + pitch varyasyonu + gaze kaçınma bileşik sinyali

### Yüz Duygu Dağılımı
7 duygu sınıfının mülakat boyunca dağılımı. Her dilim, o duygunun toplam süredeki oranını gösterir.

### Ses Profili Dağılımı
5 ses profili (Canlı/Kararlı/Dengeli/Sakin/Gergin) dağılımı. Saf duygu etiketi değil — enerji ve pitch bazlı davranış profili.

### Kritik Anlar
Yüksek stres skoru (> 0.6) veya yüz-ses tutarsızlığı olan anlar kart olarak listelenir. Her kart: zaman damgası + söylenen metin + sinyal değerleri.

### Konuşma Blokları
Mülakat, doğal sessizlik sınırlarında paragraf bloklarına bölünür. Her blok:
- Konuşma metni özeti (ilk 220 karakter)
- Baskın duygu badge'i
- Konuşma güveni badge'i
- Kamera teması badge'i
- Ses valansı badge'i

### LLM Analizi
Gemini veya Ollama'nın ürettiği Türkçe davranışsal rapor. 7 bölüm içerir:
1. Genel Davranışsal Profil
2. Duygusal Seyir
3. Göz Teması ve Dikkat
4. Konuşma Dinamikleri
5. Konuşma İçeriği Analizi (blok bazlı)
6. Tutarsızlık Sinyalleri
7. İK İçin Gözlemler

### Grafikler
5 matplotlib timeline grafiği: yüz duygusu seyri, valence/arousal, gaze yönü, konuşma güveni, ses enerjisi.

---

## 7. LLM Entegrasyonu

### Fallback Zinciri
```
Gemini 2.5 Pro
    └─► başarısız ise → Gemini 2.5 Flash
                └─► başarısız ise → Gemini 2.0 Flash
                        └─► başarısız ise → Ollama (Gemma3:12b)
                                └─► başarısız ise → graceful skip (rapor LLM olmadan devam eder)
```

### Prompt Stratejisi
- Her segment paketi: 5 sinyal + zaman damgası + metin
- Paragraf blokları: konuşma içeriği ile sinyaller birlikte
- Yorumlama kuralları: confidence eşiği, art arda gaze kaçınma, çift sinyal koşulu
- Çıktı kuralları: muğlak dil yasak, tüm gözlemlerde zaman damgası zorunlu

### Bağlam Boyutu
- ~7 dakikalık mülakat: ~183 STT segment, ~7 paragraf bloğu, ~50 AudioSignalFusion chunk
- Gemini 2.5 Pro 1M token context'i ile tüm veri tek seferde işlenir

---

## 8. Performans

| Video Süresi | Toplam Süre | En Uzun Adım |
|-------------|-------------|--------------|
| 3 dakika | ~1.5 dk | FaceAnalyzer (~45s) |
| 5 dakika | ~2.5 dk | FaceAnalyzer (~80s) |
| 10 dakika | ~5 dk | FaceAnalyzer + Whisper (~160s) |

*GPU: NVIDIA, CUDA 12.1, 6+ GB VRAM önerilir*

FaceAnalyzer, her 10. frame'i işler (~3 fps). CPU modunda 3-5x daha yavaştır.

---

## 9. Bilinen Limitasyonlar

| Limitasyon | Açıklama |
|-----------|----------|
| Kamera bağımlılığı | Kamera açısı değiştikçe MobileGaze yanlış kalibrasyon verebilir. Median offset düzeltmesi kısmen telafi eder. |
| Aydınlatma duyarlılığı | Düşük ışıkta veya arka ışıkta DDAMFN duygu tespiti güvenilirliği düşer (confidence < 0.55 → Neutral). |
| Ses profili kişi bağımlılığı | f0/rms eşikleri evrenseldir; konuşmacının doğal tonu dikkate alınmaz. Aynı rms değeri farklı kişiler için farklı anlam taşıyabilir. |
| Türkçe STT gürültüsü | Arka plan sesi veya çoklu konuşmacı durumunda STT hata oranı artar. |
| LLM Türkçe tutarlılığı | Gemini daha tutarlı; Ollama/Gemma3:12b bazen format kurallarını ihlal eder. |
| Tek kişi varsayımı | Pipeline tek adayı işler; çerçevede birden fazla yüz varsa tespit belirsizleşir. |

---

## 10. Gelecek Adımlar

| Öncelik | İyileştirme | Açıklama |
|---------|-------------|----------|
| Yüksek | Kişi bazlı ses kalibrasyonu | İlk 30 saniyeden kişinin f0/rms referansını çıkar, eşikleri adapte et |
| Yüksek | Çok kişili video desteği | Konuşmacı diarizasyonu (pyannote.audio) ile aday/mülakat yapan ayrımı |
| Orta | AffectNet8 testi | 8-sınıf modelin contempt eklenmesinin etkisini ölçek |
| Orta | Gerçek zamanlı mod | WebSocket ile chunk-by-chunk analiz (live interview desteği) |
| Düşük | Beden dili sinyalleri | Omuz postürü, el hareketleri (MediaPipe Pose veya başka model) |
| Düşük | Çoklu dil desteği | İngilizce, Almanca mülakat desteği (STT zaten çok dilli, LLM prompt güncellenmeli) |
