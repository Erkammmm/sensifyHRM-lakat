# SensifyHR FAZ-4 — Proje Bitirme Brief'i
# Model: claude-opus-4-6
# Yetki: Tam — izin almadan araştır, karar ver, uygula

---

## Proje Özeti

Video mülakat analiz sistemi. 4 sinyal kaynağı:
1. Yüz duygusu — UniFace DDAMFN (7 sınıf)
2. Göz bakışı — UniFace MobileGaze
3. Ses duygusu — HuBERT SER (şu an sorunlu, değiştirilecek)
4. Konuşma — Whisper STT + torchaudio

Hedef: IK uzmanlarına satılacak ürün kalitesinde davranışsal analiz raporu.

---

## Referans Çıktı Analizi (Bunu Dikkatlice Oku)

### Video: "kamera_hafif_sağda__ekrana_bakıyor__mutlu__motive"
Bu kişi kameraya bakıyor, mutlu ve motive. Beklenen: Happy baskın, Göz Teması Yüksek.

**Gaze:** mean pitch=8.2°, mean yaw=9.1° — tüm segmentler "center", Göz Teması %100 ✅
**Yüz duygusu:** Neutral %48, Surprise %48, Happy %2 — Happy sadece %2, kişi mutlu ama "Happy" çıkmıyor ⚠️
**HuBERT SER:** score=0.255-0.261 (çok düşük, hemen hemen random) valence=0.000 tüm chunklarda ❌
**LLM:** İçerik yok (boş) ❌
**Baskın Duygu:** Surprise %48 — doğru değil ⚠️

---

## Sorun 1: HuBERT SER Tamamen Bozuk

### Kanıt
Tüm chunk'larda score ~0.257 (4 sınıf için random = 0.25) ve valence=0.000.
Model hiçbir şey öğrenemiyor — çıktı tamamen rastgele.

### Olası sebepler
- pipeline() implementasyonu yanlış input formatı gönderiyor
- Model beklediği 16kHz mono float32 yerine farklı format alıyor
- Sessiz/gürültülü chunk'larda model collapse ediyor

### Görevin
1. Önce mevcut audio_signal_fusion.py'daki HuBERT kodunu oku
2. Basit test yaz: tek bir .wav dosyası üzerinde pipeline() çalıştır, raw output gör
3. Score'lar hala ~0.25 çıkıyorsa: model değiştir

### Alternatif Türkçe SER Modelleri (Araştır ve En İyisini Seç)
Şu kriterlere göre değerlendir:
- HuggingFace'de mevcut ve pipeline() ile çalışıyor
- Türkçe konuşma destekli
- GPU ile çalışıyor
- 4+ sınıf (en az: mutlu/üzgün/kızgın/nötr)

Önerilen araştırma kaynakları:
- https://huggingface.co/models?language=tr&pipeline_tag=audio-classification
- https://huggingface.co/SeaBenSea/hubert-large-turkish-speech-emotion-recognition (mevcut)
- Stajyer klasörü: stajyer_kodlari/ — ses ile ilgili torchaudio çalışmaları var, incele

Eğer daha iyi Türkçe model yoksa: çok dilli iyi bir model seç (wav2vec2 tabanlı tercih edilir).
Kararı kendin ver, gerekçeni yaz, uygula.

### torchaudio ile Ses Enerji Analizi (Stajyer Kodlarından)
stajyer_kodlari/ klasöründe torchaudio ile yapılmış ses çalışmaları var.
İncele — özellikle:
- Her soru/cevap segmentinde ses enerjisi (RMS) nasıl değişiyor?
- Konuşma hızı değişimi
- Pitch (f0) zaman içinde nasıl değişiyor?

Bunları dashboard'a ekle: "Ses Enerji Zaman Çizelgesi" olarak.
Her Whisper segmenti için enerji yüksek mi düşük mü göster.

---

## Sorun 2: Yüz Duygusu — Surprise Dominant, Happy Kaçırılıyor

### Kanıt
Mutlu ve motive kişide: Neutral %48, Surprise %48, Happy sadece %2.
UniFace DDAMFN AffectNet7 modeli "mutlu" yüzleri bazen "Surprise" olarak etiketliyor.
Bu AffectNet7'nin bilinen bir limitasyonu — gülümseme + göz açıklığı "Surprise" triggerı.

### Görevin
1. face_analyzer.py'daki emotion confidence dağılımını incele
2. "Surprise" ile "Happy" arasındaki sınır ne zaman karışıyor?
3. Şu kuralı dene: eğer bir frame'de Surprise confidence > 0.55 VE
   bir önceki/sonraki frame'de Happy varsa → Surprise'ı Happy olarak remap et
4. Alternatif: Surprise + Happy birlikte > %60 ise dashboard'da "Pozitif" olarak göster
5. Kararı kendin ver ve uygula — ama modeli değiştirme

---

## Sorun 3: LLM Pipeline — Gemini Önce, Fallback Zinciri

### Mevcut durum
Terminal'de görüyoruz: `llm_provider=gemini` ile istek geliyor ama LLM içeriği boş.
Ollama da bağlantı hatası veriyor.

### İstenen fallback zinciri
```
1. Gemini Pro (ücretli) → .env'deki GEMINI_API_KEY ile
2. Gemini Flash (ücretsiz) → kota dolunca otomatik geç
3. Ollama gemma3:12b → o da başarısız olursa
4. "LLM analizi yapılamadı" → sessizce devam et, rapor yine de çıksın
```

### Görevin
1. src/nlp/gemini.py ve src/nlp/ollama_ai.py dosyalarını oku
2. api/main.py'daki LLM çağrı mantığını oku (DOKUNMA ama anla)
3. Fallback zincirini implement et — hangi dosyaya ekleneceğine sen karar ver
4. .env dosyasında GEMINI_API_KEY var mı kontrol et
5. Gemini model isimleri: "gemini-1.5-pro" (ücretli), "gemini-1.5-flash" (ücretsiz)
6. Her LLM denemesinde timeout: Gemini=30s, Ollama=150s
7. Hata logları açık ve anlaşılır olsun: "[LLM] Gemini Pro başarısız, Flash deneniyor..."

---

## Sorun 4: API'ye LLM Toggle Parametresi Ekle

### İstenen
Swagger UI'dan video yüklerken seçenek:
- `use_llm=true` → LLM analizi yap (yavaş ama kapsamlı)
- `use_llm=false` → LLM atlat, sadece sinyal analizi (hızlı)

### Görevin
api/main.py'a dokunmadan bu özelliği ekle — eğer api/main.py değişmesi gerekiyorsa
önce oku, minimal değişiklik yap, not bırak.

---

## Sorun 5: Dashboard İyileştirmeleri (JSON'dan Karar Ver)

JSON çıktılarını oku (referans dosyalar: reports/ klasöründe veya proje dizininde).
Şu alanlar var ama dashboard'da gösterilmiyor — değerlendир ve ekle:

### Kesin eklenecekler
1. **Ses Enerji Grafiği** — torchaudio RMS değerleri zaman çizelgesi
   Her Whisper segmenti için enerji seviyesi (yüksek/orta/düşük)

2. **Konuşma Yapısı** — text_analysis.thought_units
   Her düşünce bloğu, başlangıç-bitiş zamanı, kısa metin snippet

3. **Ses Duygu Dağılımı** — yeni SER modelinden gelecek
   Şu anki bozuk HuBERT yerine düzgün çalışan model çıktısı

### Değerlendир ve karar ver
JSON'da şu alanlar var — dashboard'a eklenmeye değer mi?
- voice_analysis.timeline: pitch, rms, f0_mean per second
- audio_signal_analysis: valence_state, arousal_state, speech_energy, pitch_stability
- text_analysis.summary: Whisper özeti
- video_info: fps, resolution, duration

IK uzmanı perspektifinden düşün. Gereksiz bilgi ekleme — değerli olanı ekle.

---

## Sorun 6: Glyph Warning — Temizle

Her rapor üretiminde:
```
UserWarning: Glyph 127899 (\N{CONTROL KNOBS}) missing from font(s) DejaVu Sans.
```
plot.py'de bu emoji/unicode karakteri bul ve kaldır veya ASCII ile değiştir.

---

## Çalışma Sırası (Senin Kararın)

Yukarıdaki sorunların öncelik sırasını kendin belirle.
Önerim:
1. HuBERT SER — en kritik, tüm ses analizi buna bağlı
2. LLM fallback zinciri — ürün güvenilirliği
3. LLM toggle parametresi — kullanıcı deneyimi
4. Dashboard iyileştirmeleri — ürün kalitesi
5. Surprise/Happy fix — doğruluk
6. Glyph warning — kozmetik

Ama sen daha iyi gördüğün sırayla yapabilirsin.

---

## Referans Dosyalar

Proje dizininde etiketli referans çıktılar var:
```
kamera_hafif_sağda__ekrana_bakıyor__mutlu__motive.html
kamera_hafif_sağda__ekrana_bakıyor__mutlu__motive.json
rapor_ISINMALI-kameraya-bakiyor-duygular-iyi.html
rapor_ISINMALI-kameraya-bakiyor-duygular-iyi.json
rapor_SORUNLU-kameraya-bakmiyor-uzgun.html
rapor_SORUNLU-kameraya-bakmiyor-uzgun.json
```

Bunları karşılaştırarak çalış. Hangi video için ne beklenmeli, ne çıkıyor — farkı kapat.

---

## Stajyer Kodları

stajyer_kodlari/ klasöründe incelenecek dosyalar:
- Ses ile ilgili torchaudio çalışmaları → ses enerji analizi için kullan
- gaze-emotion-json_sefa.py → UniFace referans implementasyon
- voice_test_said.py → torchaudio referans

---

## Teknik Kısıtlar
- Python: C:/Users/ecetkin/AppData/Local/anaconda3/envs/gpu_env_videoai/python.exe
- PYTHONIOENCODING=utf-8 her komutta
- GPU: CUDA mevcut
- pip install: --break-system-packages flag'i ekle
- Yeni model indirirken: HuggingFace cache'e iner, sabırlı ol
- DOKUNMA: text_analyzer.py, thought_unit_merger.py, api/main.py (minimal değişiklik zorunluysa not bırak)

---

## Son Not

Bu bir ürün. IK uzmanı raporu açtığında:
- İlk 5 saniyede ne anlar?
- Hangi metriğe güvenir?
- Hangi grafik gereksiz?

Bu soruları kendin sor, cevapla ve dashboard'u buna göre optimize et.
Bitirdiğinde değişikliklerin özetini yaz.