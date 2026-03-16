# SensifyHR FAZ-4 — Self-Directed Optimization Session
# Model: claude-opus-4-6
# Yetki: Tam — izin almadan okuyabilir, değiştirebilir, test edebilir, önerebilir

---

## Sen Kimsin, Ne Yapacaksın

Sen bir AI mühendisi ve ürün optimizasyon uzmanısın.
Sana bir video mülakat analiz sistemi verildi.
Görevin: sistemi incelemek, sorunları bulmak, düzeltmek ve ürün kalitesini artırmak.
Kimseye sormana gerek yok — kendi kararlarını kendin ver ve uygula.
Sadece önemli mimari kararlar için (model değiştirme, büyük refactor) bir not bırak.

---

## Proje: SensifyHR FAZ-4

Bir iş mülakatı videosundan davranışsal analiz çıkaran sistem.
Video → 4 paralel sinyal analizi → LLM yorumu → HTML rapor + JSON çıktı.

### Kullanılan Modeller
| Sinyal | Model | Link |
|--------|-------|------|
| Yüz duygusu (7 sınıf) | UniFace DDAMFN AffectNet7 | https://github.com/yakhyo/uniface |
| Göz bakışı | UniFace MobileGaze | https://github.com/yakhyo/gaze-estimation |
| Ses duygusu (4 sınıf) | HuBERT SER Türkçe | https://huggingface.co/SeaBenSea/hubert-large-turkish-speech-emotion-recognition |
| Konuşma metni | Whisper faster-whisper-large-v3 | — |
| LLM yorum | Gemma3:12b (Ollama) | — |

### Pipeline Akışı
```
video → [Whisper STT] → text_analysis (segments + thought_units)
      → [FaceAnalyzer] → face_analysis.timeline (per-frame: emotion + gaze)
      → [AudioSignalFusion] → audio_signal_analysis.timeline (per-chunk: HuBERT SER + valence/arousal)
      → [VoiceAnalyzer] → voice_analysis (per-second: pitch, rms, speech confidence)
      → [ContextualAggregator] → segment_signal_packages (zaman hizalı paketler)
      → [ReportGenerator] → HTML rapor
      → [OllamaAI] → LLM davranışsal yorum → HTML'e inject
```

### Önemli Dosyalar
```
src/face/face_analyzer.py          ← UniFace wrapper (yüz + göz)
src/audio/audio_signal_fusion.py   ← HuBERT SER + valence/arousal
src/audio/voice_analyzer.py        ← torchaudio pitch/RMS/VAD
src/audio/text_analyzer.py         ← Whisper STT [DOKUNMA]
src/nlp/contextual_aggregator.py   ← 4 sinyali zaman hizalar
src/nlp/ollama_ai.py               ← Gemma3:12b LLM
src/nlp/prompt_phase3.txt          ← LLM prompt
src/reporting/report_generator.py  ← HTML rapor üretir
src/reporting/templates/report_v3.html ← HTML template
src/reporting/plot.py              ← matplotlib grafikler
api/main.py                        ← FastAPI [DOKUNMA]
```

### Dokunma — Bu Dosyalar Değişmez
- src/audio/text_analyzer.py
- src/audio/thought_unit_merger.py
- src/nlp/gemini.py
- api/main.py
- test_example.py

### Çevre
- Python: C:/Users/ecetkin/AppData/Local/anaconda3/envs/gpu_env_videoai/python.exe
- Her komutta: PYTHONIOENCODING=utf-8
- GPU: CUDA mevcut
- Ollama: localhost:11434, model: gemma3:12b

---

## Referans Çıktılar — Bunlara Çok Dikkat Et

Proje dizininde şu dosyalar var (reports/ veya proje kök dizininde):

### ✅ İYİ ÖRNEK — "kameraya-bakiyor-duygular-iyi"
```
rapor_ISINMALI-kameraya-bakiyor-duygular-iyi.html
rapor_ISINMALI-kameraya-bakiyor-duygular-iyi.json
```
Bu kişi:
- Sürekli kameraya bakıyor
- Duygusal olarak dengeli, pozitif konuşuyor
- Rahat ve akıcı konuşma
Beklenen rapor çıktısı: Göz Teması=Yüksek, Duygusal Denge=Dengeli, Ses Güveni=Kararlı

### ❌ SORUNLU ÖRNEK — "kameraya-bakmiyor-uzgun"
```
rapor_SORUNLU-kameraya-bakmiyor-uzgun.html
rapor_SORUNLU-kameraya-bakmiyor-uzgun.json
```
Bu kişi:
- Ekrana pek bakmıyor, göz kaçırıyor
- Üzgün/gergin görünüyor
- Konuşması çekingen
Beklenen rapor çıktısı: Göz Teması=Düşük, Duygusal Denge=Gergin, Ses Güveni=Zayıf/Orta

---

## Görevin — Adım Adım

### ADIM 1: Sistemi Tanı
1. Tüm kaynak dosyaları oku (yukarıdaki listeden)
2. Her iki referans raporu aç ve incele (HTML + JSON)
3. Şu soruları yanıtla (sadece kendi notların için — bir yere yaz):
   - İyi örnek raporda KPI kartları doğru mu? (Göz Teması Yüksek çıkıyor mu?)
   - Sorunlu örnekte KPI kartları doğru mu? (Göz Teması Düşük çıkıyor mu?)
   - JSON'da face_analysis.timeline → gaze_pitch_deg/gaze_yaw_deg değerleri ne aralıkta?
   - JSON'da audio_signal_analysis.timeline → debug.ser_top_label değerleri mantıklı mı?
   - segment_signal_packages → gaze_direction değerleri gerçekten değişiyor mu?
   - LLM çıktısı var mı, timestamp içeriyor mu, somut mu?
   - Dashboard'da eksik/yanlış bir şey var mı?
   - JSON'da görüp dashboard'a eklenmemiş değerli veri var mı?

### ADIM 2: Sorunları Tespit Et
Her sorun için yaz:
- Nerede: hangi dosya/fonksiyon
- Ne: tam olarak ne yanlış
- Neden: kök sebep
- Etki: kullanıcı ne görüyor, ne görmeli

Özellikle kontrol et:
- Gaze thresholds: pitch/yaw eşikleri gerçekçi mi? Her video için farklı kamera açısı var.
- HuBERT label mapping: model "sadness" döndürüyor, kod "sad" mi bekliyor?
- HuBERT pipeline kullanımı: AutoModel mi yoksa pipeline() mi kullanılıyor? pipeline() daha doğru.
- Sessiz chunk'lar: RMS=0 olan chunk'lar SER dağılımına dahil ediliyor mu?
- KPI mantığı: "Duygusal Denge" kartı gerçekten duygu dağılımına mı bakıyor?
- Ollama: num_predict/max_tokens var mı? Varsa kaldır.

### ADIM 3: Düzelt
Tespit ettiğin her sorunu düzelt. Sıra önerim:
1. Ollama parametreleri (hızlı, kritik)
2. HuBERT pipeline() yeniden yazımı (doğruluk kritik)
3. Gaze threshold'ları (görünürlük kritik)
4. KPI mantığı (sunum kritik)
5. Diğer bulduğun her şey

Her düzeltme sonrası basit bir test çalıştır ve sonucu kaydet.

### ADIM 4: İyileştir
JSON çıktılarına bak. Dashboard'a eklenebilecek değerli veriler:
- text_analysis.thought_units → "Konuşma Yapısı" bölümü
- audio_signal_analysis → valence/arousal trend
- face_analysis → emotion confidence zaman içinde değişimi
- Sessizlik anları (Whisper segment gap'leri)
- Konuşma hızı (kelime/dakika)
- Herhangi başka gördüğün değerli veri

Ürün perspektifinden düşün: Bu raporu gören bir IK uzmanı için en değerli bilgi nedir?
O bilgiyi öne çıkar, gereksiz tekrarları kaldır.

### ADIM 5: Prompt Optimizasyonu
src/nlp/prompt_phase3.txt dosyasını oku.
Sonra segment_signal_packages yapısına bak.
LLM'e giden veriyle prompt'un örtüşüp örtüşmediğini kontrol et.
Eksik alan referansı, yanlış field adı, ya da fırsat kaçırılan sinyal varsa düzelt.
Prompt'un ürettiği çıktı kalitesini artır: daha somut, daha az hedging, daha fazla timestamp.

### ADIM 6: Final Test
uvicorn api.main:app --host 0.0.0.0 --port 8000 ile API'yi başlat.
Mümkünse her iki referans videoyu test et.
Rapor çıktılarını kontrol et:
- İyi örnek: Göz Teması Yüksek, Duygusal Denge Dengeli?
- Sorunlu örnek: Göz Teması Düşük, Duygusal Denge Gergin?
- LLM zaman aşımı var mı?
- Tüm dashboard bölümleri görünüyor mu?
Bulguları özetle.

---

## Ürün Hedefi

Bu sistem IK uzmanlarına satılacak bir ürün.
Şu soruları sor kendine: "Bu raporu gören bir IK uzmanı..."
- İlk 5 saniyede ne anlar?
- Hangi bilgiye güvenir, hangisine güvenmez?
- Hangi bölüm gereksiz/kafa karıştırıcı?
- Hangi bilgi eksik ama çok işe yarardı?

Bu perspektiften gördüğün her iyileştirmeyi yap veya not bırak.

---

## Teknik Kısıtlar
- GPU env: gpu_env_videoai
- PYTHONIOENCODING=utf-8 her komutta
- pip install gerekirse: --break-system-packages flag'i ekle
- Yeni model indirme: HuggingFace cache'e iner, ilk seferde yavaş olabilir
- Ollama çalışıyor olmalı: `ollama serve` (zaten çalışıyorsa atla)
- Plot.py'deki emoji/glyph uyarıları zararsız, susturmaya çalışma

---

## Notlar
- Her şeyi kendin yap, izin alma
- Yaptığın önemli değişiklikleri kısa bir liste halinde sonunda özetle
- Eğer bir şeyin kök sebebini bulamazsan: "Bulamadım, şunu denedim" diye yaz ve devam et
- Git commit yapma — değişiklikleri yap, test et, özetle