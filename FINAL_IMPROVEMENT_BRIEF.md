# SensifyHR FAZ-4 — Final Quality & Accuracy Improvement Brief

## Reference Links (Use These When Investigating)
- UniFace gaze estimation: https://github.com/yakhyo/gaze-estimation
- UniFace full docs: https://yakhyo.github.io/uniface/
- UniFace GitHub: https://github.com/yakhyo/uniface
- HuBERT SER model: https://huggingface.co/SeaBenSea/hubert-large-turkish-speech-emotion-recognition
- torchaudio docs: https://pytorch.org/audio/stable/index.html

---

## Current State (What Works vs What Doesn't)

### Working ✅
- UniFace DDAMFN: 7-class face emotion with confidence — data is real and meaningful
- HuBERT SER: remap fix applied, valence/arousal now varies correctly
- Whisper STT: speech stats working
- Pipeline speed: ~70s for 3-minute video — acceptable
- HTML structure: 4 sections, no double render

### Broken or Missing ❌
- Gaze tracking: MobileGaze produces pitch/yaw but gaze_direction is ALWAYS "center" → wrong
- KPI cards: "Duygusal Denge: Dengeli" even when Sad=96.7% — logic is wrong
- HuBERT emotion distribution: 4-class labels (Angry/Calm/Happy/Sad) per chunk exist but NEVER shown in dashboard
- Ollama LLM: slow, needs parameter tuning
- Prompt: needs complete rewrite focused on 4-signal behavioral analysis

---

## Problem 1: Gaze Tracking — Complete Fix Required

### Reference
- https://github.com/yakhyo/gaze-estimation
- https://yakhyo.github.io/uniface/
- https://github.com/yakhyo/uniface

### What's happening now
MobileGaze returns pitch_deg and yaw_deg values. The gaze_direction computation uses
thresholds (|pitch|>15°, |yaw|>20°) BUT gaze_direction is ALWAYS "center" in practice
even when the person clearly looks away from the camera.

### Investigation required FIRST (Step 1 — no code changes)
1. Read face_analyzer.py — how is pitch/yaw extracted from MobileGaze?
2. Print actual pitch/yaw values from a real test video — show min/max/mean/std
3. Read uniface gaze estimation repo carefully: https://github.com/yakhyo/gaze-estimation
4. Determine: coordinate system, value ranges, what "center" means in this model
5. Check if values need to be negated, scaled, or have a different zero reference

### The fix (Step 2 — after investigation)
- Set thresholds based on OBSERVED value ranges from real video data
- gaze_direction MUST vary: person looking away must get "right"/"left"/"down"
- If MobileGaze coordinate system is incompatible, implement alternative from the repo

---

## Problem 2: KPI Card Logic — Fix Based on Real Signals

### Fix "Duygusal Denge" card
Use face emotion distribution directly, not just critical_moment count:

```python
negative_emotions = {"Sad", "Fear", "Angry", "Disgust"}
negative_pct = percentage of face frames with negative emotion (confidence >= 0.55)

if negative_pct > 60%:   label = "Gergin",  color = danger (red)
elif negative_pct > 35%: label = "Dikkat",  color = warn (amber)
else:                    label = "Dengeli", color = accent2 (green)
```

### Add 4th KPI card: "Baskın Duygu"
Most important single signal for HR. Show:
- Top emotion label from face distribution (confidence-filtered)
- Its percentage: e.g., "Sad — %96.7"
- Color: green for Happy/Neutral, red for Sad/Fear/Angry/Disgust

### "Göz Teması" card
Will auto-fix once gaze_direction actually varies (Problem 1 fix).

---

## Problem 3: HuBERT Voice Emotion Distribution — Add to Dashboard

### What we have
Model: SeaBenSea/hubert-large-turkish-speech-emotion-recognition
Reference: https://huggingface.co/SeaBenSea/hubert-large-turkish-speech-emotion-recognition

The model produces 4-class emotion labels per audio chunk:
- Angry, Calm, Happy, Sad

This data is stored in: audio_signal_analysis.timeline[].top_label (in JSON report)

### What to add in report_generator.py
Compute voice emotion distribution:
```python
# Count top_label occurrences across all audio chunks
# Compute percentage for each of 4 classes
voice_emotion_dist = {
    "Angry": X%,
    "Calm":  Y%,
    "Happy": Z%,
    "Sad":   W%
}
```

### Display in dashboard
Add new chart section "Ses Duygu Dağılımı (HuBERT)" immediately after face emotion distribution:
- Same horizontal bar chart style as face emotion distribution
- Colors: Calm/Happy = green, Angry = red, Sad = blue
- Title: "Ses Duygu Dağılımı (HuBERT SER — Türkçe)"
- Subtitle: "4 sınıf: Angry / Calm / Happy / Sad · SeaBenSea/hubert-large-turkish-speech-emotion-recognition"

Also add "Baskın Ses Duygusu" as sub-line under "Ses Güveni" KPI card.
Example: "Baskın: Calm %68 | Sad %24"

---

## Problem 4: Ollama Speed — Parameter Tuning

### Read ollama_ai.py first, then tune:
- max_tokens: reduce to 800 (focused output, no rambling)
- temperature: set to 0.3 (deterministic, less creative hedging)
- Add 90-second timeout with graceful fallback message: "LLM analizi zaman aşımına uğradı — lütfen tekrar deneyin"
- If segment_signal_packages is large (>10 segments): summarize to first+last+3 middle segments before sending to LLM

DO NOT change the model (Gemma3:12b stays) or remove Ollama.

---

## Problem 5: Prompt — Complete Rewrite

### File: src/nlp/prompt_phase3.txt
Rewrite completely. New content:

```
ROL:
Sen deneyimli bir davranış analistisin. Sana bir iş mülakatından çıkarılmış
4 farklı sinyal kaynağından veri veriliyor. Bu verileri kullanarak adayın
davranışsal profilini çıkar. Karar verme — sadece gözlemle ve yorumla.

GİRDİ — 4 SİNYAL KAYNAĞI:

1. YÜZ DUYGUSU (UniFace DDAMFN AffectNet7 — 7 sınıf)
   - dominant_emotion: Happy | Sad | Angry | Fear | Disgust | Surprise | Neutral
   - emotion_confidence: 0-1 (0.55 altı → güvenilmez, yorumlama)
   - tension_score: 0-1 (0.6 üstü → yüksek gerilim anı)
   - is_critical_moment: true/false

2. GÖZ BAKIŞI (UniFace MobileGaze)
   - gaze_direction: center | up | down | right | left
   - gaze_away: true/false (center dışı = kaçınma)
   - avg_gaze_pitch, avg_gaze_yaw: derece cinsinden ham değer

3. SES DUYGUSU (HuBERT SER — SeaBenSea Türkçe modeli, 4 sınıf: Angry/Calm/Happy/Sad)
   - hubert_valence: -1 ile +1 (-1 = çok olumsuz, +1 = çok olumlu)
   - hubert_arousal: -1 ile +1 (-1 = sakin, +1 = heyecanlı/gergin)

4. KONUŞMA ÖZELLİKLERİ (torchaudio VAD + pitch analizi)
   - speech_style: sakin | dengeli | canlı | karışık
   - speech_confidence: 0-1 (konuşma netliği ve sürekliliği)
   - f0_mean: temel frekans Hz (50-150 = düşük/monoton, 150-300 = normal, 300+ = yüksek/gergin)
   - konusma_enerjisi: düşük | orta | yüksek

YORUMLAMA KURALLARI:
- emotion_confidence < 0.55 → o segmenti "belirsiz" say, yorumlama
- tension_score > 0.6 → yüksek gerilim anı, raporda işaretle
- is_critical_moment = true → mutlaka Duygusal Seyir ve Tutarsızlık bölümlerinde ele al
- incongruence = true → yüz ve ses çelişiyor, Tutarsızlık bölümünde belirt
- gaze_away art arda 3+ segmentte → dikkat dağınıklığı veya kaçınma örüntüsü
- hubert_valence < -0.5 → ses tonu belirgin olumsuz, yorumla
- voice_stress = true → yüksek pitch varyasyonu + düşük konuşma güveni birlikte
- Minimum 3 segment olmadan genel örüntü iddiasında bulunma

ÇIKTI KURALLARI — BUNLARA KESİNLİKLE UY:
1. Her gözlem için MM:SS formatında zaman damgası zorunlu — damgasız gözlem yazma
2. Muğlak dil YASAK: "olabilir", "belki", "düşündürebilir", "olasılık dahilinde" kullanma
3. Somut yaz: "02:15'te ses güveni 0.85'ten 0.42'ye düştü, eş zamanlı göz kaçırma var"
4. Yeterli veri yoksa: "Bu bölüm için yeterli sinyal yok" yaz ve geç — tahmin yapma
5. Türkçe, profesyonel ton, bir IK uzmanına raporluyorsun
6. İşe alım kararı verme, "elenmeli" veya "uygun değil" yazma
7. 4 sinyal kaynağını birlikte değerlendir — tek kaynaktan yorum yapma

ÇIKTI FORMATI — Aşağıdaki 6 başlığı aynen kullan:

## Genel Davranışsal Profil
3 boyutta, her biri 1 somut cümle + sinyal değeri + timestamp:
- Özgüven: [gözlem + en düşük/yüksek speech_confidence değeri + timestamp]
- İletişim tutarlılığı: [yüz-ses-göz uyumu veya çelişkisi + timestamp]
- Stres yönetimi: [tension_score pik değeri + ne zaman + nasıl değişti]

## Duygusal Seyir
- Mülakat boyunca yüz duygusu nasıl değişti? (her değişimde timestamp)
- Baskın duygu neydi ve toplam sürenin yüzde kaçında görüldü?
- En belirgin duygu geçişi hangi anda yaşandı?

## Göz Teması ve Dikkat
- Kamera teması kalitesi (% tahmini, art arda kaçınma var mı?)
- Göz kaçırma anları: timestamp + kaç segment sürdü
- Gaze ve konuşma güveni aynı anda düştüğü anlar (çift sinyal)

## Konuşma Dinamikleri
- Ses enerjisi ve güveni zaman içinde nasıl değişti?
- f0_mean düşük mü yüksek mi — monotonluk veya gerginlik belirtisi?
- Sessizlik veya konuşma kopukluğu var mıydı? (timestamp)

## Tutarsızlık Sinyalleri
- Yüz duygusu ile HuBERT ses tonu çelişen anlar (timestamp + hangi sinyaller çelişiyor)
- Göz kaçırma + düşük ses güveni eş zamanlı anlar (timestamp)
- Hiç tutarsızlık yoksa: "Bu mülakatta belirgin tutarsızlık sinyali gözlemlenmedi" yaz

## IK İçin Gözlemler
Maksimum 3 madde. Her madde = tam olarak 1 cümle, aksiyon odaklı, timestamp içermeli.
Format zorunlu: "[MM:SS] — [Ne gözlemlendi] → [IK için ne yapılabilir]"
Örnek: "[02:30] — Ses güveni ani düşüş + göz kaçırma eş zamanlı → Bu konuyu derinlemesine araştırın."
```

---

## Problem 6: Matplotlib Signal Charts — Replace Crowded Chart.js

Current Chart.js combo chart has 4 datasets — too crowded for HR.
Replace with 4 separate matplotlib PNG charts embedded in HTML as base64.

### Chart 1 — Yüz Duygu Zaman Çizelgesi
- X: zaman (saniye), Y: emotion as colored horizontal bands per segment
- Colors: Happy/Surprise=green, Neutral=gray, Sad=blue, Fear/Angry/Disgust=red
- Opacity = emotion_confidence

### Chart 2 — Ses Duygu (HuBERT Valence) Zaman Çizelgesi
- X: zaman, Y: hubert_valence line (-1 to +1)
- Zero reference line in gray
- Fill: green above zero, red below zero

### Chart 3 — Göz Bakış Zaman Çizelgesi
- X: zaman, Y: gaze_direction as colored bands per segment
- center=green, up/down/right/left=red/orange

### Chart 4 — Konuşma Güveni Zaman Çizelgesi
- X: zaman, Y: speech_confidence line (0-1)
- Red zone below 0.5
- Mark is_critical_moment segments with vertical red line

Generate all 4 in plot.py. Embed as base64 PNG in HTML.
Remove the complex Chart.js combo chart entirely.

---

## Work Order

**Step 1 — Investigate gaze values (READ ONLY, no code changes)**
Print actual pitch/yaw min/max/mean/std from test_videolar/Test_erkam.mp4.
Read https://github.com/yakhyo/gaze-estimation for coordinate system.
Report findings. Stop and wait.

**Step 2 — Fix gaze_direction computation**
Based on Step 1 findings, fix thresholds.
Test: run on video → gaze_direction must vary.
Stop and wait.

**Step 3 — Fix KPI card logic**
- Fix "Duygusal Denge" using negative_pct from face emotion distribution
- Add "Baskın Duygu" as 4th KPI card
Stop and wait.

**Step 4 — Add HuBERT voice emotion distribution chart**
Read top_label from audio_signal_analysis.timeline.
Add horizontal bar chart "Ses Duygu Dağılımı".
Add baskın ses duygusu to "Ses Güveni" card.
Stop and wait.

**Step 5 — Tune Ollama parameters**
Read ollama_ai.py → reduce max_tokens to 800, temperature to 0.3, add timeout.
Stop and wait.

**Step 6 — Rewrite prompt_phase3.txt**
Use exact prompt content defined in Problem 5 above.
Stop and wait.

**Step 7 — Replace Chart.js with 4 matplotlib charts**
Add to plot.py, embed in HTML as base64.
Remove Chart.js combo chart.
Stop and wait.

**Step 8 — Final end-to-end test via API**
uvicorn start → POST /analyze?phase3=true&llm_provider=ollama
Verify all 8 success criteria:
1. gaze_direction actually varies
2. "Duygusal Denge" = "Gergin" for Sad=96.7% video
3. "Baskın Duygu" card shows Sad %96.7
4. HuBERT distribution chart visible with 4 bars
5. LLM output has MM:SS timestamps, no hedging
6. 4 separate matplotlib charts visible
7. No crashes
8. Total time (pipeline + LLM) under 3 minutes
Stop and report full results.

---

## Hard Rules (Unchanged)
- Python: C:/Users/ecetkin/AppData/Local/anaconda3/envs/gpu_env_videoai/python.exe
- Always use PYTHONIOENCODING=utf-8
- DO NOT touch: text_analyzer.py, thought_unit_merger.py, gemini.py, api/main.py, test_example.py
- ollama_ai.py: READ ONLY for Step 5 param tuning — do not change logic
- One step at a time, stop and wait for approval after each step
- If a step reveals unexpected complexity — STOP and report, do not improvise