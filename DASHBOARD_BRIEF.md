# SensifyHR FAZ-4 — Dashboard & Signal Quality Improvement Brief

## Context
Pipeline runs end-to-end. All modules work. But the output quality is not good enough for HR.
This brief defines the next set of improvements.

---

## Problem 1: HuBERT SER Weight Warning (Fix First)

On every run we see:
```
Some weights of HubertForSequenceClassification were not initialized from the model checkpoint
at SeaBenSea/hubert-large-turkish-speech-emotion-recognition and are newly initialized:
['classifier.bias', 'classifier.weight', 'projector.bias', 'projector.weight']
You should probably TRAIN this model on a down-stream task...
[HuBERTSerProjector] UYARI: classifier/projector ağırlıkları eksik görünüyor. Remap ile tekrar yükleniyor...
```

**Investigate and fix in audio_signal_fusion.py:**
- Check how the model is loaded — is it using the correct `from_pretrained` with the right
  model class? SeaBenSea model may require a specific loading approach.
- Check if the remap logic actually works — does valence/arousal output make sense after remap?
- If remap works correctly, suppress the warning cleanly (it's misleading to users).
- If remap does NOT work correctly, fix the root cause loading method.
- Test: after fix, HuBERT should produce non-zero, varying valence/arousal values across segments.

---

## Problem 2: Dominant Emotion = "Surprise" Always (Calibration Bug)

UniFace DDAMFN produces "Surprise" for almost every frame even on neutral faces.
This is a known AffectNet7 calibration issue — the model is not wrong, but Surprise threshold
is too sensitive for frontal interview footage.

**Fix in contextual_aggregator.py and/or face_analyzer.py:**
- Apply confidence filtering: if emotion_confidence < 0.55 → treat as "Neutral"
- Apply frequency filtering: if an emotion appears in >80% of frames with low avg confidence
  (<0.65), demote it to "Neutral" for that segment
- This will give much more realistic emotion distribution

---

## Problem 3: HTML Report Bug

The LLM analysis renders TWICE in the HTML:
1. Inside `.ai-section-card` div → shows "Analiz metni mevcut değil" (empty)
2. Inside a second `.ai-analysis` div → shows the actual LLM text

**Fix in report_generator.py and report_v3.html template:**
- Find where ai_analysis text is injected — there are two render paths
- Keep only one: the `.ai-section-card` section with proper formatting
- Remove the duplicate `.ai-analysis` div entirely

---

## Problem 4: Dashboard Needs 3 New Data Sections

We have 3 powerful signal sources that are NOT shown in the dashboard:

### 4A. Emotion Distribution Chart (UniFace — 7 class)
We have frame-by-frame emotion data. Show it as a horizontal bar chart:
```
Neutral  ████████████████░░░░  62%
Happy    ██████░░░░░░░░░░░░░░  21%
Sad      ███░░░░░░░░░░░░░░░░░   9%
Fear     ██░░░░░░░░░░░░░░░░░░   8%
```
- Filter: only count frames where emotion_confidence >= 0.55
- If filtered count < 10% of total frames → show "Yeterli veri yok"
- Add this as a new card/section between Section 1 (cards) and Section 2 (timeline)

### 4B. Voice Emotion Trend (HuBERT valence over time)
HuBERT produces valence/arousal per chunk but it's invisible in the dashboard.
Add a simple line to the existing timeline chart:
- New dataset: hubert_valence over time (normalize to 0-1 range for display)
- Color: purple/violet line
- Label: "Ses Tonu" in legend
- This makes yüz duygu vs ses tonu comparison visible on same chart

### 4C. Speech Statistics from Whisper (NEW — currently unused)
We have 309 Whisper segments with timestamps. Extract and show:
- Total speech time vs silence time (pie or bar)
- Number of silence gaps > 5 seconds (with timestamps)
- Average words per segment (speaking pace indicator)
- Longest silence gap (timestamp + duration)

Compute these in report_generator.py from the `text_analysis` field in the JSON report.
Show as a 4th behavioral card row or a compact stats section.

Example display:
```
┌─────────────────────────────────────────────────────────┐
│  KONUŞMA ANALİZİ (Whisper'dan)                         │
│  Toplam konuşma: 9dk 42sn  │  Sessizlik: 1dk 18sn      │
│  Uzun sessizlik (>5sn): 3x  │  En uzun: 8.2sn @ 03:24  │
│  Konuşma hızı: Normal (orta yoğunluk)                  │
└─────────────────────────────────────────────────────────┘
```

---

## Problem 5: LLM Prompt — Make It More Concrete

The LLM output is too vague ("daha fazla araştırmaya ihtiyaç duyulabilir").
HR needs specific, timestamped, actionable observations.

**Update prompt_phase3.txt:**
- Add instruction: "Her gözlem için mutlaka bir zaman damgası (MM:SS formatında) ver"
- Add instruction: "Muğlak ifadeler kullanma. 'Stres gözlemlendi' yerine '02:15-02:45 arasında
  ses güveni 0.6'dan 0.3'e düştü, eş zamanlı göz kaçırma tespit edildi' gibi somut yaz"
- Add instruction: "HR İçin Öneriler bölümünde maksimum 3 madde, her madde 1 cümle, aksiyon odaklı"
- Add instruction: "Genel Davranışsal Profil bölümünü şu 3 boyutta özetle: Özgüven seviyesi,
  İletişim tutarlılığı, Stres yönetimi"
- Remove: any instruction that leads to "belki", "olabilir", "düşündürebilir" type hedging
  Replace with: "Eğer yeterli veri yoksa o bölümü 'Yeterli sinyal yok' diyerek geç"

---

## Work Order

**Step A — Fix HuBERT weight warning (audio_signal_fusion.py)**
Investigate loading, fix or suppress cleanly. Test valence output is non-zero and varies.
Stop and wait.

**Step B — Fix Surprise calibration (face_analyzer.py or contextual_aggregator.py)**
Apply confidence + frequency filtering. Test on existing JSON data if possible.
Stop and wait.

**Step C — Fix HTML double-render bug (report_generator.py + template)**
Find both render paths, keep one, remove duplicate.
Stop and wait.

**Step D — Add emotion distribution chart (report_generator.py + template)**
Horizontal bar chart from face timeline data. Confidence-filtered.
Stop and wait.

**Step E — Add HuBERT valence line to timeline chart**
Add as new dataset to existing Chart.js chart.
Stop and wait.

**Step F — Add Whisper speech statistics section**
Compute from text_analysis segments. Show as compact stats card.
Stop and wait.

**Step G — Update prompt_phase3.txt**
More concrete, timestamped, less hedging.
Stop and wait.

**Step H — Final end-to-end test**
Run full pipeline on Test_erkam.mp4.
Verify: no HuBERT warning, realistic emotion distribution, no double render,
all 3 new sections visible, LLM output has timestamps.

---

## Hard Rules (Unchanged)
- Python: C:/Users/ecetkin/AppData/Local/anaconda3/envs/gpu_env_videoai/python.exe
- Always use PYTHONIOENCODING=utf-8
- DO NOT touch: text_analyzer.py, thought_unit_merger.py, ollama_ai.py, gemini.py, api/main.py, test_example.py
- One step at a time, stop and wait for approval after each step
- HuBERT SER, Whisper, Ollama+Gemma3:12b → unchanged models