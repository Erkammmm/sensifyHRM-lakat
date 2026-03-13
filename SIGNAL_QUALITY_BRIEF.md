# SensifyHR FAZ-4 — Signal Quality & Dashboard Review Brief

## Context: What We Had vs What We Have Now

### FAZ-3 (old)
- Face: MediaPipe blendshape → rule-based emotion (unreliable, noisy, no confidence score)
- Gaze: MediaPipe eyeLookIn/Out → string label only ("Ekrana Bakıyor" etc.), not metric
- Audio: librosa → RMS, pitch, mel spectrogram (CPU-only, slow)
- Audio emotion: HuBERT SER → valence/arousal buckets
- Problem: Too many noisy signals, LLM couldn't reason well, HTML was too technical for HR

### FAZ-4 (current — what you just built)
- Face: UniFace DDAMFN → 7-class emotion + confidence score (real classifier, much better)
- Gaze: UniFace MobileGaze → pitch_deg + yaw_deg in actual degrees (metric, precise)
- Audio ham: torchaudio → RMS dBFS, F0 mean/std, spectral centroid/flatness, mel energy,
  konusma_stili, konusma_guveni (GPU-native, fast)
- Audio emotion: HuBERT SER → valence/arousal (kept from FAZ-3, works well)
- Dashboard: simplified 3-section HTML (KPIs + timeline chart + LLM report)

### What Is Still Not Right

**Problem 1: Raw signals are not being converted to HR-meaningful information**

The pipeline collects good raw data but the contextual_aggregator and prompt still don't
fully convert this into the 5 questions HR actually cares about:

1. Was the candidate nervous? When exactly?
   → Need: tension score per segment = f(emotion in [Angry,Fear,Disgust], low konusma_guveni,
     high f0_std, gaze aversion combined)

2. Did they maintain eye contact?
   → Need: % segments where gaze_direction == "center" vs away
   → Need: sustained aversion detection (3+ consecutive segments looking away)

3. Was their voice steady and confident?
   → Need: speech confidence trend (rising/falling/stable over interview)
   → Need: long silence detection (>5s gaps in is_speech)
   → Need: voice stress indicator = high f0_std + low konusma_guveni combined

4. Did their emotion match their words?
   → Need: incongruence detection already exists but needs refinement:
     positive text sentiment + negative emotion_label (Fear/Angry/Disgust) = incongruence
     high arousal (HuBERT) + neutral emotion (DDAMFN) = possible suppression signal

5. What happened at critical moments?
   → Need: "critical moment" detection = segments where 2+ signals show stress simultaneously

**Problem 2: contextual_aggregator.py needs to compute derived behavioral indicators**

Currently it passes raw averages to LLM. It should also compute:

```python
# Add these computed fields to each segment package:
{
    # existing fields...
    
    # NEW derived behavioral indicators:
    "tension_score": float,        # 0-1: composite stress signal
                                   # = weighted avg of:
                                   #   emotion in [Fear, Angry, Disgust] → 0.4 weight
                                   #   konusma_guveni inverted (1 - value) → 0.3 weight  
                                   #   f0_std normalized → 0.2 weight
                                   #   gaze_direction != "center" → 0.1 weight
    
    "gaze_away": bool,             # True if gaze_direction != "center"
    
    "voice_stress": bool,          # True if f0_std > threshold AND konusma_guveni < 0.6
    
    "incongruence": bool,          # True if emotion vs hubert_valence conflict
                                   # (positive valence + negative emotion_label or vice versa)
    
    "is_critical_moment": bool,    # True if tension_score > 0.6 OR
                                   # (incongruence AND voice_stress)
}
```

**Problem 3: Dashboard needs behavioral summary cards, not just raw KPIs**

Current KPIs: Dominant Emotion | Kamera Odağı % | Avg Speech Confidence | Incongruence Count
These are still too raw for HR.

Replace with behavioral interpretation cards:

```
┌─────────────────────────────────────────────────────────────┐
│  GÖZ TEMASI          │  SES GÜVENİ          │  DUYGUSAL DENGE  │
│  Yüksek              │  Kararlı             │  Dengeli         │
│  %87 kamera odağı    │  Ort. 0.89 güven     │  2 kritik an     │
│  2 kaçınma anı       │  Düşüş: 00:32-00:41  │  00:15, 00:38    │
└─────────────────────────────────────────────────────────────┘
```

Each card has: label (Yüksek/Orta/Düşük) + supporting metric + notable moment timestamp.

**Problem 4: Timeline chart needs improvement**

Current chart shows raw values. For HR it should show:
- Emotion as colored bands (Happy=green, Neutral=gray, Fear/Angry=red, Sad=blue)
- Gaze as binary: IN FRAME (green) vs OUT (red) line
- Speech confidence as a simple line 0-1
- Critical moments marked with a vertical red line + label

---

## Work Order

### Step A — Read and audit current state first
Before changing anything:
1. Read contextual_aggregator.py — check what derived fields are currently computed
2. Read report_generator.py — check current KPI computation and chart data
3. Read prompt_phase3.txt — check if tension/critical moment concepts are mentioned
4. Report what exists vs what's missing from the lists above
5. Do NOT change anything in this step — just audit and report gaps

### Step B — contextual_aggregator.py: add derived behavioral indicators
Add tension_score, gaze_away, voice_stress, incongruence, is_critical_moment to each
segment package. Use the formulas defined above. Keep all existing fields.

### Step C — report_generator.py: improve dashboard
1. Replace raw KPI cards with 3 behavioral interpretation cards (Göz Teması, Ses Güveni, Duygusal Denge)
2. Each card: compute label (Yüksek/Orta/Düşük) from segment data + show key metric + show timestamp of notable moment
3. Improve timeline chart: emotion as colored bands, gaze as binary, confidence as line, critical moments as red markers
4. Keep LLM report section unchanged

### Step D — prompt_phase3.txt: use new derived fields
Update prompt to reference the new derived fields:
- tension_score → "gerilim skoru"
- is_critical_moment → "kritik an"  
- incongruence → "tutarsızlık"
- gaze_away → "göz kaçırma"
LLM should use these pre-computed signals instead of reasoning from raw degrees/labels.

### Step E — Dead code cleanup (from previous brief)
1. audio_analyzer.py → add "# LEGACY: FAZ-2 only" comment at top
2. prompt.txt → add "# LEGACY: FAZ-2" comment at top
3. Scan all files for remaining mediapipe imports → remove
4. Scan all files for remaining librosa imports → report which and why still needed

### Step F — Final end-to-end test
Run: PYTHONIOENCODING=utf-8 python test_example.py "test_videolar/Test_erkam.mp4" --phase3 --ollama
Verify: all 5 HR questions answered in report, dashboard readable, no crashes.

---

## Hard Rules (Unchanged)
- Python: C:/Users/ecetkin/AppData/Local/anaconda3/envs/gpu_env_videoai/python.exe
- Always use PYTHONIOENCODING=utf-8
- DO NOT touch: text_analyzer.py, thought_unit_merger.py, ollama_ai.py, gemini.py, api/main.py, test_example.py
- One step at a time, stop and wait for approval after each step
- HuBERT SER, Whisper, Ollama+Gemma3:12b → unchanged