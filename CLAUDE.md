# SensifyHR FAZ-4 — Claude Code Master Guide

## Project Overview
AI-powered video interview analysis system. Analyzes a candidate's video recording by extracting
face/emotion signals, gaze tracking, speech features, and transcribed text — aligns them into
time-based segment packages — and feeds them to an LLM to generate an HR evaluation report.

**This is a FAZ-3 → FAZ-4 migration.** The existing working pipeline must be preserved.
Only the weak/outdated modules are being replaced. Do NOT touch anything outside the scope below.

---

## Repository Structure
```
SensifyHR-FAZ3/
├── src/
│   ├── vision/
│   │   ├── face_analyzer.py          # ← REPLACE (MediaPipe → UniFace)
│   │   └── video_processor.py        # DO NOT TOUCH
│   ├── audio/
│   │   ├── audio_signal_fusion.py    # ← UPDATE (librosa → torchaudio, keep HuBERT)
│   │   ├── voice_analyzer.py         # ← REPLACE (librosa → torchaudio)
│   │   ├── text_analyzer.py          # DO NOT TOUCH (Whisper STT, works fine)
│   │   └── thought_unit_merger.py    # DO NOT TOUCH
│   ├── nlp/
│   │   ├── contextual_aggregator.py  # ← UPDATE (new signal formats)
│   │   ├── ollama_ai.py              # DO NOT TOUCH
│   │   ├── prompt_phase3.txt         # ← UPDATE (new signal names + interpretation guide)
│   │   └── prompt.txt                # DO NOT TOUCH
│   ├── reporting/                    # DO NOT TOUCH (entire folder)
│   └── pipeline.py                   # ← UPDATE (rewire new modules)
├── stajyer_kodlari/
│   ├── gaze-emotion-json_sefa.py     # Reference implementation: UniFace face/emotion/gaze
│   └── voice_test_said.py            # Reference implementation: torchaudio audio analysis
├── api/main.py                       # DO NOT TOUCH
├── test_example.py                   # DO NOT TOUCH
└── requirements.txt                  # ← UPDATE (remove mediapipe, add uniface)
```

---

## What Needs to Change and Why

### 1. face_analyzer.py → Replace with UniFace

**Current problem:**
- Uses MediaPipe FaceLandmarker with blendshape scores
- Rule-based emotion inference (unreliable, noisy)
- No actual emotion classification — only proxy signals (facial_state, attention_state, stress_indicator)

**New implementation:**
- Reference: `stajyer_kodlari/gaze-emotion-json_sefa.py` (written by intern, may be incomplete)
- If the intern code is incomplete, refer to the official repo: https://github.com/yakhyo/uniface
- Use **RetinaFace** for face detection
- Use **DDAMFN AffectNet7** for 7-class emotion classification with confidence score
- Use **MobileGaze** for gaze estimation (pitch_deg, yaw_deg in degrees)
- Process every 10th frame for performance (already in intern code)
- **Critical addition missing from intern code:** Convert frame_index → timestamp_sec using video FPS

**Gaze estimation note:**
- Also evaluate **GaZeL**: https://github.com/fkryan/gazelle
- Compare MobileGaze vs GaZeL stability on a short test clip
- Use whichever gives more stable/reliable results
- Document your choice in ARCHITECTURE.md with reasoning

**Required output format (per frame):**
```python
{
    "timestamp_sec": float,
    "emotion_label": str,        # "Happy" | "Sad" | "Angry" | "Fear" | "Disgust" | "Surprise" | "Neutral"
    "emotion_confidence": float, # 0.0 – 1.0
    "gaze_pitch_deg": float,     # positive = looking up, negative = looking down
    "gaze_yaw_deg": float,       # positive = right, negative = left
    "face_detected": bool
}
```

---

### 2. voice_analyzer.py + audio_signal_fusion.py → Replace with torchaudio

**Current problem:**
- `voice_analyzer.py`: librosa-based, CPU-only, slow
- `audio_signal_fusion.py`: HuBERT SER works well but uses librosa for preprocessing

**New architecture — TWO LAYERS:**

**Layer A — Raw Audio Features → voice_analyzer.py:**
- Reference: `stajyer_kodlari/voice_test_said.py` (written by intern, may be incomplete)
- Use torchaudio (GPU-native, CUDA 12.1 compatible)
- **Remove tkinter GUI** — accept `wav_path: str` as function parameter
- If intern code is missing features, complete using torchaudio documentation
- Must be pipeline-integrated (not standalone)

**Layer B — Emotion Signal → audio_signal_fusion.py:**
- Keep `SeaBenSea/hubert-large-turkish-speech-emotion-recognition` — it works well
- Only replace librosa preprocessing with torchaudio equivalents
- Do NOT change HuBERT model weights or inference logic

**Required output format for voice_analyzer (per-second segments):**
```python
{
    "start_sec": int,
    "end_sec": float,
    "segment_type": str,         # "konuşma" | "sessiz" | "konuşma_dışı"
    "is_speech": bool,
    "rms_dbfs": float,
    "f0_mean": float,            # mean pitch
    "f0_std": float,             # pitch variation
    "spectral_centroid": float,
    "spectral_flatness": float,
    "mel_energy": float,
    "konusma_guveni": float,     # speech confidence 0.0 – 1.0
    "konusma_stili": str,        # "heyecanlı" | "sakin" | "gergin" | "monoton"
    "konusma_enerjisi": str      # "yüksek" | "orta" | "düşük"
}
```

---

### 3. contextual_aggregator.py → Update

**What it must do:**
- For each STT segment (start, end, text):
  - Collect face_analyzer frames in that time range → compute dominant emotion + average gaze
  - Collect voice_analyzer second-segments in that range → speech style + energy
  - Pull HuBERT valence/arousal from audio_signal_fusion
- Package everything into a single dict per segment for the LLM

**Required LLM segment package format:**
```python
{
    "segment_id": int,
    "start": float,
    "end": float,
    "text": str,                  # STT transcript
    "dominant_emotion": str,      # most frequent emotion_label in this segment
    "emotion_confidence": float,  # average confidence
    "avg_gaze_pitch": float,
    "avg_gaze_yaw": float,
    "gaze_direction": str,        # computed label: "center"|"up"|"down"|"right"|"left"
                                  # thresholds: |pitch|>15° → up/down, |yaw|>20° → right/left
    "speech_style": str,
    "speech_confidence": float,
    "f0_mean": float,
    "rms_dbfs": float,
    "hubert_valence": float,
    "hubert_arousal": float
}
```

---

### 4. pipeline.py → Update
- Rewire imports to new modules
- Preserve Phase3 flow exactly
- Error handling: if no face detected → face_detected=False, do NOT crash, continue

---

### 5. prompt_phase3.txt → Update
- Add new signal names: emotion_label, gaze_direction, speech_style, speech_confidence
- Add gaze interpretation guide for LLM:
  - pitch > 15° down = possible avoidance or low confidence
  - yaw > 20° = possible distraction or discomfort
  - sustained neutral + downward gaze = possible anxiety
- Add emotion confidence rule: if confidence < 0.5 → mark as "uncertain", do not over-interpret
- Keep existing prompt structure, only extend it

---

## Technical Constraints
| Parameter | Value |
|-----------|-------|
| Python | 3.10 |
| CUDA | 12.1 |
| torch | 2.5.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| uniface | 3.0.0 (already installed) |
| HuBERT model | SeaBenSea/hubert-large-turkish-speech-emotion-recognition (keep as-is) |
| Whisper | openai/whisper-large-v3-turbo (keep as-is) |
| LLM | Ollama + Gemma3:12b (keep as-is) |

**Hard rules:**
- NO tkinter anywhere (headless pipeline)
- NO mediapipe anywhere after migration
- librosa: replace with torchaudio wherever possible; only keep if strictly necessary
- Do not introduce new heavy dependencies without checking if torchaudio/torch already covers it

---

## Do NOT Touch
- `src/audio/text_analyzer.py`
- `src/audio/thought_unit_merger.py`
- `src/nlp/ollama_ai.py`
- `src/nlp/gemini.py`
- `src/reporting/` (entire folder)
- `api/main.py`
- `test_example.py`

---

## Success Criteria
1. `python test_example.py video.mp4 --phase3 --ollama` runs without errors
2. Every segment package contains: emotion_label + gaze_direction + speech_style
3. HTML report shows emotion and gaze data in "Kritik Anlar" cards
4. Zero mediapipe imports in any active file
5. HuBERT SER continues to work in pipeline

---

## Claude Code — Step-by-Step Work Order

**Step 1 — Read and understand (do not write any code yet):**
- Read all files under `src/`
- Read both files in `stajyer_kodlari/`
- Understand the full pipeline flow from video input to HTML report output

**Step 2 — Document before coding:**
- Write `ARCHITECTURE.md` showing: current architecture vs target architecture
- Include your decision on MobileGaze vs GaZeL with reasoning
- Stop and wait for approval before proceeding

**Step 3 — Implement in this order (one file at a time):**
1. `src/vision/face_analyzer.py` (UniFace)
2. `src/audio/voice_analyzer.py` (torchaudio)
3. `src/audio/audio_signal_fusion.py` (keep HuBERT, replace librosa)
4. `src/nlp/contextual_aggregator.py` (new signal formats)
5. `src/pipeline.py` (rewire)
6. `src/nlp/prompt_phase3.txt` (extend)
7. `requirements.txt` (cleanup)

**Step 4 — Test:**
- Run `python test_example.py video.mp4 --phase3 --ollama`
- Fix any import errors or runtime crashes
- Confirm all 5 success criteria are met