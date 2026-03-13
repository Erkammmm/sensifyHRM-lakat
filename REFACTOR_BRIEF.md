# SensifyHR FAZ-4 — Major Refactor Brief

## What Just Happened
We completed the FAZ-3 → FAZ-4 migration (MediaPipe → UniFace, librosa → torchaudio).
The pipeline runs end-to-end without errors. However the output quality and performance
are unacceptable. This document defines the next refactor.

---

## Core Problem 1: Speed (Critical)

A 12-second video takes 15 minutes. Target: 30-minute video in under 5 minutes.

**Root causes to investigate and fix:**
- FaceAnalyzer processes every 10th frame but UniFace ONNX runs on CPU (onnxruntime).
  Check if onnxruntime-gpu is available and switch to it. If not, increase frame skip to
  every 15th or 20th frame for interview footage (frontal face, slow movement).
- HuBERT SER runs on full audio in 3s chunks with 1s step → too many overlapping chunks.
  Switch to non-overlapping 3s chunks (3s step instead of 1s step).
- VoiceAnalyzer and FaceAnalyzer run sequentially. They can run in parallel (separate threads
  or ProcessPoolExecutor) since they use different resources (CPU/GPU split).
- Profile the pipeline first: measure time spent in each module, report it, then optimize
  the slowest ones first. Do not guess — measure.

---

## Core Problem 2: Output Quality (Critical)

The current output is noisy raw signals fed to LLM. The LLM produces poor analysis because
the input is low-quality. The entire output/report philosophy needs to change.

### What We NO Longer Need
- CV matching (done in a separate project)
- Interview question generation
- Candidate scoring against job descriptions
- Any mention of "soru" (question) in prompts or reports

### What We Actually Want: Behavioral Analysis for HR

The system must answer ONE question for an HR professional:
**"How did this candidate behave during the interview — and what does it mean?"**

This means:

**1. Emotional Arc** — How did the candidate's emotional state change over time?
- Not frame-by-frame noise. Smoothed timeline: every 30-60 seconds, what was the dominant emotion?
- Key moments: when did emotion shift significantly? (e.g., "At 4:30, candidate switched from
  Neutral to Anxious and stayed there for 2 minutes")

**2. Gaze Behavior** — Where was the candidate looking and what does it suggest?
- % time looking at camera (center gaze) vs away
- Sustained gaze aversion patterns (not single frames — sustained >3s)
- Correlation with speech: does gaze drop when answering certain topics?

**3. Speech Patterns** — How did the candidate speak?
- Speech rate consistency (monotone vs varied energy)
- Long silence periods (>5s) — when and how often
- Speech confidence score over time
- Voice stress indicators (f0 variation, energy drops)

**4. Behavioral Consistency** — Do signals align or contradict?
- Does the candidate say positive things while showing negative emotion? (incongruence)
- Does gaze drop precisely when speech confidence drops? (compound signal)
- These compound signals are HIGH VALUE for HR

---

## Core Problem 3: Prompt Rewrite

`src/nlp/prompt_phase3.txt` must be completely rewritten.

**New prompt goal:** Given time-aligned behavioral signals, produce an HR behavioral report.

**New prompt structure:**
1. Role: "You are a behavioral analyst providing insights to an HR professional."
2. Input description: the segment packages (emotion, gaze, speech per segment)
3. Task: Produce a structured behavioral report with these sections:
   - **Genel Davranışsal Profil** (2-3 sentences: overall behavioral impression)
   - **Duygusal Seyir** (emotional arc: how emotions evolved, key shifts with timestamps)
   - **Göz Teması ve Dikkat** (gaze: % camera contact, notable aversion patterns)
   - **Konuşma Dinamikleri** (speech: energy, confidence, silence patterns)
   - **Tutarsızlık Sinyalleri** (incongruence: where emotion/gaze/speech didn't align)
   - **HR İçin Öneriler** (2-3 actionable observations for the HR professional)
4. Rules:
   - Write in Turkish, professional tone
   - Do NOT score or rank the candidate
   - Do NOT say "elenmeli" or make hiring decisions
   - Cite specific timestamps for all observations
   - If emotion_confidence < 0.5, do not interpret that signal
   - Minimum 3 segments needed for pattern claims

---

## Core Problem 4: Report/Dashboard Simplification

The current HTML report has too many tabs and technical details that HR cannot use.

**New dashboard — single page, 3 sections:**

**Section 1 — Behavioral Summary (top)**
- 4 KPI cards: Dominant Emotion | Avg Gaze (% camera) | Speech Confidence | Incongruence Count
- These are single numbers, easy to scan

**Section 2 — Timeline Chart (middle)**
- A simple timeline visualization showing emotion + gaze_direction + speech_confidence over time
- X-axis: time (seconds), Y-axis: the three signals
- Use Chart.js or matplotlib — whichever is already in the project
- HR should be able to see "at minute 3, everything dropped"

**Section 3 — LLM Behavioral Report (bottom)**
- The full LLM text output, formatted cleanly
- No raw JSON, no technical terms, no blendshape scores

**Remove from report:**
- "Detaylar" tab with raw signal graphs
- Technical metrics (valence_state, arousal_state raw values)
- Any CV/job matching sections
- Whisper transcript display (keep in JSON only)

---

## Core Problem 5: Code Cleanup

After speed and quality fixes, clean up dead code:

- `src/audio/audio_analyzer.py` — legacy wav2vec2 v2 module, only used in non-phase3 path.
  Check if non-phase3 path is used anywhere in production. If not, remove it.
- `src/nlp/prompt.txt` — legacy v2 prompt. Remove if unused.
- Any remaining mediapipe imports anywhere (double-check all files)
- Any remaining librosa imports (replace or remove)
- `stajyer_kodlari/` — these were reference only, do not delete but do not import from them

---

## Work Order for This Session

**Step 1 — Profile first (measure, don't guess):**
Run the pipeline with timing instrumentation on each module.
Report: "Module X took Y seconds" for each step.
Do not optimize yet — just measure and report.

**Step 2 — Speed optimization:**
Based on profiling results, fix the top 2-3 slowest bottlenecks.
Target: 30-minute video < 5 minutes total.
Re-test after each optimization.

**Step 3 — Prompt rewrite:**
Rewrite src/nlp/prompt_phase3.txt per the behavioral analysis structure above.
Keep Turkish language. Remove all CV/question generation references.

**Step 4 — Dashboard simplification:**
Update src/reporting/report_generator.py to produce the new 3-section dashboard.
Keep JSON output intact (only change HTML).

**Step 5 — Dead code cleanup:**
Remove confirmed-unused legacy code.

**Step 6 — End-to-end test:**
Run full pipeline on Test_erkam.mp4 with PYTHONIOENCODING=utf-8.
Verify: speed target met, report is HR-readable, no crashes.

---

## Hard Rules (Unchanged)
- Python path: C:/Users/ecetkin/AppData/Local/anaconda3/envs/gpu_env_videoai/python.exe
- Always set PYTHONIOENCODING=utf-8 before running
- DO NOT touch: text_analyzer.py, thought_unit_merger.py, ollama_ai.py, gemini.py, api/main.py
- HuBERT SER model stays (SeaBenSea/hubert-large-turkish-speech-emotion-recognition)
- Whisper STT stays (openai/whisper-large-v3-turbo)
- LLM stays (Ollama + Gemma3:12b)
- One file at a time, stop and wait for approval after each step