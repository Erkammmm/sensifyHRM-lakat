# SensifyHR — Architecture Document
## FAZ-3 → FAZ-4 Migration

---

## 1. Current Architecture (FAZ-3)

### Pipeline Flow

```
video.mp4
    │
    ├─[VideoProcessor]──────────────────────────────── video_info (fps, resolution, duration)
    │       │
    │       └── extract_audio() ──────────────────────► audio.wav (16kHz mono, ffmpeg)
    │
    ├─[TextAnalyzer]  ◄── audio.wav
    │       │   faster-whisper (Systran/faster-whisper-large-v3-turbo) + CUDA
    │       └── STT segments: [{start, end, text}, ...]
    │               │
    │       [ThoughtUnitMerger] → merged thought units (15–35s blocks)
    │
    ├─[VoiceAnalyzer]  ◄── audio.wav
    │       │   librosa: load → trim → pre-emphasis → normalize
    │       │   Features: RMS, pitch F0 (pyin), mel spectrogram, VAD
    │       └── raw_voice_features (windowed 8s, for plotting only)
    │
    ├─[AudioSignalFusion]  ◄── audio.wav
    │       │   Layer A: librosa → RMS, pitch_std, onset_strength → energy/rate/pitch states
    │       │   Layer B: HuBERT SER (SeaBenSea) → label probs → valence/arousal projection
    │       └── timeline: [{start, end, valence_state, arousal_state,
    │                        speech_energy, speech_rate, pitch_stability}, ...]
    │
    ├─[FaceAnalyzer]  ◄── video.mp4
    │       │   MediaPipe FaceLandmarker (.task file, blendshape scores)
    │       │   Rule-based inference from blendshape scores:
    │       │     - emotion: smile/brow/jaw thresholds → Mutlu/Ofkeli/Korku/Tiksinti/...
    │       │     - gaze: eyeLookIn/Out/Down → string label ("Ekrana Bakiyor" etc.)
    │       │     - phase3: facial_state (POSITIVE/TENSE/NEUTRAL) + attention_state + stress_indicator
    │       └── timeline: [{timestamp, facial_state, attention_state, stress_indicator, gaze}, ...]
    │
    └─[ContextualAggregator]  ◄── thought_units + audio_signal_timeline + face_timeline
            │   Time-range intersection → mode() per field
            └── segment_signal_packages: [{timestamp, start, end, text,
                                           audio_signal{valence,arousal,speech_energy,...},
                                           visual_signal{facial_state,attention_state,stress_indicator}}, ...]
                    │
                [LLM (Ollama/Gemma3:12b)]
                    │
                [ReportGenerator] → HTML report
```

### Key Problems in FAZ-3

| Module | Problem |
|--------|---------|
| `face_analyzer.py` | MediaPipe blendshapes → rule-based emotion is unreliable and noisy. Gaze is a string label, not metric degrees. No actual classifier confidence. Requires `.task` model file with Unicode path workaround. |
| `voice_analyzer.py` | librosa-only, CPU-bound, produces plotting data not pipeline-ready per-second segments. Output format doesn't match what contextual_aggregator needs. |
| `audio_signal_fusion.py` | HuBERT SER inference works well; physical feature extraction uses librosa (should be torchaudio). |
| `contextual_aggregator.py` | Consumes `facial_state/attention_state/stress_indicator` — these proxy fields must be replaced with `emotion_label/gaze_direction/speech_style`. |
| `prompt_phase3.txt` | Doesn't mention new signal names (emotion_label, gaze_direction, speech_style, speech_confidence, emotion_confidence). No gaze interpretation guide for LLM. |

---

## 2. Target Architecture (FAZ-4)

### Pipeline Flow

```
video.mp4
    │
    ├─[VideoProcessor]  (UNCHANGED) ─────────────────── video_info
    │       └── extract_audio() ──────────────────────► audio.wav
    │
    ├─[TextAnalyzer]  (UNCHANGED) ◄── audio.wav
    │       └── STT segments: [{start, end, text}, ...]
    │               │
    │       [ThoughtUnitMerger]  (UNCHANGED) → thought units
    │
    ├─[VoiceAnalyzer]  ◄── audio.wav          ← REPLACED (librosa → torchaudio)
    │       │   torchaudio: load → resample → per-second segmentation
    │       │   VAD: RMS + ZCR + spectral flatness → "konuşma"/"sessiz"/"konuşma_dışı"
    │       │   Per-second features: rms_dbfs, f0_mean, f0_std, spectral_centroid,
    │       │                        spectral_flatness, mel_energy
    │       │   Derived: konusma_stili, konusma_enerjisi, konusma_guveni
    │       └── per_second_timeline: [{start_sec, end_sec, segment_type, is_speech,
    │                                   rms_dbfs, f0_mean, f0_std, spectral_centroid,
    │                                   spectral_flatness, mel_energy,
    │                                   konusma_guveni, konusma_stili, konusma_enerjisi}, ...]
    │
    ├─[AudioSignalFusion]  ◄── audio.wav       ← UPDATED (librosa preprocessing → torchaudio)
    │       │   Layer A: torchaudio → physical states (speech_energy, speech_rate, pitch_stability)
    │       │   Layer B: HuBERT SER (UNCHANGED) → valence/arousal projection
    │       └── timeline: [{start, end, valence_state, arousal_state, ...}, ...]
    │
    ├─[FaceAnalyzer]  ◄── video.mp4            ← REPLACED (MediaPipe → UniFace)
    │       │   RetinaFace (MNET_025) → face detection + landmarks
    │       │   DDAMFN AffectNet7 → 7-class emotion classification with confidence
    │       │   MobileGaze (ResNet18) → pitch_deg, yaw_deg in degrees
    │       │   Every 10th frame; timestamp_sec = frame_index / fps
    │       │   face_detected=False → skip inference, include empty record, no crash
    │       └── timeline: [{timestamp_sec, emotion_label, emotion_confidence,
    │                        gaze_pitch_deg, gaze_yaw_deg, face_detected}, ...]
    │
    └─[ContextualAggregator]  ◄── thought_units + audio_signal_timeline + face_timeline
            │                                    ← UPDATED (new signal field names)
            │   Face frames in range → dominant emotion_label + avg emotion_confidence
            │                        → avg gaze_pitch_deg, avg gaze_yaw_deg → gaze_direction label
            │   Voice seconds in range → mode(konusma_stili), mean(konusma_guveni)
            │                          → mean(f0_mean), mean(rms_dbfs)
            │   HuBERT timeline in range → hubert_valence, hubert_arousal
            └── segment_signal_packages: [{segment_id, start, end, text,
                                           dominant_emotion, emotion_confidence,
                                           avg_gaze_pitch, avg_gaze_yaw, gaze_direction,
                                           speech_style, speech_confidence,
                                           f0_mean, rms_dbfs,
                                           hubert_valence, hubert_arousal}, ...]
                    │
                [LLM (Ollama/Gemma3:12b)]  (UNCHANGED)
                    │
                [ReportGenerator]  (UNCHANGED) → HTML report
```

### Module-by-Module Change Summary

| Module | Status | Change |
|--------|--------|--------|
| `src/vision/video_processor.py` | **DO NOT TOUCH** | — |
| `src/vision/face_analyzer.py` | **REPLACE** | MediaPipe → UniFace (RetinaFace + DDAMFN + MobileGaze) |
| `src/audio/text_analyzer.py` | **DO NOT TOUCH** | — |
| `src/audio/thought_unit_merger.py` | **DO NOT TOUCH** | — |
| `src/audio/voice_analyzer.py` | **REPLACE** | librosa → torchaudio, remove tkinter GUI |
| `src/audio/audio_signal_fusion.py` | **UPDATE** | Keep HuBERT SER, replace librosa preprocessing with torchaudio |
| `src/audio/audio_analyzer.py` | **DO NOT TOUCH** | Legacy v2; not used in phase3 path |
| `src/nlp/contextual_aggregator.py` | **UPDATE** | Consume new signal formats, compute gaze_direction label |
| `src/nlp/ollama_ai.py` | **DO NOT TOUCH** | — |
| `src/nlp/gemini.py` | **DO NOT TOUCH** | — |
| `src/nlp/prompt_phase3.txt` | **UPDATE** | Add new signal names + gaze interpretation guide |
| `src/nlp/prompt.txt` | **DO NOT TOUCH** | — |
| `src/reporting/` | **DO NOT TOUCH** | Entire folder |
| `src/pipeline.py` | **UPDATE** | Rewire imports, update step labels |
| `api/main.py` | **DO NOT TOUCH** | — |
| `test_example.py` | **DO NOT TOUCH** | — |
| `requirements.txt` | **UPDATE** | Remove mediapipe, add uniface (already installed) |

---

## 3. Gaze Estimator Decision: MobileGaze vs GaZeL

### Decision: **MobileGaze (via uniface)**

### Evaluation

| Criterion | MobileGaze | GaZeL |
|-----------|-----------|-------|
| Dependency | Part of `uniface 3.0.0` — **already installed** | Separate package, not installed |
| API surface | `gaze_estimator.estimate(face_crop)` → `.pitch`, `.yaw` in radians | Different API, requires own model download |
| Model size | ResNet18 backbone — lightweight | Larger transformer-based architecture |
| Integration | Proven working in intern code (`stajyer_kodlari/gaze-emotion-json_sefa.py`) | No existing integration code |
| Output format | `(pitch_rad, yaw_rad)` → convert to degrees | Would need API investigation |
| New dependency risk | **Zero** — already in uniface | Introduces new package + potential CUDA conflicts |
| Interview suitability | Adequate for frontal/near-frontal faces; ResNet18 trained on gaze datasets | Better stability claimed but untested in this env |

### Reasoning

The CLAUDE.md hard rule states: *"Do not introduce new heavy dependencies without checking if torchaudio/torch already covers it."* GaZeL would be a new heavy dependency with no proven benefit over MobileGaze in this specific pipeline context.

MobileGaze is already working in the intern code. The intern code (`gaze-emotion-json_sefa.py`) uses `GazeWeights.RESNET18` via `uniface.gaze.MobileGaze` and successfully produces `pitch_deg`/`yaw_deg` values. For a seated interview scenario (frontal face, controlled environment, camera at desk level), MobileGaze's accuracy is sufficient. The pipeline requires relative gaze classification (`center`/`up`/`down`/`left`/`right`) with ±15°/±20° thresholds — not sub-degree precision.

**Conclusion:** Use MobileGaze (uniface built-in). If future validation shows GaZeL produces substantially better stability on interview footage, migration is straightforward since the output format (pitch_deg, yaw_deg) is identical.

---

## 4. Data Flow: Signal Alignment

```
Video timeline (seconds):
0────────────────────────────────────────────────────► t

FaceAnalyzer output (every 10th frame, ~3 fps):
  ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●   ●
  {timestamp_sec, emotion_label, emotion_confidence,
   gaze_pitch_deg, gaze_yaw_deg, face_detected}

VoiceAnalyzer output (per second):
  [──1s──][──1s──][──1s──][──1s──][──1s──][──1s──]
  {start_sec, end_sec, rms_dbfs, f0_mean, konusma_stili, ...}

AudioSignalFusion output (3s chunks, 1s step):
  [────3s────]
       [────3s────]
            [────3s────]
  {start, end, valence_state, arousal_state, ...}

ThoughtUnit (merged STT, 15–35s blocks):
  [══════════════════20s════════════════════]
  {start, end, text}

ContextualAggregator: for each thought_unit window:
  → face frames in [start, end] → dominant_emotion, avg_gaze_*
  → voice seconds overlapping → mode(speech_style), mean(f0_mean)
  → audio signal chunks overlapping → hubert_valence, hubert_arousal
  → gaze_direction = classify(avg_gaze_pitch, avg_gaze_yaw)
       |pitch| > 15° → "up" or "down"
       |yaw|   > 20° → "right" or "left"
       else         → "center"
```

---

## 5. New Output Formats (FAZ-4)

### FaceAnalyzer — per-frame record
```python
{
    "timestamp_sec": float,        # frame_index / fps
    "emotion_label": str,          # "Happy"|"Sad"|"Angry"|"Fear"|"Disgust"|"Surprise"|"Neutral"
    "emotion_confidence": float,   # 0.0 – 1.0 (DDAMFN softmax output)
    "gaze_pitch_deg": float,       # positive = looking up
    "gaze_yaw_deg": float,         # positive = looking right
    "face_detected": bool
}
```

### VoiceAnalyzer — per-second record
```python
{
    "start_sec": int,
    "end_sec": float,
    "segment_type": str,           # "konuşma"|"sessiz"|"konuşma_dışı"
    "is_speech": bool,
    "rms_dbfs": float,
    "f0_mean": float,
    "f0_std": float,
    "spectral_centroid": float,
    "spectral_flatness": float,
    "mel_energy": float,
    "konusma_guveni": float,       # 0.0 – 1.0
    "konusma_stili": str,          # "heyecanlı"|"sakin"|"gergin"|"monoton"
    "konusma_enerjisi": str        # "yüksek"|"orta"|"düşük"
}
```

### ContextualAggregator — LLM segment package
```python
{
    "segment_id": int,
    "start": float,
    "end": float,
    "text": str,
    "dominant_emotion": str,       # most frequent emotion_label in window
    "emotion_confidence": float,   # average confidence (uncertain if < 0.5)
    "avg_gaze_pitch": float,
    "avg_gaze_yaw": float,
    "gaze_direction": str,         # "center"|"up"|"down"|"right"|"left"
    "speech_style": str,
    "speech_confidence": float,
    "f0_mean": float,
    "rms_dbfs": float,
    "hubert_valence": float,
    "hubert_arousal": float
}
```

---

## 6. Intern Code: What Is Complete vs What Needs Work

### `stajyer_kodlari/gaze-emotion-json_sefa.py`

**Complete:**
- RetinaFace detection, DDAMFN emotion prediction, MobileGaze estimation
- Every-10th-frame skip logic
- JSON output with bbox, confidence, emotion label+confidence, gaze pitch/yaw degrees

**Missing (must add in new face_analyzer.py):**
- `timestamp_sec` field: must compute as `frame_index / video_fps`
- `face_detected = False` path (no crash when no face)
- Class-based interface (`FaceAnalyzer.process_video()`) instead of CLI script
- Returning `(timeline, summary)` tuple matching existing pipeline contract
- No tkinter / no GUI (headless)

### `stajyer_kodlari/voice_test_said.py`

**Complete:**
- torchaudio-native audio loading and resampling
- Per-second segmentation (1s chunks)
- RMS dBFS, F0 via pyin (torchaudio functional), spectral centroid/flatness, mel energy
- VAD logic: active ratio → speech-like ratio → real speech ratio → konusma_guveni
- `konusma_stili` and `konusma_enerjisi` classification
- tkinter GUI for file selection (must be removed)

**Missing (must adapt in new voice_analyzer.py):**
- Remove tkinter — accept `wav_path: str` as function parameter
- Class-based interface (`VoiceAnalyzer`) with `analyze_audio(wav_path)` method
- Return value must be the per-second list directly (not write to file)
- Config constants should remain as class/module-level defaults
