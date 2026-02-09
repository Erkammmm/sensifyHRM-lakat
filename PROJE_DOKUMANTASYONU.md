# SensifyHR v3.0 (FAZ‑3) - Teknik Dokümantasyon

## 1. Proje Özeti

SensifyHR, online iş görüşmelerinde adayların multimodal (çok modlu) analizini yapan bir yapay zekâ sistemidir.

v3 ana paradigma:
- **Metin = sadece içerik (STT)**
- **Ses = valence/arousal + fiziksel sinyal state’leri**
- **Görsel = facial_state / attention_state / stress_indicator**
- **LLM = tek karar verici (signal reasoning)**
 - Sistem **emotion/sentiment sınıflandırmaz**.

---

## 2. Proje Yapısı

```
sensifyHRMülakay/
├── src/                        # Kaynak kodlar
│   ├── __init__.py
│   ├── text_analyzer.py        # STT (turbo) -> segment text
│   ├── audio_signal_fusion.py  # v3: HuBERT SER projection + librosa -> audio_signal
│   ├── contextual_aggregator.py# v3: segment signal package builder (LLM input)
│   ├── face_analyzer.py        # visual signal + gaze/blink
│   ├── voice_analyzer.py       # Librosa ham ses özellikleri
│   ├── video_processor.py      # Video işleme yardımcıları
│   ├── pipeline.py             # Ana pipeline (orchestrator)
│   ├── ollama_ai.py            # Ollama + Gemma istemcisi
│   ├── gemini.py               # Opsiyonel: Gemini istemcisi
│   ├── report_generator.py     # HTML + JSON rapor
│   ├── plot.py                 # Matplotlib grafik üretici
│   └── prompt_phase3.txt       # v3 prompt (signal reasoning, TR/HR)
├── api/
│   └── main.py                 # FastAPI endpoint'leri
├── weights/
│   └── face_landmarker.task    # MediaPipe model
├── stajyer_çalışma/            # Referans dosyalar
├── Dockerfile
├── requirements.txt
├── requirements.lock.sonn.txt
└── test_example.py
```

---

## 3. Kullanılan Teknolojiler ve Modeller

### 3.1 Metin Analizi (text_analyzer.py)

| Bileşen | Detay |
|---------|-------|
| STT (Speech-to-Text) | `openai/whisper-large-v3-turbo` (Transformers) + fallback |
| Duygu Analizi | **YOK** (v3) |
| Çıktı | `{start,end,text}` |
| GPU | CUDA varsa kullanır; OOM durumunda otomatik CPU fallback |

**Akış:**
1. Video → WAV çıkarma
2. WAV → STT segmentleri (`{start,end,text}`)

### 3.2 FAZ‑3 Audio Signal (audio_signal_fusion.py)

FAZ‑3’te ses tarafı **emotion etiketi üretmez**. Bunun yerine:
- **SER (zayıf sinyal)**: `SeaBenSea/hubert-large-turkish-speech-emotion-recognition` ([model card](https://huggingface.co/SeaBenSea/hubert-large-turkish-speech-emotion-recognition))
- SER çıktısı `EMOTION_TO_SIGNAL` ile **valence/arousal** skorlarına projekte edilir ve state’e bucketize edilir:
  - `valence_state`: `NEGATIVE | NEUTRAL | POSITIVE`
  - `arousal_state`: `LOW | MEDIUM | HIGH`
- Librosa’dan fiziksel state’ler üretilir:
  - `speech_energy`: `LOW | MEDIUM | HIGH`
  - `speech_rate`: `SLOW | NORMAL | FAST`
  - `pitch_stability`: `STABLE | UNSTABLE`
- Zayıf sinyal dalgalanmalarını azaltmak için **EMA smoothing** uygulanır.

### 3.3 Yüz Analizi (face_analyzer.py)

| Bileşen | Detay |
|---------|-------|
| Model | MediaPipe FaceLandmarker (float16) |
| Duygu | **v2 legacy:** kural tabanlı emotion label |
| Bakış | eyeLookIn/Out/Down skorlarından yön tespiti |
| Göz Kırpma | eyeBlinkLeft/Right > 0.5 eşiği |
| Çıktı | Timeline + özet istatistikler |

**Duygu Kuralları:**
- Korku: browInnerUp > 0.2 && browOuterUp > 0.1 && eyeWide > 0.1
- Tiksinti: noseSneer > 0.15
- Mutlu: mouthSmile > 0.35
- Şaşkın: browInnerUp > 0.4 && jawOpen > 0.10
- Öfkeli: browDown > 0.35
- Stresli: mouthRoll+mouthShrug / 3 > 0.25
- Nötr: hiçbiri tetiklenmezse

### 3.4 Ses Özellikleri (voice_analyzer.py)

| Bileşen | Detay |
|---------|-------|
| Kütüphane | Librosa |
| Ön İşleme | Trim (sessizlik), Pre-emphasis (0.97), Normalize (0.95) |
| RMS Enerji | frame_length=2048, hop_length=512 |
| Pitch (F0) | librosa.pyin, fmin=50, fmax=400 Hz |
| VAD | librosa.effects.split, top_db=20 |
| Mel Spectrogram | 128 mel band, power=2.0 |

### 3.5 FAZ‑3 Visual Signal (face_analyzer.py)

FAZ‑3’te görsel çıktı davranışsal state’lerdir:
- `facial_state`: `NEUTRAL | POSITIVE | TENSE`
- `attention_state`: `FOCUSED | AVERTED`
- `stress_indicator`: `LOW | ELEVATED | HIGH`

### 3.8 Contextual Aggregator (contextual_aggregator.py)

Whisper segmentleri için zaman bazlı hizalama yapar ve LLM’ye giden **Segment Signal Package**’ı üretir:

```json
{
  "timestamp": "00:45 - 00:52",
  "start": 45.0,
  "end": 52.0,
  "text": "…",
  "audio_signal": {
    "valence": "NEUTRAL",
    "arousal": "MEDIUM",
    "speech_energy": "LOW",
    "speech_rate": "NORMAL",
    "pitch_stability": "STABLE"
  },
  "visual_signal": {
    "facial_state": "NEUTRAL",
    "attention_state": "FOCUSED",
    "stress_indicator": "LOW"
  }
}
```

### 3.6 AI Değerlendirme

**Ollama + Gemma (ollama_ai.py) – Önerilen:**
- Model: **gemma3:12b** (yerel)
- Segment paketleri **10’arlı chunk**’lar halinde analiz edilip final sentez üretilir

**Gemini (gemini.py) – Opsiyonel:**
- Prompt: `src/prompt_phase3.txt`

**Ollama + Gemma (ollama_ai.py):**
- Model: **gemma3:12b** (yerel)
- Tamamen offline çalışır
- Stajyer çalışmasındaki interview_ai.py baz alınmıştır
- v3’te segment paketleri **10’arlı chunk**’lar halinde analiz edilip final sentez üretilir

---

## 4. Pipeline Yapısı

```
Video (MP4)
    │
    ├─→ [1] VideoProcessor.extract_audio() → WAV
    │       │
    │       ├─→ [2] TextAnalyzer.process_video()
    │       │       → STT segmentleri (text only)
    │       │
    │       ├─→ [3] Audio Signal Fusion
    │       │       → valence/arousal + fiziksel state’ler
    │       │
    │       └─→ [4] VoiceAnalyzer.analyze_audio()
    │               → Librosa → RMS, Pitch, VAD, Mel
    │
    ├─→ [5] FaceAnalyzer.process_video()
    │       → visual signal (facial/attention/stress) + gaze/blink
    │
    └─→ [6] Contextual Aggregator
            → Segment Signal Packages (LLM input)
            → LLM reasoning (chunking)
            → Rapor (JSON + HTML product UI + AI)
```

---

## 5. Çıktı Formatı

### JSON Rapor Yapısı

```json
{
    "interview_id": "uuid",
    "phase": "v3",
    "duration_seconds": 45.2,
    "video_info": {"fps": 30, "width": 1920, "height": 1080, "duration_seconds": 120},
    "text_analysis": {
        "segments": [{"start": 0.0, "end": 3.5, "text": "..."}],
        "summary": {"total_sentences": 15}
    },
    "audio_signal_analysis": {
        "timeline": [{"start": 0.0, "end": 3.0, "valence_state": "NEUTRAL", "arousal_state": "MEDIUM"}],
        "summary": {"dominant_valence": "NEUTRAL", "dominant_arousal": "MEDIUM"}
    },
    "visual_signal_analysis": {
        "timeline": [{"timestamp": 0.5, "facial_state": "NEUTRAL", "attention_state": "FOCUSED"}],
        "summary": {"focus_score": 85.2, "blink_rate_per_min": 15.3, "dominant_emotion": "NEUTRAL"}
    },
    "voice_analysis": {
        "raw_voice_features": {
            "energy_rms": {"mean": 0.045, "variance": 0.001},
            "pitch_f0": {"mean": 180.5, "variability": 45.2, "jump_count": 3},
            "speech_silence": {"total_speech_seconds": 90.5, "total_silence_seconds": 29.5}
        }
    },
    "ai_analysis": {"analysis": "..."}
}
```

---

## 6. Grafikler (16 adet)

v3’te grafikler “Detaylar” sekmesinde tutulur:
- Visual signal dağılımları (facial/attention/stress)
- Audio signal (valence/arousal) dağılım + timeline
- Teknik ses grafikleri (RMS/Pitch/VAD/Mel)

---

## 7. API Endpoint'leri

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | API bilgileri |
| GET | `/health` | Sağlık kontrolü |
| POST | `/analyze` | Video yükle + analiz et |
| GET | `/status/{id}` | Analiz durumu |

### POST /analyze

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@video.mp4" \
  -F "interview_id=test-001"
```

FAZ‑3 opt‑in ve LLM seçimi:
- `POST /analyze?phase3=true&llm_provider=ollama`
- `llm_provider`: `gemini|ollama|none` (default gemini; hata/kota olursa otomatik ollama fallback)

---

## 8. Bağımlılıklar

### Sistem
- Python 3.10+
- FFmpeg
- CUDA 12.1 (GPU için)
- Ollama (opsiyonel, yerel AI için)

### Python Paketleri
Tam liste: `requirements.txt` veya `requirements.lock.sonn.txt`

Kritik paketler:
- `torch>=2.1.0` (CUDA 12.1)
- `openai-whisper`
- `transformers>=4.39.0`
- `mediapipe==0.10.9`
- `librosa>=0.10.0`
- `fastapi>=0.104.0`
- `matplotlib>=3.8.0`
- `google-generativeai>=0.8.0`
- `ollama>=0.3.0`

---

## 9. Notlar

- MediaPipe sadece CPU'da çalışır (Python API sınırlaması)
- STT ve HuBERT SER GPU’da çalışır (CUDA otomatik tespit); OOM durumunda otomatik CPU fallback yapılır
- `protobuf==3.20.3` MediaPipe uyumluluğu için sabit tutulmalıdır
- Gemini ayrı subprocess'te çalışır (protobuf çakışması önlemi)
- `requirements.lock.sonn.txt` dosyası Anaconda ortamından freeze edilmiştir
- Ollama + Gemma tamamen yerel/offline çalışır, internet gerektirmez
