# SensifyHR FAZ-4 — Sistem Mimarisi

---

## Genel Bakış

SensifyHR, bir video mülakat kaydını analiz ederek davranışsal bir İK raporu üreten multimodal bir AI pipeline'ıdır.
Dört bağımsız sinyal kaynağını zaman bazlı hizalar ve LLM ile yorumlar.

**Temel ilke:** Sistem karar vermez, gözlemler ve yorumlar. "Elenmeli" gibi yargılar üretmez.

---

## Pipeline Akışı

```
video.mp4
    │
    ├─[VideoProcessor]────────────────────────── video_info (fps, resolution, duration_seconds)
    │       └── extract_audio() ──────────────► audio.wav (16kHz mono, ffmpeg)
    │                                              ~2-5s
    │
    ├─[TextAnalyzer]  ◄── audio.wav
    │       │   faster-whisper (Systran/faster-whisper-large-v3-turbo) + CUDA
    │       │   ~30-120s (video uzunluğuna göre)
    │       └── STT segments: [{start, end, text}, ...]
    │               │
    │       [ThoughtUnitMerger] → thought_units (15-35s bloklar)
    │
    ├─[VoiceAnalyzer]  ◄── audio.wav                          ─┐
    │       │   torchaudio: per-second segmentation            │ Paralel
    │       │   VAD: RMS + ZCR + spectral flatness             │ çalışır
    │       │   ~5-15s                                         │
    │       └── per_second_timeline: [{start_sec, end_sec,     │
    │               rms_dbfs, f0_mean, f0_std,                 │
    │               konusma_guveni, konusma_stili}, ...]        │
    │                                                           │
    ├─[AudioSignalFusion]  ◄── audio.wav                       │
    │       │   torchaudio: 3s chunk'lar (non-overlapping)     │
    │       │   f0_std + rms_dbfs → ses profili kuralları      │
    │       │   EMA yumuşatma (alpha=0.65)                     │
    │       │   ~3-10s                                         │
    │       └── audio_signal_timeline: [{start, end,            │
    │               valence_state, arousal_state,               │
    │               debug: {voice_profile, voice_energy,        │
    │                       valence_score_ema}}, ...]           │
    │                                                           │
    ├─[FaceAnalyzer]  ◄── video.mp4                            │
    │       │   RetinaFace: yüz tespiti (her 10. frame)        │
    │       │   DDAMFN AffectNet7: 7-sınıf duygu + confidence  │
    │       │   MobileGaze ResNet18: pitch_deg, yaw_deg         │
    │       │   ~60-180s (video uzunluğuna göre)               ─┘
    │       └── face_timeline: [{timestamp_sec, emotion_label,
    │               emotion_confidence, gaze_pitch_deg,
    │               gaze_yaw_deg, face_detected}, ...]
    │
    ├─[ContextualAggregator]
    │       │   Sinyal hizalama: zaman aralığı kesişimi
    │       │   Gaze offset: video geneli median bias düzeltmesi
    │       │   build_segment_signal_packages(): STT segment başına sinyal paketi
    │       │   build_smart_blocks(): doğal sessizlik sınırlı paragraf blokları
    │       └── segment_signal_packages + time_blocks
    │
    ├─[LLM — Gemini / Ollama]
    │       │   Gemini 2.5 Pro → 2.5 Flash → 2.0 Flash → Ollama → graceful skip
    │       │   Prompt: 5 sinyal kaynağı + 7 bölüm çıktı
    │       └── ai_analysis (Türkçe davranışsal rapor metni)
    │
    └─[ReportGenerator]
            │   matplotlib grafikleri (5 timeline chart)
            │   Jinja2 HTML template (report_v3.html)
            └── reports/{id}.json + reports/{id}.html
```

---

## Modüller ve Sorumluluklar

| Modül | Dosya | Sorumluluk |
|-------|-------|------------|
| VideoProcessor | `src/vision/video_processor.py` | Video metadata + audio extraction (ffmpeg) |
| FaceAnalyzer | `src/vision/face_analyzer.py` | UniFace: yüz tespiti, duygu, gaze (her 10. frame) |
| TextAnalyzer | `src/audio/text_analyzer.py` | faster-whisper STT, Türkçe segment üretimi |
| ThoughtUnitMerger | `src/audio/thought_unit_merger.py` | Kısa STT segmentlerini anlamlı blokta birleştirir |
| VoiceAnalyzer | `src/audio/voice_analyzer.py` | torchaudio per-second ses özellikleri + VAD |
| AudioSignalFusion | `src/audio/audio_signal_fusion.py` | f0+enerji kural sistemi → valence/arousal |
| ContextualAggregator | `src/nlp/contextual_aggregator.py` | Sinyal hizalama + paragraf bölme |
| OllamaAI | `src/nlp/ollama_ai.py` | Ollama yerel LLM entegrasyonu |
| Gemini | `src/nlp/gemini.py` | Google Gemini API entegrasyonu + fallback zinciri |
| ReportGenerator | `src/reporting/report_generator.py` | HTML + JSON rapor üretimi |
| Plot | `src/reporting/plot.py` | 5 matplotlib timeline grafiği |
| Pipeline | `src/pipeline.py` | Ana orchestrator |
| API | `api/main.py` | FastAPI REST endpoint'leri |

---

## Kullanılan Modeller

| Model | Sürüm/ID | Amaç | Neden Seçildi |
|-------|----------|------|---------------|
| RetinaFace | uniface 3.0.0 (MNET_025) | Yüz tespiti | Hızlı, güvenilir, uniface'e entegre |
| DDAMFN | uniface 3.0.0 (AffectNet7) | 7-sınıf yüz duygusu | Akademik benchmark SOTA, confidence skorlu |
| MobileGaze | uniface 3.0.0 (ResNet18) | Göz bakış tahmini (pitch/yaw derece) | Mevcut ortamda kurulu, inference'ı kanıtlanmış |
| faster-whisper | Systran/faster-whisper-large-v3-turbo | Türkçe STT | CTranslate2 optimizasyonu, CUDA hızlı |
| Gemini 2.5 Pro/Flash | google-generativeai | Birincil LLM | Büyük context, Türkçe performansı güçlü |
| Gemma3:12b | Ollama (yerel) | Yedek LLM | Offline çalışır, veri gizliliği |

> **Not:** SER modeli (wav2vec2 tabanlı ehcalabres) kaldırıldı — Türkçe mülakat ses tonu için
> prosody-based modeller sakin konuşmayı "angry/sad" etiketliyordu. Yerine dil bağımsız
> torchaudio f0+enerji kural sistemi kullanılmaktadır.

---

## Sinyal Kaynakları ve Formatları

### 1. Yüz Duygusu (FaceAnalyzer)
```python
# Per-frame, her 10. video frame (~3 fps)
{
    "timestamp_sec": float,         # frame_index / video_fps
    "emotion_label": str,           # Happy|Sad|Angry|Fear|Disgust|Surprise|Neutral
    "emotion_confidence": float,    # 0.0 – 1.0 (DDAMFN softmax)
    "gaze_pitch_deg": float,        # pozitif = yukarı bakış
    "gaze_yaw_deg": float,          # pozitif = sağa bakış
    "face_detected": bool
}
```

### 2. Ses Özellikleri (VoiceAnalyzer)
```python
# Per-second segmentler
{
    "start_sec": int,
    "end_sec": float,
    "segment_type": str,            # "konuşma" | "sessiz" | "konuşma_dışı"
    "is_speech": bool,
    "rms_dbfs": float,              # ses seviyesi (örn. -18 to -31 dBFS)
    "f0_mean": float,               # temel frekans Hz (örn. 90-160 Hz)
    "f0_std": float,                # pitch varyasyonu Hz (örn. 3-64 Hz)
    "spectral_centroid": float,
    "spectral_flatness": float,
    "mel_energy": float,
    "konusma_guveni": float,        # 0.0 – 1.0
    "konusma_stili": str,           # "canlı" | "dengeli" | "sakin" | "monoton"
    "konusma_enerjisi": str         # "yüksek" | "orta" | "düşük"
}
```

### 3. Ses Profili / Valence-Arousal (AudioSignalFusion)
```python
# 3 saniyelik chunk'lar
{
    "start": float, "end": float,
    "valence_state": str,           # "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    "arousal_state": str,           # "HIGH" | "MEDIUM" | "LOW"
    "debug": {
        "voice_profile": str,       # "Canlı"|"Kararlı"|"Dengeli"|"Sakin"|"Gergin"
        "voice_energy": str,        # "high"|"medium"|"low"
        "voice_variation": str,     # "high"|"medium"|"low"
        "rms_dbfs": float,
        "f0_std": float,
        "valence_score_ema": float, # -1 ile +1 (EMA yumuşatılmış)
        "arousal_score_ema": float
    }
}
```

**Ses profili kuralları:**
```
rms_dbfs > -25 → high; > -40 → medium; ≤ -40 → low
f0_std > 50 Hz → high; > 20 Hz → medium; ≤ 20 Hz → low

high enerji + high varyasyon → Canlı    (valence=+0.6, arousal=+0.7)
high enerji + orta/düşük    → Kararlı  (valence=+0.3, arousal=+0.5)
orta enerji + high varyasyon → Gergin   (valence=-0.4, arousal=+0.6)
orta enerji + orta varyasyon → Dengeli  (valence=+0.1, arousal=+0.2)
düşük/orta enerji + düşük   → Sakin    (valence=-0.1, arousal=-0.2)
```

### 4. LLM Segment Paketi (ContextualAggregator)
```python
# STT segment başına, thought_unit penceresinde tüm sinyaller hizalanır
{
    "segment_id": int,
    "timestamp": str,               # "MM:SS - MM:SS"
    "start": float, "end": float,
    "text": str,
    "dominant_emotion": str,        # confidence filtreli en sık yüz duygusu
    "emotion_confidence": float,
    "avg_gaze_pitch": float,
    "avg_gaze_yaw": float,
    "gaze_direction": str,          # center|up|down|right|left
    "speech_style": str,
    "speech_confidence": float,
    "f0_mean": float, "f0_std": float,
    "rms_dbfs": float,
    "hubert_valence": float,        # EMA valence_score
    "hubert_arousal": float,
    "gaze_away": bool,
    "voice_stress": bool,
    "incongruence": bool,
    "tension_score": float,         # 0-1 bileşik stres sinyali
    "is_critical_moment": bool
}
```

### 5. Zaman Blokları (build_smart_blocks)
```python
# Doğal sessizlik sınırlarında bölünmüş paragraf blokları
{
    "block_id": int,
    "label": str,                   # "MM:SS - MM:SS"
    "text": str,                    # blokta söylenen tüm metin
    "segment_count": int,
    "dominant_emotion": str,
    "avg_speech_confidence": float,
    "avg_gaze_away_pct": float,     # 0-1 kameradan uzak bakış oranı
    "hubert_valence_mean": float,
    "word_count": int
}
```

---

## Çıktı Formatları

### JSON Raporu (`reports/{interview_id}.json`)
```
{
    "interview_id": str,
    "phase": "v3",
    "video_info": {duration_seconds, fps, resolution, ...},
    "text_analysis": {segments: [...], word_count, ...},
    "voice_analysis": {per_second_timeline: [...], summary: {...}},
    "audio_signal_analysis": {timeline: [...], summary: {...}},
    "face_analysis": {timeline: [...], summary: {...}},
    "segment_signal_packages": [...],    # LLM'e giden paketler
    "time_blocks": [...],                # Paragraf blokları
    "ai_analysis": {text, provider, model, ...},
    "created_at": str
}
```

### HTML Raporu (`reports/{interview_id}.html`)
Jinja2 tabanlı, Chart.js grafikler içerir:

| Bölüm | İçerik |
|-------|--------|
| 1A — KPI Kartları | Baskın duygu, gaze oranı, konuşma güveni, stres skoru |
| 1B1 — Yüz Duygu Dağılımı | Doughnut chart (7 duygu sınıfı) |
| 1B2 — Ses Profili Dağılımı | Bar chart (5 ses profili) |
| 1C — Kritik Anlar | is_critical_moment=true segmentlerin kartları |
| 1D — Konuşma Blokları | Doğal paragraf bloklarının badge'li özeti |
| 2 — LLM Analizi | Gemini/Ollama davranışsal rapor metni |
| 3 — Grafikler | 5 matplotlib timeline chart (duygu, valence, gaze, güven, enerji) |

---

## API Endpoint'leri

```
POST   /analyze
       phase3: bool = True
       use_llm: bool = True
       llm_provider: str = "gemini"  # "gemini" | "ollama" | "none"
       → {interview_id, status, report_paths, ...}

GET    /status/{interview_id}
       → {status: "processing"|"completed"|"error", ...}

GET    /health
       → {status: "ok", models_loaded: bool, ...}

GET    /
       → API bilgisi + versiyon

Swagger UI: http://localhost:8000/docs
```

---

## Performans (Tipik Süreler)

| Adım | ~5 dk video | ~10 dk video |
|------|-------------|--------------|
| Whisper STT | ~45s | ~90s |
| FaceAnalyzer | ~80s | ~160s |
| VoiceAnalyzer | ~8s | ~15s |
| AudioSignalFusion | ~5s | ~10s |
| ContextualAggregator | <1s | <1s |
| Gemini LLM | ~15s | ~20s |
| ReportGenerator | ~5s | ~8s |
| **Toplam** | **~2.5 dk** | **~5 dk** |

> Not: FaceAnalyzer her 10. frame işler (~3 fps). GPU belleği yeterli değilse CUDA OOM olabilir.

---

## Gaze Estimator Kararı: MobileGaze (uniface) Seçildi

| Kriter | MobileGaze | GaZeL |
|--------|-----------|-------|
| Bağımlılık | uniface 3.0.0 — zaten kurulu | Ayrı paket, kurulu değil |
| API | `gaze_estimator.estimate(face_crop)` → pitch/yaw (radyan) | Farklı API, model indirme gerekli |
| Model | ResNet18 — hafif | Transformer — ağır |
| Kanıtlanmış | Stajyer kodunda çalışıyor | Test edilmemiş |
| Yeni bağımlılık riski | Sıfır | Yüksek (CUDA çakışması olasılığı) |

**Sonuç:** MobileGaze kullanılmaktadır. Pipeline oturma mülakatı için (frontal yüz, masaüstü kamera) yeterli hassasiyette.

---

## Bilinen Limitasyonlar

- Kamera açısına bağımlılık: MobileGaze kalibrasyon gerektirir. Video geneli median offset uygulanır ama aşırı açılarda hata payı artar.
- Aydınlatma duyarlılığı: DDAMFN düşük ışıkta yanlış sınıflandırabilir (emotion_confidence < 0.55 → Neutral sayılır).
- Ses profili dil bağımsız: f0/enerji kuralları dil ve kişi bazlı kalibre edilmemiştir; mutlak yorumdan kaçınılmalı.
- Türkçe LLM performansı: Gemini tercih edilir; Ollama/Gemma3:12b Türkçe yeterince güçlüdür ama daha yavaş.
- Glyph rendering: matplotlib Türkçe özel karakter için uyarı verebilir; emoji kullanılmamıştır.
