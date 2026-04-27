# SensifyHR FAZ-5 — Claude Code Master Guide

## Project Overview
AI-powered video interview analysis system. Analyzes a candidate's video recording by extracting
face/emotion signals, gaze tracking, speech features, and transcribed text — aligns them into
time-based segment packages — and feeds them to an LLM to generate an HR evaluation report.

**FAZ-5 aktif.** Rapor sadeleştirme, prompt mühendisliği, gaze delta analizi ve metin tabanlı davranışsal analiz geliştirmeleri yapılıyor.

---

## Repository Structure
```
SensifyHR-FAZ3/
├── src/
│   ├── vision/
│   │   ├── face_analyzer.py          # UniFace: RetinaFace + DDAMFN AffectNet7 + MobileGaze
│   │   └── video_processor.py        # DO NOT TOUCH
│   ├── audio/
│   │   ├── audio_signal_fusion.py    # torchaudio f0+enerji kural sistemi → valence/arousal
│   │   ├── voice_analyzer.py         # torchaudio per-second ses özellikleri
│   │   ├── text_analyzer.py          # DO NOT TOUCH (faster-whisper STT)
│   │   └── thought_unit_merger.py    # DO NOT TOUCH
│   ├── nlp/
│   │   ├── contextual_aggregator.py  # Segment signal packages + build_smart_blocks()
│   │   ├── ollama_ai.py              # DO NOT TOUCH
│   │   ├── gemini.py                 # DO NOT TOUCH
│   │   └── prompt_phase3.txt         # 5 sinyal kaynağı, 7 çıktı bölümü
│   ├── reporting/                    # FAZ-5: rapor yeniden yapılandırılıyor
│   └── pipeline.py                   # Ana orchestrator
├── api/main.py                       # DO NOT TOUCH
├── test_example.py                   # DO NOT TOUCH
├── ARCHITECTURE.md                   # Sistem mimarisi (güncel)
├── PROJE_DOKUMANTASYONU.md           # Sunum/rapor için kaynak
└── requirements.txt                  # Güncel bağımlılıklar
```

---

## Mevcut FAZ-4 Mimarisi

### 1. face_analyzer.py — UniFace
- **RetinaFace** (MNET_025): yüz tespiti
- **DDAMFN AffectNet7**: 7-sınıf duygu sınıflandırması (Happy/Sad/Angry/Fear/Disgust/Surprise/Neutral) + confidence
- **MobileGaze** (ResNet18): gaze pitch_deg / yaw_deg
- Her 10. frame; timestamp_sec = frame_index / fps
- face_detected=False → boş kayıt, crash yok

### 2. voice_analyzer.py — torchaudio
- Per-second segmentation (1s windows)
- VAD: RMS + ZCR + spectral flatness
- Özellikler: rms_dbfs, f0_mean, f0_std, spectral_centroid, spectral_flatness, mel_energy
- Türetilmiş: konusma_guveni, konusma_stili, konusma_enerjisi

### 3. audio_signal_fusion.py — torchaudio kural sistemi
- **SER modeli KALDIRILDI** (ehcalabres/wav2vec2 → Türkçe için güvenilir değildi)
- Yeni sistem: f0_std + rms_dbfs → 5 ses profili → valence/arousal
- Profiller: Canlı / Kararlı / Dengeli / Sakin / Gergin
- Eşikler: rms > -25 dBFS → high; f0_std > 50 Hz → high varyasyon
- EMA yumuşatma (alpha=0.65)

### 4. contextual_aggregator.py
- `build_segment_signal_packages()`: STT segment başına tüm sinyalleri hizalar
- `build_smart_blocks()`: doğal sessizlik sınırlarına göre paragraf bloklarına böler
  - Kural: gap > 2s + 50 kelime → kes; 150 kelime + gap > 0.05s → kes; 200 kelime → zorla kes
- Gaze offset normalizasyonu: video geneli median ile bias düzeltmesi

### 5. LLM Zinciri
- Birincil: Gemini 2.5 Pro → 2.5 Flash → 2.0 Flash
- Yedek: Ollama (Gemma3:12b)
- Son yedek: graceful skip (rapor LLM olmadan devam eder)

---

## API Endpoints

```
POST /analyze
  Parametreler:
    phase3: bool = True        # v3 analiz (her zaman true kullan)
    use_llm: bool = True       # LLM analizi yap/atla
    llm_provider: str = "gemini"  # "gemini" | "ollama" | "none"

GET /status/{interview_id}     # Analiz durumu
GET /health                    # Sistem sağlık kontrolü
GET /                          # API bilgisi

Swagger: http://localhost:8000/docs
```

---

## Technical Constraints
| Parameter | Value |
|-----------|-------|
| Python | 3.10 |
| Conda env | gpu_env_videoai |
| CUDA | 12.1 |
| torch | 2.5.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| uniface | 3.0.0 |
| Whisper | faster-whisper-large-v3-turbo (Systran) |
| LLM birincil | Gemini 2.5 Pro/Flash |
| LLM yedek | Ollama + Gemma3:12b |

**Hard rules:**
- NO tkinter anywhere (headless pipeline)
- NO mediapipe anywhere
- NO SER modeli (wav2vec2 tabanlı) — kural sistemi kullan
- librosa: sadece gerekliyse; torchaudio tercih et

---

## Do NOT Touch
- `src/audio/text_analyzer.py`
- `src/audio/thought_unit_merger.py`
- `src/nlp/ollama_ai.py`
- `src/nlp/gemini.py`
- `api/main.py`
- `test_example.py`

## FAZ-5 Değiştirilecek Dosyalar
- `src/reporting/report_generator.py` — hero kart kaldırıldı, yeni bölümler eklendi
- `src/reporting/templates/report_v3.html` — 7 bölümlü IK odaklı yeni template
- `src/nlp/prompt_phase3.txt` — XML tabanlı yeni prompt, 5 bölüm çıktı
- `src/nlp/contextual_aggregator.py` — Gemini'ye giden veri filtrelendi, mülakatçı ayrımı kaldırıldı
- `src/vision/face_analyzer.py` — delta tabanlı gaze analizi, telefon videosu desteği

---

## Success Criteria (FAZ-4 — tamamlandı)
1. `python test_example.py video.mp4 --phase3 --ollama` runs without errors
2. Every segment package contains: emotion_label + gaze_direction + speech_style
3. HTML report shows time blocks with behavioral indicators
4. Zero mediapipe imports in any active file
5. Zero SER model imports in audio_signal_fusion.py
6. Voice profile distribution (Canlı/Kararlı/Dengeli/Sakin/Gergin) in dashboard

## Success Criteria (FAZ-5 — aktif)
1. HTML raporda f0, pitch, tension_score, rms, hubert terimleri yok
2. Hero kart (puan kartı) görünmüyor
3. Gemini çıktısı `<rapor>` XML yapısında geliyor ve parse ediliyor
4. Rapor şu 7 bölümden oluşuyor: Header → Genel İzlenim → Güçlü/Gelişim → Duygusal Seyir → İçerik Analizi → Davranışsal Uyarılar → IK Soruları
5. Metin tabanlı davranışsal analiz bölümü: özgüven, tecrübe tutarlılığı, motivasyon, iletişim tarzı
6. Gaze delta analizi: baseline + sapma bazlı, telefon videosu desteği
7. Mülakatçı/aday ayırma kural mantığı tamamen kaldırıldı
