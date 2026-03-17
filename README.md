# SensifyHR FAZ-4 — Multimodal Mülakat Analiz Sistemi

Video mülakat kaydından **yüz duygusu**, **göz bakışı**, **ses profili** ve **konuşma metni** sinyallerini çıkararak davranışsal bir İK raporu üreten multimodal AI sistemi.

---

## Temel Özellikler

| Sinyal | Teknoloji | Çıktı |
|--------|-----------|-------|
| Yüz Duygusu | UniFace DDAMFN AffectNet7 | 7-sınıf duygu + güven skoru |
| Göz Bakışı | UniFace MobileGaze | pitch/yaw derece → center/up/down/left/right |
| Ses Profili | torchaudio f0+enerji kural sistemi | Canlı/Kararlı/Dengeli/Sakin/Gergin |
| Konuşma | faster-whisper-large-v3-turbo | Türkçe STT + zaman damgaları |
| LLM Yorumu | Gemini 2.5 Pro/Flash → Ollama/Gemma3:12b | Davranışsal Türkçe rapor |

---

## Kurulum

### 1. Conda Ortamı

```bash
conda create -n gpu_env_videoai python=3.10
conda activate gpu_env_videoai
```

### 2. PyTorch (CUDA 12.1)

```bash
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Proje Bağımlılıkları

```bash
pip install -r requirements.txt
```

### 4. Ollama (Yedek LLM)

```bash
# Windows: https://ollama.com/download
ollama pull gemma3:12b
ollama serve
```

### 5. Gemini API Anahtarı (Birincil LLM)

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

---

## Çalıştırma

### CLI Test

```bash
# Phase3 analiz (Gemini LLM)
python test_example.py video.mp4 --phase3

# Phase3 + Ollama (offline)
python test_example.py video.mp4 --phase3 --ollama

# LLM olmadan (sadece sinyal analizi)
python test_example.py video.mp4 --phase3 --no-llm
```

### FastAPI Sunucu

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Swagger UI: http://localhost:8000/docs
```

### API Kullanımı

```bash
# Video yükle ve analiz et
curl -X POST "http://localhost:8000/analyze?phase3=true&use_llm=true&llm_provider=gemini" \
     -F "file=@video.mp4"

# Durum kontrol
curl http://localhost:8000/status/{interview_id}
```

**Parametreler:**
- `phase3=true` — v4 analiz modunu aktif et (her zaman true kullan)
- `use_llm=true/false` — LLM analizi yap/atla
- `llm_provider=gemini|ollama|none` — LLM seçimi

---

## Çıktılar

Her analizde `reports/` klasörüne iki dosya yazılır:

- `{id}.json` — ham sinyal verileri + LLM analizi (makine tarafından okunabilir)
- `{id}.html` — İK dashboard (tarayıcıda açılır)

### Dashboard Bölümleri

| Bölüm | İçerik |
|-------|--------|
| KPI Kartları | Baskın duygu, kamera teması %, konuşma güveni, stres skoru |
| Yüz Duygu Dağılımı | 7 sınıf pasta grafik (Happy/Sad/Angry/Fear/Disgust/Surprise/Neutral) |
| Ses Profili Dağılımı | 5 profil bar grafik (Canlı/Kararlı/Dengeli/Sakin/Gergin) |
| Kritik Anlar | Yüksek gerilim veya tutarsızlık gözlemlenen anlar |
| Konuşma Blokları | Doğal paragraf blokları (duygu/güven/kamera badge'leri ile) |
| LLM Analizi | Gemini/Ollama davranışsal Türkçe rapor metni |
| Grafikler | 5 timeline chart: duygu, valence, gaze, güven, ses enerjisi |

---

## Proje Yapısı

```
SensifyHR-FAZ3/
├── src/
│   ├── vision/
│   │   ├── face_analyzer.py          # UniFace (RetinaFace + DDAMFN + MobileGaze)
│   │   └── video_processor.py        # ffmpeg audio extraction
│   ├── audio/
│   │   ├── audio_signal_fusion.py    # torchaudio ses profili → valence/arousal
│   │   ├── voice_analyzer.py         # torchaudio per-second ses özellikleri
│   │   ├── text_analyzer.py          # faster-whisper STT
│   │   └── thought_unit_merger.py    # segment birleştirici
│   ├── nlp/
│   │   ├── contextual_aggregator.py  # sinyal hizalama + akıllı paragraf bölme
│   │   ├── ollama_ai.py              # Ollama entegrasyonu
│   │   ├── gemini.py                 # Gemini API + fallback zinciri
│   │   └── prompt_phase3.txt         # LLM sistem promptu
│   ├── reporting/
│   │   ├── report_generator.py       # HTML + JSON rapor üretimi
│   │   ├── plot.py                   # matplotlib grafikler
│   │   └── templates/report_v3.html  # Jinja2 dashboard template
│   └── pipeline.py                   # Ana orchestrator
├── api/main.py                       # FastAPI endpoint'leri
├── test_example.py                   # CLI test scripti
├── requirements.txt                  # Bağımlılıklar (pin'li versiyon)
├── ARCHITECTURE.md                   # Detaylı sistem mimarisi
├── PROJE_DOKUMANTASYONU.md           # Sunum/rapor kaynağı
└── .env                              # GEMINI_API_KEY (git'e girmiyor)
```

Runtime klasörler (git'e girmiyor):
- `reports/` — üretilen raporlar
- `temp_uploads/` — API yüklemeleri için geçici (analiz sonrası temizlenir)

---

## Teknik Gereksinimler

| Bileşen | Versiyon |
|---------|---------|
| Python | 3.10 |
| CUDA | 12.1 |
| torch | 2.5.1+cu121 |
| torchaudio | 2.5.1+cu121 |
| uniface | 3.0.0 |
| faster-whisper | 1.2.1 |

GPU önerilir (NVIDIA, minimum 6GB VRAM). CPU modunda çalışır ama çok yavaştır.

---

## Lisans

Bu proje SensifyHR tarafından geliştirilmektedir.
