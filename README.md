# 🎯 SensifyHR v3.0 – Multimodal Signal Fusion + LLM Reasoning

**Multimodal yapay zekâ destekli online mülakat değerlendirme sistemi.**

Video kaydı üzerinden adayın **metin (ne söylüyor)**, **ses (nasıl söylüyor)** ve **görsel davranış (nasıl davranıyor)** sinyallerini analiz ederek karar destekleyici bir İK raporu üretir.

Bu sürümde temel ilke:
- **Metin = sadece içerik (STT)**
- **Ses & görüntü = sadece sinyal**
- **LLM = tek karar verici (signal reasoning)**
- Sistem **emotion/sentiment sınıflandırmaz**.

---

## 🚀 Özellikler

| Modül | Teknoloji | Açıklama |
|-------|-----------|----------|
| 📝 Metin (STT) | `openai/whisper-large-v3-turbo` (Transformers) + fallback | Metin = **sadece içerik** (`{start,end,text}`) |
| 🎛️ Audio Signal | HuBERT SER projection + librosa | **Valence/Arousal + fiziksel ses state’leri** (emotion etiketi yok) |
| 👁️ Visual Signal | MediaPipe FaceLandmarker | **facial_state / attention_state / stress_indicator** (emotion etiketi yok) |
| 🔊 Ses Özellikleri | Librosa | RMS enerji, Pitch, VAD, Mel spectrogram |
| 🔗 Contextual Aggregator | `contextual_aggregator.py` | Metin+Audio+Visual sinyalleri **segment bazlı hizalar** (LLM input) |
| 🤖 LLM Reasoning | **Ollama/Gemma12B** (+ opsiyonel Gemini) | Segment paketlerini chunk’layıp yorumlar |
| 📊 Görselleştirme | Matplotlib | v3’te **signal** grafikleri + teknik grafikler “Detaylar”da |
| 📄 Raporlama | HTML + JSON | v3’te **Product Mode** (tab’lı UI: Dashboard / LLM / Detaylar) |

---

## 📦 Kurulum

### 1. Ortam Hazırlığı (Anaconda + GPU)

```bash
# Sanal ortam oluştur (CUDA 12.1)
conda create -n sensifyhr python=3.10
conda activate sensifyhr

# PyTorch GPU kurulumu
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Proje bağımlılıkları
pip install -r requirements.txt

# veya lock dosyası ile (tam sürüm eşleşmesi):
pip install -r requirements.lock.sonn.txt
```

### 2. Model Dosyaları

```bash
# MediaPipe FaceLandmarker modeli (otomatik indirilir veya manuel)
mkdir -p weights
wget -O weights/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 3. (Opsiyonel) Gemini API Anahtarı

```bash
# .env dosyası oluştur
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 4. Ollama + Gemma12B (Önerilen / Primary)

```bash
ollama serve
ollama pull gemma3:12b
```

---

## 🎯 Kullanım

### Hızlı Test (CLI)

```bash
# v3 (FAZ-3) analiz
python test_example.py video.mp4 --phase3

# v3 + Ollama
python test_example.py video.mp4 --phase3 --ollama
```

### FastAPI (Web API)

```bash
# Sunucuyu başlat
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Swagger UI: http://localhost:8000/docs
# Video yükle → JSON + HTML rapor al
```

v3 için:
- `POST /analyze?phase3=true`
- LLM seçimi: `llm_provider=ollama|gemini|none`

### Ollama + Gemma ile Test

```bash
# 1) Ollama'yı kurun
#    Windows: https://ollama.com/download
#    Linux:   curl -fsSL https://ollama.com/install.sh | sh

# 2) Gemma modelini indirin (12B parametre - en iyi sonuç)
ollama pull gemma3:12b

# 3) Servisi başlatın (arka planda çalışır)
ollama serve

# 4) Test edin
python test_example.py video.mp4 --ollama
```

---

## 🐳 Docker

```bash
# Build
docker build -t sensifyhr .

# Çalıştır (GPU)
docker run --gpus all -p 8000:8000 --env-file .env sensifyhr

# Çalıştır (CPU only)
docker run -p 8000:8000 -e CUDA_VISIBLE_DEVICES="" --env-file .env sensifyhr
```

---

## 📁 Proje Yapısı

```
sensifyHRMülakay/
├── src/
│   ├── text_analyzer.py          # STT (turbo) -> segment text
│   ├── audio_signal_fusion.py    # HuBERT SER projection + librosa -> audio_signal
│   ├── contextual_aggregator.py  # Segment signal package builder (LLM input)
│   ├── face_analyzer.py          # Visual signal (facial/attention/stress) + gaze/blink
│   ├── voice_analyzer.py         # Librosa ham ses özellikleri (detay grafikler)
│   ├── video_processor.py        # Audio çıkarma (WAV)
│   ├── pipeline.py               # Ana pipeline
│   ├── ollama_ai.py              # Ollama + Gemma LLM reasoning (chunking + synthesis)
│   ├── gemini.py                 # Opsiyonel: Gemini LLM
│   ├── report_generator.py       # Product Mode HTML + JSON
│   ├── plot.py                   # Signal grafikleri + teknik grafikler
│   └── prompt_phase3.txt         # v3 prompt (signal reasoning, TR/HR)
├── api/
│   └── main.py               # FastAPI endpoint'leri
├── weights/
│   └── face_landmarker.task   # MediaPipe model dosyası
├── stajyer_çalışma/           # Referans: stajyer çalışma dosyaları
├── Dockerfile
├── .env                       # GEMINI_API_KEY (gitignore'da)
├── requirements.txt           # Temiz gereksinimler
├── requirements.lock.sonn.txt # Lock dosyası (tam sürümler)
└── test_example.py            # Hızlı test scripti
```

---

## 📊 Rapor İçeriği

### JSON Rapor
- `phase`: `"v3"`
- `text_analysis`: `{start,end,text}` segmentleri (sentiment yok)
- `audio_signal_analysis`: audio signal timeline + özet
- `visual_signal_analysis`: visual signal timeline + özet
- `voice_analysis`: RMS enerji, Pitch, konuşma/sessizlik metrikleri
- `segment_signal_packages`: LLM’ye giden zaman hizalı paketler
- `ai_analysis`: LLM değerlendirmesi (Ollama/Gemma12B veya Gemini)

### HTML Rapor
- **v3 Product Mode**: Dashboard / LLM / Detaylar tabları
- Dashboard’da: sinyal KPI’ları + “Kritik Anlar” + “Soft Skill Karnesi” kartları
- Detaylar’da: teknik grafikler

---

## 🛠️ Teknoloji Yığını

| Kategori | Araç |
|----------|------|
| STT | `openai/whisper-large-v3-turbo` (Transformers) |
| Audio Signal | HuBERT SER projection `SeaBenSea/hubert-large-turkish-speech-emotion-recognition` + librosa |
| Visual Signal | MediaPipe FaceLandmarker (blendshape) |
| Ses Özellik | Librosa (RMS, Pitch, VAD, Mel) |
| AI Değerlendirme | **Ollama + Gemma3:12b** (+ opsiyonel Gemini) |
| API | FastAPI + Uvicorn |
| Görselleştirme | Matplotlib |
| Raporlama | Jinja2 + HTML |
| GPU | PyTorch CUDA 12.1 |

---

## 📜 Lisans

Bu proje SensifyHR tarafından geliştirilmektedir.
