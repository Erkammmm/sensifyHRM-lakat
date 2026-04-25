# NOTE: librosa import here is intentional — used for audio preprocessing before Whisper. Do not remove.
"""
Metin Analizi Modülü
Phase-2 (legacy davranış): STT + Türkçe BERT sentiment
Phase-3: STT-only (sentiment YOK)

Güncelleme:
  - STT varsayılanı: whisper-large-v3-turbo (faster-whisper) -> daha kaliteli transkript
"""
import os
import re
from collections import Counter
from typing import List, Dict, Any

import librosa
import torch
from transformers import pipeline as hf_pipeline

# Amaç:
# Ortak logging altyapısını kullanmak.
# Modül tek başına çalıştırıldığında da import edilebilsin diye fallback bırakıyoruz.
try:
    from ..logging_config import get_logger
except ImportError:
    from src.logging_config import get_logger

logger = get_logger(__name__)

# --- Ayarlar ---
# STT modeli (faster-whisper)
# Not: faster-whisper genellikle CTranslate2 formatındaki modelleri sorunsuz yükler.
# "whisper-large-v3-turbo" adı bazı ortamlarda doğrudan yüklenemeyebilir; bu durumda fallback uygulanır.
STT_MODEL = os.getenv("SENSIFYHR_STT_MODEL", "whisper-large-v3-turbo").strip()


def _sanitize_stt_model_name(name: str) -> str:
    n = (name or "").strip()
    # Env yanlış set edilirse (örn. "Systran/faster-whisper-") burada güvenli default'a dön
    if not n or n.endswith(("-", ".")) or n.startswith(("-", ".")):
        return "whisper-large-v3-turbo"
    return n
SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"




def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) or ("cuda failed" in msg) or ("cublas" in msg and "alloc" in msg)


def _get_audio_duration_seconds(path: str) -> float:
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0

# Amaç:
# Phase-3 STT segmentlerini daha küçük cümle/parçalara bölmek.
# Bu, aynı segment içine hem soru hem cevap dolmasını azaltır.
_PHASE3_SPLIT_RE = re.compile(r"(?<=[\?\!\.])\s+")


def _split_phase3_segment(start: float, end: float, text: str) -> List[Dict[str, Any]]:
    text = " ".join((text or "").strip().split())
    if not text:
        return []

    parts = [p.strip() for p in _PHASE3_SPLIT_RE.split(text) if p and p.strip()]
    if len(parts) <= 1:
        return [{"start": float(start), "end": float(end), "text": text}]

    total_dur = max(0.0, float(end) - float(start))
    if total_dur <= 0.0:
        return [{"start": float(start), "end": float(end), "text": p} for p in parts]

    # Amaç:
    # Süreyi parça uzunluklarına göre oransal paylaştırmak.
    total_chars = sum(max(1, len(p)) for p in parts)
    cur = float(start)
    out = []

    for i, part in enumerate(parts):
        weight = max(1, len(part)) / total_chars
        piece_dur = total_dur * weight

        piece_start = cur
        piece_end = float(end) if i == len(parts) - 1 else min(float(end), cur + piece_dur)

        out.append(
            {
                "start": round(piece_start, 2),
                "end": round(piece_end, 2),
                "text": part,
            }
        )
        cur = piece_end

    return out

class TextAnalyzer:
    """
    Video/ses dosyasından konuşmayı metne çevirir (Whisper)
    ve Türkçe BERT ile duygu analizi yapar.
    """

    def __init__(self):
        logger.info("[%s] Başlatılıyor...", self.__class__.__name__)

        # GPU kullanımı
        self.device ="cpu"

        # Sentiment (v2 legacy) artık lazy-load: v3 modunda gereksiz model yüklemesini engeller
        self.sentiment_pipeline = None

        # STT (lazy load)
        self._fw_model = None
        self._hf_asr_pipe = None

    def _ensure_sentiment(self):
        if self.sentiment_pipeline is not None:
            return
        sentiment_device = os.getenv("SENSIFYHR_SENTIMENT_DEVICE", "cpu").strip().lower()
        use_cuda_for_sentiment = (sentiment_device == "cuda") and torch.cuda.is_available()
        device_id = 0 if use_cuda_for_sentiment else -1
        self.sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=device_id,
        )
        logger.info("[%s] Sentiment modeli yüklendi. Device=%s", self.__class__.__name__, "CUDA" if device_id == 0 else "CPU")

    def analyze_sentiment(self, text):
        """Tek bir cümlenin duygu analizini yapar."""
        if not text.strip():
            return None, 0.0
        self._ensure_sentiment()

        result = self.sentiment_pipeline(text)[0]
        label = result["label"]   # "positive" veya "negative"
        score = result["score"]
        return label, score

    def _ensure_fw_model(self):
        """faster-whisper modelini lazy-load eder."""
        if self._fw_model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise ImportError(
                "'faster-whisper' paketi gerekli. Yükleyin: pip install faster-whisper"
            ) from exc

        # faster-whisper cihaz seçimi (GPU varsa cuda)
        stt_device_pref = os.getenv("SENSIFYHR_STT_DEVICE", "auto").strip().lower()
        device = "cuda" if (torch.cuda.is_available() and stt_device_pref != "cpu") else "cpu"
        # RTX 3050 gibi düşük VRAM için default: int8_float16 (GPU) / int8 (CPU)
        default_ct = os.getenv("SENSIFYHR_FW_COMPUTE_TYPE", "").strip()
        compute_type = default_ct or ("int8_float16" if device == "cuda" else "int8")

        model_name = _sanitize_stt_model_name(STT_MODEL)

        def is_valid_repo_id(s: str) -> bool:
            s = (s or "").strip()
            if not s:
                return False
            # HF repo id kuralları (kabaca): parçalar '-' veya '.' ile başlayıp bitmemeli
            if s.startswith(("-", ".")) or s.endswith(("-", ".")):
                return False
            if " " in s:
                return False
            return True

        def build_candidates(name: str) -> List[str]:
            name = (name or "").strip()
            cands: List[str] = []

            # turbo/distil adaylarını öne al (varsa)
            if "turbo" in name.lower():
                cands.extend(
                    ["Systran/faster-whisper-large-v3-turbo", "Systran/faster-whisper-large-v3", "large-v3"]
                )
            if "distil" in name.lower():
                cands.extend(["distil-whisper/distil-large-v3", "Systran/faster-whisper-large-v3", "large-v3"])

            # kullanıcı tam repo id verdiyse olduğu gibi dene
            if name:
                cands.append(name)

            # "whisper-*" kısaltması verildiyse sadece prefix'i kaldırarak dene (replace değil!)
            if "/" not in name and name.startswith("whisper-"):
                cands.append(name[len("whisper-") :].strip())

            # "openai/whisper-*" gibi transformer repo id'leri faster-whisper ile çalışmayabilir;
            # yine de isimden kısaltma çıkarmayı deneyelim.
            if name.startswith("openai/whisper-"):
                short = name.split("/", 1)[1]
                if short.startswith("whisper-"):
                    cands.append(short[len("whisper-") :].strip())

            # dedupe + geçersizleri filtrele
            seen = set()
            out: List[str] = []
            for c in cands:
                c = (c or "").strip()
                if not c or c in seen:
                    continue
                seen.add(c)
                # built-in model adları (large-v3 vb.) için repo id validasyonu çok katı olabilir;
                # sadece bariz bozuk olanları ele.
                if "/" in c and not is_valid_repo_id(c.split("/", 1)[0]) and not is_valid_repo_id(c.split("/", 1)[1]):
                    continue
                if c.endswith(("-", ".")) or c.startswith(("-", ".")):
                    continue
                out.append(c)
            return out

        candidates = build_candidates(model_name)
        if not candidates:
            # Son çare: çalışması muhtemel iki aday
            candidates = ["Systran/faster-whisper-large-v3", "large-v3"]

        # CUDA OOM'e karşı kademeli compute_type fallback
        compute_fallbacks = [compute_type]
        if device == "cuda":
            compute_fallbacks.extend(["float16", "int8_float16", "int8"])

        cpu_threads = int(os.getenv("SENSIFYHR_STT_CPU_THREADS", "8"))
        last_exc = None
        for cand in candidates:
            for ct in compute_fallbacks:
                try:
                    self._fw_model = WhisperModel(cand, device=device, compute_type=ct, cpu_threads=cpu_threads)
                    logger.info("[%s] faster-whisper (%s) yüklendi -> %s (%s)", self.__class__.__name__, cand, device.upper(), ct)
                    return
                except Exception as exc:
                    last_exc = exc
                    self._fw_model = None
                    if _is_cuda_oom(exc) and torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Son çare: CPU int8
        if torch.cuda.is_available() and device == "cuda":
            try:
                self._fw_model = WhisperModel(candidates[0], device="cpu", compute_type="int8", cpu_threads=cpu_threads)
                logger.info("[%s] faster-whisper CPU fallback yüklendi -> CPU (int8)", self.__class__.__name__)
                return
            except Exception as exc:
                last_exc = exc
                self._fw_model = None

        raise RuntimeError(
            f"STT modeli yüklenemedi: {model_name}. Son hata: {last_exc}"
        ) from last_exc

    def _ensure_hf_asr(self, force_cpu: bool = False):
        """
        Transformers ASR pipeline ile Whisper STT (özellikle openai/whisper-large-v3-turbo) yükler.
        Bu yol timestamps destekler ve turbo modelini gerçek anlamda kullanır.
        """
        if self._hf_asr_pipe is not None:
            return

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as t_pipeline

        # HF model id: env'de openai/.. verilmediyse normalize et
        model_id = _sanitize_stt_model_name(STT_MODEL)
        if "/" not in model_id and model_id.startswith("whisper-"):
            model_id = f"openai/{model_id}"

        stt_device_pref = os.getenv("SENSIFYHR_STT_DEVICE", "auto").strip().lower()
        use_cuda = torch.cuda.is_available() and (stt_device_pref != "cpu") and (not force_cpu)
        device = "cuda:0" if use_cuda else "cpu"
        torch_dtype = torch.float16 if use_cuda else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        device_idx = 0 if use_cuda else -1
        self._hf_asr_pipe = t_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device_idx,
        )
        logger.info("[%s] transformers ASR (%s) yüklendi -> %s", self.__class__.__name__, model_id, "CUDA" if device_idx == 0 else "CPU")

    def process_video(self, video_path: str, phase3_enabled: bool = False) -> List[Dict[str, Any]]:
        """
        Video/ses dosyasını alır:
        Phase-2 (varsayılan):
        1) Whisper ile metne çevirir
        2) Her cümle için duygu analizi yapar
        3) Yapılandırılmış veri listesi döndürür

        Phase-3:
          1) faster-whisper ile daha iyi segmentleme/transkript
          2) Sentiment YOK (metin = sadece içerik)

        Returns:
            list[dict]: Her segment için:
              Phase-2:
                  {"start","end","text","sentiment","confidence"}
              Phase-3:
                  {"start","end","text"}
        """
        if not os.path.exists(video_path):
            logger.error("HATA: '%s' dosyası bulunamadı!", video_path)
            return []

        segments_data: List[Dict[str, Any]] = []
        duration_sec = _get_audio_duration_seconds(video_path)
        long_audio_cpu_threshold = float(os.getenv("SENSIFYHR_STT_LONG_AUDIO_CPU_SEC", "900"))  # 15 dk
        force_cpu_for_long_audio = duration_sec >= long_audio_cpu_threshold and duration_sec > 0

        stt_backend = os.getenv("SENSIFYHR_STT_BACKEND", "faster-whisper").strip().lower()
        use_transformers = stt_backend in ("transformers", "auto")

        # 1) Opsiyonel: Transformers pipeline (hız için default kapalı)
        if use_transformers:
            try:
                logger.info("Dosya işleniyor (transformers ASR): %s", video_path)
                self._ensure_hf_asr(force_cpu=force_cpu_for_long_audio)
                result = self._hf_asr_pipe(
                    video_path,
                    return_timestamps=True,
                    # Long-form chunking: bellek kullanımını ciddi düşürür
                    chunk_length_s=float(os.getenv("SENSIFYHR_STT_CHUNK_LENGTH_S", "30")),
                    stride_length_s=float(os.getenv("SENSIFYHR_STT_STRIDE_LENGTH_S", "5")),
                    batch_size=int(os.getenv("SENSIFYHR_STT_BATCH_SIZE", "1")),
                    generate_kwargs={"language": "turkish", "task": "transcribe"},
                )
                chunks = result.get("chunks", []) if isinstance(result, dict) else []
                for ch in chunks:
                    text = (ch.get("text", "") or "").strip()
                    ts = ch.get("timestamp", None)
                    if not text or not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
                        continue
                    start_time = float(ts[0] or 0.0)
                    end_time = float(ts[1] or start_time)

                    if phase3_enabled:
                        segments_data.extend(_split_phase3_segment(start_time, end_time, text))
                    else:
                        label, score = self.analyze_sentiment(text)
                        if label:
                            segments_data.append(
                                {
                                    "start": start_time,
                                    "end": end_time,
                                    "text": text,
                                    "sentiment": label,
                                    "confidence": score,
                                }
                            )
                if segments_data:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("Metin analizi tamamlandı: %s segment.", len(segments_data))
                    return segments_data
            except Exception as _exc:
                # CUDA OOM ise transformers'ı CPU'da tekrar dene
                if _is_cuda_oom(_exc) and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        self._hf_asr_pipe = None
                        logger.info("[TextAnalyzer] CUDA OOM -> transformers ASR CPU fallback deneniyor...")
                        self._ensure_hf_asr(force_cpu=True)
                        result = self._hf_asr_pipe(
                            video_path,
                            return_timestamps=True,
                            chunk_length_s=float(os.getenv("SENSIFYHR_STT_CHUNK_LENGTH_S", "30")),
                            stride_length_s=float(os.getenv("SENSIFYHR_STT_STRIDE_LENGTH_S", "5")),
                            batch_size=int(os.getenv("SENSIFYHR_STT_BATCH_SIZE", "1")),
                            generate_kwargs={"language": "turkish", "task": "transcribe"},
                        )
                        chunks = result.get("chunks", []) if isinstance(result, dict) else []
                        for ch in chunks:
                            text = (ch.get("text", "") or "").strip()
                            ts = ch.get("timestamp", None)
                            if not text or not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
                                continue
                            start_time = float(ts[0] or 0.0)
                            end_time = float(ts[1] or start_time)
                            if phase3_enabled:
                                segments_data.extend(_split_phase3_segment(start_time, end_time, text))
                            else:
                                label, score = self.analyze_sentiment(text)
                                if label:
                                    segments_data.append(
                                        {
                                            "start": start_time,
                                            "end": end_time,
                                            "text": text,
                                            "sentiment": label,
                                            "confidence": score,
                                        }
                                    )
                        if segments_data:
                            logger.info("Metin analizi tamamlandı: %s segment.", len(segments_data))
                            return segments_data
                    except Exception:
                        self._hf_asr_pipe = None
                # transformers-only modunda hata yukarı fırlasın
                if stt_backend == "transformers":
                    raise

        # 2) Fallback: faster-whisper
        logger.info("Dosya işleniyor (faster-whisper fallback): %s", video_path)
        self._ensure_fw_model()

        # Mülakat bağlamı prompt'u: Whisper'ın Türkçe mülakat transkriptini iyileştirir
        _interview_prompt = (
            "Bu bir Türkçe iş mülakatıdır. Mülakatçı sorular soruyor, aday cevap veriyor. "
            "Konuşmacılar sırayla konuşuyor."
        )

        try:
            logger.info("[TextAnalyzer] Transcribe cagrisi hazirlaniyor (VAD filter=True, beam_size=1)...")
            segments, _info = self._fw_model.transcribe(
                video_path,
                language="tr",
                vad_filter=True,
                word_timestamps=False,
                beam_size=1,
                initial_prompt=_interview_prompt,
                condition_on_previous_text=False,
            )
            logger.info("[TextAnalyzer] Transcribe çağrıldı (dil: %s). İlk segment bekleniyor (CPU'da işlem yapıyorsa VAD ve ilk ses analizi birkaç dakika sürebilir)...", _info.language)
        except Exception as exc:
            # Transcribe sırasında CUDA OOM olursa CPU int8 ile tekrar dene
            if _is_cuda_oom(exc) and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                from faster_whisper import WhisperModel

                logger.info("[TextAnalyzer] CUDA OOM -> faster-whisper CPU(int8) retry...")
                model_name = _sanitize_stt_model_name(STT_MODEL)
                retry_name = "Systran/faster-whisper-large-v3" if "turbo" in model_name.lower() else model_name
                cpu_threads_retry = int(os.getenv("SENSIFYHR_STT_CPU_THREADS", "8"))
                self._fw_model = WhisperModel(retry_name, device="cpu", compute_type="int8", cpu_threads=cpu_threads_retry)
                segments, _info = self._fw_model.transcribe(
                    video_path,
                    language="tr",
                    vad_filter=True,
                    word_timestamps=False,
                    beam_size=1,
                    initial_prompt=_interview_prompt,
                    condition_on_previous_text=False,
                )
            else:
                raise

        segment_count = 0
        for seg in segments:
            segment_count += 1
            logger.info("[TextAnalyzer] Segment %s çözüldü: %.2f - %.2f", segment_count, getattr(seg, "start", 0.0), getattr(seg, "end", 0.0))
            start_time = float(getattr(seg, "start", 0.0))
            end_time = float(getattr(seg, "end", 0.0))
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue

            if phase3_enabled:
                segments_data.extend(_split_phase3_segment(start_time, end_time, text))
            else:
                label, score = self.analyze_sentiment(text)
                if label:
                    segments_data.append(
                        {
                            "start": start_time,
                            "end": end_time,
                            "text": text,
                            "sentiment": label,
                            "confidence": score,
                        }
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Metin analizi tamamlandı: %s segment.", len(segments_data))
        return segments_data

    @staticmethod
    def get_summary(segments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Metin analizi sonuçlarının özetini döndürür.

        Returns:
            dict: {
                "total_sentences": int,
                "sentiment_distribution": {"positive": int, "negative": int},
                "sentiment_percentages": {"positive": float, "negative": float},
                "dominant_sentiment": str
            }
        """
        if not segments_data:
            return {
                "total_sentences": 0,
                "sentiment_distribution": {},
                "sentiment_percentages": {},
                "dominant_sentiment": "Veri Yok",
            }

        # Phase-3 (sentiment alanı yok): sadece sayısal özet ver
        if segments_data and isinstance(segments_data[0], dict) and "sentiment" not in segments_data[0]:
            return {
                "total_sentences": len(segments_data),
                "sentiment_distribution": {},
                "sentiment_percentages": {},
                "dominant_sentiment": "N/A (phase3)",
            }

        labels = [s["sentiment"] for s in segments_data if "sentiment" in s]
        counter = Counter(labels)
        total = len(labels)

        percentages = {k: round((v / total) * 100, 1) for k, v in counter.items()}
        dominant = counter.most_common(1)[0][0] if counter else "Veri Yok"

        return {
            "total_sentences": total,
            "sentiment_distribution": dict(counter),
            "sentiment_percentages": percentages,
            "dominant_sentiment": dominant,
        }


if __name__ == "__main__":
    analyzer = TextAnalyzer()
    logger.info("TextAnalyzer modülü hazır.")
