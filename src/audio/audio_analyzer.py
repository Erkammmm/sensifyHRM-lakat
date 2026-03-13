# LEGACY: FAZ-2 only — not used in phase3 pipeline. Do not import from phase3 code.
"""
Ses Duygu Analizi Modülü
HuBERT tabanlı Türkçe SER modeli ile ses duygu tahmini yapar (legacy / v2).

Not:
  - FAZ-3 (v3) modunda bu "duygu" çıktıları kullanılmaz; yalnızca signal (valence/arousal) kullanılır.
  - Bu modül v2 geriye uyumluluk için tutulur.
"""

import os
import subprocess
import uuid
import random
import gc

import numpy as np
import librosa
import torch
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from transformers import (
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    AutoConfig,
)
# hf_hub_download legacy model patching için kullanılıyordu; HuBERT SER için gerekmez.
from huggingface_hub import hf_hub_download

# --- Ayarlar ---
MODEL_ID = "SeaBenSea/hubert-large-turkish-speech-emotion-recognition"
TARGET_SR = 16000
CHUNK_SEC = 3.0    # Her parçanın süresi (saniye)
STEP_SEC = 1.0     # Kaydırma miktarı (saniye)
WINDOW_SIZE = 5     # Grafik yumuşatma penceresi


class AudioAnalyzer:
    """
    Video/ses dosyasından duygu analizi yapar.
    HuBERT tabanlı Türkçe SER modeli kullanılır (HuggingFace).
    """

    def __init__(self):
        print(f"[{self.__class__.__name__}] Başlatılıyor...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._set_deterministic()

        # Lazy-load: GPU OOM riskini azaltır (STT önce yüklensin)
        self.config = None
        self.feature_extractor = None
        self.model = None
        print(f"[{self.__class__.__name__}] Hazır! (lazy-load) Cihaz tercihi: {self.device.upper()}")

    def _ensure_model(self):
        if self.model is not None:
            return
        try:
            self.config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
            # Bu modelin checkpoint'inde projector boyutu 1024 görünüyor (model card çıktısı ile uyumlu)
            # Aksi halde classifier/projector ağırlıkları yüklenemeyip rastgele initialize edilebiliyor.
            if hasattr(self.config, "classifier_proj_size"):
                self.config.classifier_proj_size = 1024
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
            self.model, loading_info = AutoModelForAudioClassification.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                use_safetensors=True,
                output_loading_info=True,
                config=self.config,
            )
            missing = set((loading_info or {}).get("missing_keys", []) or [])
            if any(k.startswith("classifier.") or k.startswith("projector.") for k in missing):
                print("[AudioAnalyzer] UYARI: classifier/projector ağırlıkları eksik görünüyor. Remap ile tekrar yükleniyor...")
                self.model = AutoModelForAudioClassification.from_config(self.config)
                state_dict = _download_and_load_state_dict(MODEL_ID)
                state_dict = _remap_classifier_keys(state_dict)
                state_dict = _filter_state_dict_by_shape(self.model, state_dict)
                self.model.load_state_dict(state_dict, strict=False)
            try:
                self.model.to(self.device)
            except Exception as exc:
                # CUDA OOM veya benzeri -> CPU fallback
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.model.to(self.device)
            self.model.eval()
            print(f"[{self.__class__.__name__}] Ses modeli yüklendi -> {self.device.upper()}")
        except Exception as e:
            print(f"Ses Modeli Yükleme Hatası: {e}")
            raise

    def _set_deterministic(self, seed=42):
        """Sonuçların tekrarlanabilir olması için seed ayarla."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _set_deterministic(self, seed=42):
        """Sonuçların tekrarlanabilir olması için seed ayarla."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def process_video(self, video_path):
        """
        Video/ses dosyasını alır, ses analizi yapar ve yapılandırılmış zaman serisi döndürür.

        Returns:
            List[Dict[str, Any]]: {"start","end","emotion","confidence"}
        """
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        if not os.path.exists(video_path):
            print(f"Dosya bulunamadı: {video_path}")
            return []

        print(f"Ses işleniyor: {video_path}")
        # inline: convert video -> temporary wav (ffmpeg)
        unique_name = f"temp_{str(uuid.uuid4())[:8]}.wav"
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(TARGET_SR),
            "-ac",
            "1",
            unique_name,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        try:
            subprocess.run(cmd, check=True)
            wav_file = unique_name
        except Exception as e:
            print(f"FFmpeg Hatası: {e}")
            return []

        try:
            y, sr = librosa.load(wav_file, sr=TARGET_SR, dtype=np.float32)
        except Exception as e:
            print(f"Ses okuma hatası: {e}")
            if os.path.exists(wav_file):
                os.remove(wav_file)
            return []

        duration = librosa.get_duration(y=y, sr=sr)
        timeline = []

        print(f"Ses süresi: {duration:.2f} sn. Analiz ediliyor...")

        cursor = 0.0
        while cursor + CHUNK_SEC <= duration:
            start_sample = int(cursor * sr)
            end_sample = int((cursor + CHUNK_SEC) * sr)
            chunk = y[start_sample:end_sample].copy()
            # inline prediction logic (single-use helper inlined)
            self._ensure_model()
            if np.isnan(chunk).any():
                emotion, confidence = "SESSİZLİK", 0.0
            else:
                rms_val = np.mean(librosa.feature.rms(y=chunk)) if chunk.size else 0.0
                if rms_val < 0.002:
                    emotion, confidence = "SESSİZLİK", 0.0
                else:
                    inputs = self.feature_extractor(
                        chunk, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
                    ).to(self.device)
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    score, predicted_id = torch.max(probs, dim=-1)
                    raw_label = self.config.id2label[predicted_id.item()]
                    l = (raw_label or "").strip().lower()
                    normalize = {"anger": "angry", "fearful": "fear", "joy": "happy"}
                    l = normalize.get(l, l)
                    tr_mapping = {
                        "angry": "KIZGIN",
                        "calm": "SAKİN",
                        "happy": "MUTLU",
                        "sad": "ÜZGÜN",
                        "neutral": "NÖTR",
                        "fear": "KORKU",
                        "disgust": "TİKSİNME",
                        "surprised": "ŞAŞIRMA",
                    }
                    emotion, confidence = tr_mapping.get(l, raw_label), score.item()

            if emotion != "HATALI":
                data_packet = {
                    "start": round(cursor, 2),
                    "end": round(cursor + CHUNK_SEC, 2),
                    "emotion": emotion,
                    "confidence": round(confidence, 4),
                }
                timeline.append(data_packet)

            cursor += STEP_SEC

        if os.path.exists(wav_file):
            os.remove(wav_file)

        print(f"Ses duygu analizi tamamlandı: {len(timeline)} parça.")
        return timeline

    @staticmethod
    def get_summary(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ses duygu analizi sonuçlarının özetini döndürür."""
        if not timeline:
            return {
                "total_chunks": 0,
                "emotion_distribution": {},
                "emotion_percentages": {},
                "dominant_emotion": "Veri Yok",
                "avg_confidence": 0.0,
            }

        emotions = [t["emotion"] for t in timeline]
        counter = Counter(emotions)
        total = len(emotions)

        percentages = {k: round((v / total) * 100, 1) for k, v in counter.items()}
        dominant = counter.most_common(1)[0][0] if counter else "Veri Yok"
        avg_conf = round(np.mean([t["confidence"] for t in timeline]), 4)

        return {
            "total_chunks": total,
            "emotion_distribution": dict(counter),
            "emotion_percentages": percentages,
            "dominant_emotion": dominant,
            "avg_confidence": avg_conf,
        }


def _download_and_load_state_dict(model_id: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file as st_load

        path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        return st_load(path)
    except Exception:
        pass
    path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    return torch.load(path, map_location="cpu")


def _remap_classifier_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        if "classifier.dense" in k:
            nk = k.replace("classifier.dense", "projector")
        elif "classifier.output" in k:
            nk = k.replace("classifier.output", "classifier")
        new_sd[nk] = v
    return new_sd


def _filter_state_dict_by_shape(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    strict=False bile shape mismatch'te hata verir; bu yüzden sadece şekli uyan key'leri yükleriz.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    model_sd = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        try:
            if tuple(model_sd[k].shape) != tuple(v.shape):
                continue
        except Exception:
            continue
        filtered[k] = v
    return filtered


if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    print("AudioAnalyzer modülü hazır.")
