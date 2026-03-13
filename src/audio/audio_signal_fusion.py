"""
FAZ-3 Audio Signal Fusion  (FAZ-4 güncelleme: librosa → torchaudio)

Amaç:
  - Duygu sınıflandırması YOK
  - Ses + öğrenilmiş zayıf sinyal (HuBERT SER) -> Valence/Arousal sinyali
  - Fiziksel ses özellikleri (torchaudio) -> yorumlanabilir discrete state'ler
  - Nihai çıktı: emotion label değil, sinyal state paketleri

Not:
  - SER modelinin çıktıları "duygu" olarak raporlanmaz.
  - Sadece valence/arousal projeksiyonu için zayıf sinyal olarak kullanılır.
  - HuBERT SER modeli ve inference mantığı değiştirilmedi.
  - Yalnızca librosa önişleme adımları torchaudio karşılıklarıyla değiştirildi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
from huggingface_hub import hf_hub_download


# FAZ-3'te SER için tek kaynak model (değiştirilmesi istenmiyor)
SER_MODEL_ID = "SeaBenSea/hubert-large-turkish-speech-emotion-recognition"

TARGET_SR = 16000
CHUNK_SEC = 3.0
STEP_SEC = 3.0  # non-overlapping chunks (FAZ-4 optimisation: was 1.0)
EMA_ALPHA = float(os.getenv("SENSIFYHR_AUDIO_SIGNAL_EMA_ALPHA", "0.65"))

EMOTION_TO_SIGNAL: Dict[str, Dict[str, float]] = {
    "happy":   {"valence": +1.0, "arousal": +0.6},
    "sad":     {"valence": -0.8, "arousal": -0.4},
    "fear":    {"valence": -0.7, "arousal": +0.7},
    "angry":   {"valence": -0.9, "arousal": +0.9},
    "neutral": {"valence":  0.0, "arousal":  0.0},
}

_EPS = 1e-8


def _bucketize_valence(v: float) -> str:
    if v <= -0.25:
        return "NEGATIVE"
    if v >= 0.25:
        return "POSITIVE"
    return "NEUTRAL"


def _bucketize_arousal(a: float) -> str:
    if a <= 0.20:
        return "LOW"
    if a >= 0.60:
        return "HIGH"
    return "MEDIUM"


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def _normalize_label(label: str) -> str:
    """
    Model label'larını beklenen kümeye normalize etmeye çalışır.
    Beklenen temel etiketler: happy, sad, fear, angry, neutral
    """
    if not label:
        return "neutral"
    l = label.strip().lower()

    # Bazı modellerde fearful/fear, anger/angry, joy/happy vb. görülebiliyor.
    mapping = {
        "fearful":   "fear",
        "anger":     "angry",
        "joy":       "happy",
        "happiness": "happy",
        "sadness":   "sad",
        "calm":      "neutral",
    }
    l = mapping.get(l, l)
    if l not in EMOTION_TO_SIGNAL:
        return "neutral"
    return l


@dataclass
class SerProjectionResult: pass


class HuBERTSerProjector:
    """
    HuBERT tabanlı SER modelini yükler ve çıktıları valence/arousal sinyaline projekte eder.
    Model ve inference mantığı değiştirilmedi.
    """

    def __init__(self, model_id: str = SER_MODEL_ID):
        print(f"[HuBERTSerProjector] Başlatılıyor... Model: {model_id}")
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(self.config, "classifier_proj_size"):
            self.config.classifier_proj_size = 1024
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        # Model yükle (head ağırlıkları eşleşmiyorsa manual remap fallback)
        self.model, loading_info = AutoModelForAudioClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True,
            output_loading_info=True,
            config=self.config,
        )
        missing = set((loading_info or {}).get("missing_keys", []) or [])
        # Kritik head anahtarları missing ise sonuçlar rastgeleleşir -> remap ile tekrar yükle
        if any(k.startswith("classifier.") or k.startswith("projector.") for k in missing):
            print("[HuBERTSerProjector] UYARI: classifier/projector ağırlıkları eksik görünüyor. Remap ile tekrar yükleniyor...")
            self.model = AutoModelForAudioClassification.from_config(self.config)
            state_dict = _download_and_load_state_dict(model_id)
            state_dict = _remap_classifier_keys(state_dict)
            state_dict = _filter_state_dict_by_shape(self.model, state_dict)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[HuBERTSerProjector] Hazır! Cihaz: {self.device.upper()}")

    def project(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        if audio.size == 0:
            return {
                "valence_score": 0.0,
                "arousal_score": 0.0,
                "valence_state": "NEUTRAL",
                "arousal_state": "LOW",
                "top_label": "neutral",
                "top_score": 0.0,
            }

        # Sessizlik kontrolü (çok düşük enerji -> nötr)
        # [librosa.feature.rms → numpy RMS]
        rms_val = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        if rms_val < 0.002:
            return {
                "valence_score": 0.0,
                "arousal_score": 0.0,
                "valence_state": "NEUTRAL",
                "arousal_state": "LOW",
                "top_label": "neutral",
                "top_score": 0.0,
            }

        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = _safe_softmax(logits)[0]  # (num_labels,)
        probs_np = probs.detach().float().cpu().numpy()

        labels = [self.config.id2label[i] for i in range(len(probs_np))]
        normalized = [_normalize_label(l) for l in labels]

        valence = 0.0
        arousal = 0.0
        for p, lab in zip(probs_np, normalized):
            sig = EMOTION_TO_SIGNAL.get(lab, EMOTION_TO_SIGNAL["neutral"])
            valence += float(p) * float(sig["valence"])
            arousal += float(p) * float(sig["arousal"])

        # debug: top-1
        top_idx   = int(np.argmax(probs_np)) if probs_np.size else 0
        top_label = normalized[top_idx] if normalized else "neutral"
        top_score = float(probs_np[top_idx]) if probs_np.size else 0.0

        return {
            "valence_score": float(valence),
            "arousal_score": float(arousal),
            "valence_state": _bucketize_valence(float(valence)),
            "arousal_state": _bucketize_arousal(float(arousal)),
            "top_label":     top_label,
            "top_score":     top_score,
        }


def _download_and_load_state_dict(model_id: str) -> Dict[str, torch.Tensor]:
    """
    HF'den weight dosyasını indirip state_dict döndürür.
    Öncelik: model.safetensors -> pytorch_model.bin
    """
    try:
        from safetensors.torch import load_file as st_load
        path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        return st_load(path)
    except Exception:
        pass

    path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    return torch.load(path, map_location="cpu")


def _remap_classifier_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Bazı checkpoint'lerde head isimleri farklı olabilir.
    Örn:
      - classifier.dense.*  -> projector.*
      - classifier.output.* -> classifier.*
    """
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


def _filter_state_dict_by_shape(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
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


def _energy_state(rms_mean: float) -> str:
    # RMS absolute thresholds after normalize; heuristic buckets.
    if rms_mean < 0.02:
        return "LOW"
    if rms_mean < 0.05:
        return "MEDIUM"
    return "HIGH"


def _pitch_stability_state(pitch_std: float, jump_count: int) -> str:
    # Daha stabil ses: düşük varyans + az sıçrama
    if pitch_std <= 30.0 and jump_count <= 2:
        return "STABLE"
    return "UNSTABLE"


def _speech_rate_state(onsets_per_sec: float) -> str:
    # Onset yoğunluğu ile kaba konuşma hızı tahmini (heuristic)
    if onsets_per_sec < 2.0:
        return "SLOW"
    if onsets_per_sec < 4.0:
        return "NORMAL"
    return "FAST"


def _estimate_onsets_per_second(audio: np.ndarray, sr: int) -> float:
    """
    Konuşma hızı proxy'si: torchaudio tabanlı enerji zarfı + peak picking.

    [librosa.onset.onset_strength + librosa.util.peak_pick →
     torch frame RMS + half-wave rectified diff + numpy peak picking]
    """
    if audio.size == 0:
        return 0.0
    duration = float(len(audio)) / float(sr) if sr > 0 else 0.0
    if duration <= 0.0:
        return 0.0

    # Frame-level RMS enerji zarfı (25ms frames, 10ms hop)
    frame_len = max(2, int(sr * 0.025))
    hop_len   = max(1, int(sr * 0.010))

    tensor = torch.from_numpy(audio).float()
    if tensor.numel() < frame_len:
        return 0.0

    frames = tensor.unfold(0, frame_len, hop_len)            # (T, frame_len)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + _EPS)  # (T,)

    if rms.numel() < 2:
        return 0.0

    # Half-wave rectified first difference → onset strength zarfı
    onset_env = torch.clamp(rms[1:] - rms[:-1], min=0.0).numpy()
    if onset_env.size < 5:
        return 0.0

    # Adaptif delta: mean + 0.5*std
    delta = float(np.mean(onset_env)) + 0.5 * float(np.std(onset_env))
    delta = max(delta, float(np.max(onset_env)) * 0.10)  # minimum floor

    # Basit peak picking: lokal maksimum, delta eşiği üstünde, minimum mesafe 3 frame
    min_dist = 3
    peaks = []
    half_win = 5
    for i in range(half_win, len(onset_env) - half_win):
        if onset_env[i] < delta:
            continue
        if onset_env[i] != np.max(onset_env[i - half_win : i + half_win + 1]):
            continue
        if peaks and (i - peaks[-1]) < min_dist:
            continue
        peaks.append(i)

    return float(len(peaks)) / duration if duration > 0 else 0.0


def _extract_pitch_stats(audio: np.ndarray, sr: int) -> Tuple[float, float, int]:
    """
    Pitch istatistikleri: (pitch_mean, pitch_std, jump_count).

    [librosa.pyin → torchaudio.functional.detect_pitch_frequency]
    Sesli frame filtresi: F0 aralığı [50, 400] Hz.
    """
    if audio.size == 0:
        return 0.0, 0.0, 0

    try:
        tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, N)
        pitch = torchaudio.functional.detect_pitch_frequency(
            tensor,
            sample_rate=sr,
            frame_time=0.01,
            win_length=11,
            freq_low=50.0,
            freq_high=400.0,
        ).squeeze(0).cpu().numpy()
    except Exception:
        return 0.0, 0.0, 0

    # Sesli frame: F0 beklenen aralıkta
    voiced_mask = np.isfinite(pitch) & (pitch >= 50.0) & (pitch <= 400.0)
    voiced_f0   = pitch[voiced_mask]

    if voiced_f0.size == 0:
        return 0.0, 0.0, 0

    pitch_mean = float(np.mean(voiced_f0))
    pitch_std  = float(np.std(voiced_f0))

    # Pitch sıçrama sayısı (ardışık sesli framelerde ≥ 50 Hz değişim)
    jump_threshold = 50.0
    jump_count = 0
    prev: Optional[float] = None
    for val, voiced in zip(pitch, voiced_mask):
        if not voiced or not np.isfinite(val):
            prev = None
            continue
        if prev is not None and abs(float(val) - prev) >= jump_threshold:
            jump_count += 1
        prev = float(val)

    return pitch_mean, pitch_std, int(jump_count)


def _extract_physical_signal_states(audio: np.ndarray, sr: int) -> Dict[str, str]:
    """
    Fiziksel ses sinyal state'leri: speech_energy, speech_rate, pitch_stability.

    [librosa.util.normalize → numpy; librosa.feature.rms → torch frame RMS]
    """
    if audio.size == 0:
        return {
            "speech_energy":   "LOW",
            "speech_rate":     "SLOW",
            "pitch_stability": "STABLE",
        }

    # [librosa.util.normalize → numpy]
    max_val    = float(np.max(np.abs(audio)))
    audio_norm = (audio / max_val).astype(np.float32) if max_val > 0 else audio.astype(np.float32)

    # [librosa.feature.rms → torch frame RMS]
    frame_len = 2048
    hop_len   = 512
    tensor    = torch.from_numpy(audio_norm).float()
    if tensor.numel() < frame_len:
        tensor = torch.nn.functional.pad(tensor, (0, frame_len - tensor.numel()))
    frames  = tensor.unfold(0, frame_len, hop_len)
    rms_arr = torch.sqrt(torch.mean(frames ** 2, dim=1) + _EPS).numpy()
    rms_mean = float(np.mean(rms_arr)) if rms_arr.size else 0.0

    _, pitch_std, jump_count = _extract_pitch_stats(audio_norm, sr)
    onsets_per_sec = _estimate_onsets_per_second(audio_norm, sr)

    return {
        "speech_energy":   _energy_state(rms_mean),
        "speech_rate":     _speech_rate_state(onsets_per_sec),
        "pitch_stability": _pitch_stability_state(pitch_std, jump_count),
    }


class AudioSignalFusion:
    """
    Fiziksel ses özellikleri + SER projeksiyonunu birleştirip audio signal timeline üretir.
    """

    def __init__(self, ser_model_id: str = SER_MODEL_ID):
        print(f"[AudioSignalFusion] Başlatılıyor...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ser_projector = HuBERTSerProjector(ser_model_id)
        print(f"[AudioSignalFusion] Hazır!")

    def process_audio(self, audio_path: str) -> List[Dict]:
        if not os.path.exists(audio_path):
            print(f"[AudioSignalFusion] Audio bulunamadı: {audio_path}")
            return []

        # [librosa.load → torchaudio.load + resample]
        waveform, sr_orig = torchaudio.load(audio_path)
        if sr_orig != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr_orig, TARGET_SR)
        sr = TARGET_SR

        # Mono (HuBERT feature_extractor 1-D numpy array bekler)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        y = waveform.squeeze(0).numpy().astype(np.float32)

        # [librosa.get_duration → len / sr]
        duration = float(len(y)) / float(sr)
        if duration <= 0:
            return []

        timeline: List[Dict] = []
        alpha = EMA_ALPHA
        if not (0.0 < alpha <= 1.0):
            alpha = 0.65
        val_ema: Optional[float] = None
        aro_ema: Optional[float] = None
        cursor = 0.0
        while cursor + CHUNK_SEC <= duration:
            s = int(cursor * sr)
            e = int((cursor + CHUNK_SEC) * sr)
            chunk = y[s:e].copy()

            phys = _extract_physical_signal_states(chunk, sr)
            proj = self.ser_projector.project(chunk, sr)

            # SER projeksiyonunu yumuşat (zayıf sinyal dalgalanmasını azaltır)
            if val_ema is None:
                val_ema = float(proj.get("valence_score", 0.0))
            else:
                val_ema = (alpha * float(proj.get("valence_score", 0.0))) + ((1.0 - alpha) * float(val_ema))
            if aro_ema is None:
                aro_ema = float(proj.get("arousal_score", 0.0))
            else:
                aro_ema = (alpha * float(proj.get("arousal_score", 0.0))) + ((1.0 - alpha) * float(aro_ema))

            val_state = _bucketize_valence(float(val_ema))
            aro_state = _bucketize_arousal(float(aro_ema))

            timeline.append(
                {
                    "start": round(cursor, 2),
                    "end":   round(cursor + CHUNK_SEC, 2),
                    "valence_state":   val_state,
                    "arousal_state":   aro_state,
                    "speech_energy":   phys["speech_energy"],
                    "speech_rate":     phys["speech_rate"],
                    "pitch_stability": phys["pitch_stability"],
                    # Debug değerler (rapora yazdırmak zorunlu değil)
                    "debug": {
                        "valence_score_raw": round(proj.get("valence_score", 0.0), 3),
                        "arousal_score_raw": round(proj.get("arousal_score", 0.0), 3),
                        "valence_score_ema": round(float(val_ema), 3),
                        "arousal_score_ema": round(float(aro_ema), 3),
                        "ser_top_label":     proj.get("top_label", ""),
                        "ser_top_score":     round(proj.get("top_score", 0.0), 3),
                    },
                }
            )

            cursor += STEP_SEC

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[AudioSignalFusion] Audio signal tamamlandı: {len(timeline)} parça.")
        return timeline

    @staticmethod
    def get_summary(timeline: List[Dict]) -> Dict:
        if not timeline:
            return {
                "total_chunks":        0,
                "valence_distribution": {},
                "arousal_distribution": {},
                "dominant_valence":    "Veri Yok",
                "dominant_arousal":    "Veri Yok",
            }

        from collections import Counter

        valences  = [t.get("valence_state", "") for t in timeline]
        arousals  = [t.get("arousal_state", "") for t in timeline]
        v_counter = Counter(valences)
        a_counter = Counter(arousals)
        return {
            "total_chunks":        len(timeline),
            "valence_distribution": dict(v_counter),
            "arousal_distribution": dict(a_counter),
            "dominant_valence":    v_counter.most_common(1)[0][0] if v_counter else "Veri Yok",
            "dominant_arousal":    a_counter.most_common(1)[0][0] if a_counter else "Veri Yok",
        }
