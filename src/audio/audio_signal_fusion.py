"""
FAZ-4 Audio Signal Fusion

Amaç:
  - wav2vec2 SER (ehcalabres, 8 sınıf) -> Valence/Arousal sinyali
  - Fiziksel ses özellikleri (torchaudio) -> yorumlanabilir discrete state'ler
  - Nihai çıktı: emotion label değil, sinyal state paketleri

Not:
  - SER modelinin çıktıları "duygu" olarak raporlanmaz.
  - Sadece valence/arousal projeksiyonu için zayıf sinyal olarak kullanılır.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from transformers import pipeline as hf_pipeline


# FAZ-4 SER modeli (güncel):
# ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
#   — XLSR-53 backbone (53 dil, Türkçe dahil pretraining)
#   — RAVDESS + CREMA-D + TESS + SAVEE veri seti üzerinde fine-tune edilmiş
#   — 8 sınıf: angry, calm, disgust, fearful, happy, neutral, sad, surprised
#   — "calm" sınıfı sayesinde enerjik ama sakin konuşma "angry" değil "calm" etiketleniyor
#   — Önceki model (firdhokk): XLSR-53 tabanlı ama yalnızca 7 sınıf ve
#     enerjik Türkçe konuşmayı %51 "angry" etiketliyordu.
#   — HuggingFace'de en popüler SER modellerinden biri (~1M indirme).
SER_MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Eski model (referans):
# firdhokk/speech-emotion-recognition-with-facebook-wav2vec2-large-xlsr-53

TARGET_SR = 16000
CHUNK_SEC = 3.0
STEP_SEC = 3.0  # non-overlapping chunks (FAZ-4 optimisation: was 1.0)
EMA_ALPHA = float(os.getenv("SENSIFYHR_AUDIO_SIGNAL_EMA_ALPHA", "0.65"))

EMOTION_TO_SIGNAL: Dict[str, Dict[str, float]] = {
    # ehcalabres model labels (8 sınıf: angry/calm/disgust/fearful/happy/neutral/sad/surprised)
    "happy":     {"valence": +1.0, "arousal": +0.6},
    "calm":      {"valence": +0.2, "arousal": -0.3},   # enerjik ama sakin → düşük arousal
    "neutral":   {"valence":  0.0, "arousal":  0.0},
    "sad":       {"valence": -0.8, "arousal": -0.4},
    "angry":     {"valence": -0.9, "arousal": +0.9},
    "fearful":   {"valence": -0.7, "arousal": +0.7},
    "disgust":   {"valence": -0.8, "arousal": +0.3},
    "surprised": {"valence": +0.3, "arousal": +0.8},
    # Compat aliases (diğer model formatları için)
    "fear":      {"valence": -0.7, "arousal": +0.7},
    "hap":       {"valence": +1.0, "arousal": +0.6},
    "ang":       {"valence": -0.9, "arousal": +0.9},
    "neu":       {"valence":  0.0, "arousal":  0.0},
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


_RMS_SILENCE_THRESHOLD = 10 ** (-45.0 / 20)  # -45 dBFS ≈ 0.00562

_LABEL_MAP_RAW = {
    # ehcalabres model labels (8 sınıf) — identity mapping
    "angry":     "angry",
    "calm":      "calm",
    "disgust":   "disgust",
    "fearful":   "fearful",
    "happy":     "happy",
    "neutral":   "neutral",
    "sad":       "sad",
    "surprised": "surprised",
    # Compat aliases (diğer model formatları)
    "sadness":   "sad",
    "fear":      "fearful",
    "hap":       "happy",
    "ang":       "angry",
    "neu":       "neutral",
}
# Case-insensitive: model may output "Happy", "Angry" etc.
_LABEL_MAP = {k.lower(): v for k, v in _LABEL_MAP_RAW.items()}

_SILENT_RESULT: Dict[str, Any] = {
    "valence_score": 0.0,
    "arousal_score": 0.0,
    "valence_state": "NEUTRAL",
    "arousal_state": "LOW",
    "top_label":     "neutral",
    "top_score":     0.0,
    "all_scores":    {},
}


class SerProjector:
    """
    wav2vec2 tabanlı SER — transformers pipeline() ile basit implementasyon.
    Çıktıları valence/arousal sinyaline projekte eder.
    Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (8 sınıf)
    """

    def __init__(self, model_id: str = SER_MODEL_ID):
        print(f"[SER] Başlatılıyor... Model: {model_id}")
        self.model_id = model_id
        device = 0 if torch.cuda.is_available() else -1
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not used when initializing.*")
            warnings.filterwarnings("ignore", message=".*were not initialized.*")
            warnings.filterwarnings("ignore", message=".*should probably TRAIN.*")
            warnings.filterwarnings("ignore", message=".*gradient_checkpointing.*")
            self._pipeline = hf_pipeline(
                "audio-classification",
                model=model_id,
                device=device,
            )
        print(f"[SER] Hazır! Cihaz: {'CUDA' if device == 0 else 'CPU'}")

    def project(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        # Sessiz chunk filtresi: -45 dBFS altı veya boş
        if audio.size == 0:
            return dict(_SILENT_RESULT)
        rms_val = float(np.sqrt(np.mean(audio ** 2)))
        if rms_val < _RMS_SILENCE_THRESHOLD:
            return dict(_SILENT_RESULT)

        # pipeline() float32 numpy array (shape: N,) kabul eder, sr=16000 varsayılan
        result = self._pipeline(audio)
        # result = [{'label': 'sadness', 'score': 0.85}, ...]

        top       = result[0]
        top_label = _LABEL_MAP.get(top["label"].lower(), top["label"].lower())
        top_score = round(float(top["score"]), 3)
        all_scores = {
            _LABEL_MAP.get(r["label"].lower(), r["label"].lower()): round(float(r["score"]), 3)
            for r in result
        }

        # Valence/arousal projeksiyonu (ağırlıklı toplam)
        valence = 0.0
        arousal = 0.0
        for r in result:
            lab = _LABEL_MAP.get(r["label"].lower(), r["label"].lower())
            sig = EMOTION_TO_SIGNAL.get(lab, EMOTION_TO_SIGNAL["neutral"])
            valence += float(r["score"]) * float(sig["valence"])
            arousal += float(r["score"]) * float(sig["arousal"])

        return {
            "valence_score": float(valence),
            "arousal_score": float(arousal),
            "valence_state": _bucketize_valence(float(valence)),
            "arousal_state": _bucketize_arousal(float(arousal)),
            "top_label":     top_label,
            "top_score":     top_score,
            "all_scores":    all_scores,
        }



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
        self.ser_projector = SerProjector(ser_model_id)
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
                        "ser_all_scores":    proj.get("all_scores", {}),
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
