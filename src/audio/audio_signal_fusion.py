"""
FAZ-4 Audio Signal Fusion

Amaç:
  - torchaudio f0 + enerji tabanlı kural sistemi → Valence/Arousal sinyali
  - Fiziksel ses özellikleri (torchaudio) → yorumlanabilir discrete state'ler
  - Nihai çıktı: duygu etiketi değil, sinyal state paketleri

Not:
  - SER modeli (wav2vec2) kaldırıldı — Türkçe için güvenilir değildi.
  - Valence/arousal artık f0_std + rms_dbfs kural tabanlı ses profilinden geliyor.
  - Dil bağımsız: sadece ses enerjisi ve pitch varyasyonu kullanılıyor.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio


TARGET_SR = 16000
CHUNK_SEC = 3.0
STEP_SEC = 3.0  # non-overlapping chunks
EMA_ALPHA = float(os.getenv("SENSIFYHR_AUDIO_SIGNAL_EMA_ALPHA", "0.65"))

# Ses profili → valence/arousal mapping
# Gerçek veriden kalibre edildi (sadievrenhocamız: f0_std mean=15.5, rms mean=-18.2 dBFS)
VOICE_PROFILES: Dict[str, Dict[str, float]] = {
    "Canlı":    {"valence": +0.6, "arousal": +0.7},   # yüksek enerji + yüksek varyasyon
    "Kararlı":  {"valence": +0.3, "arousal": +0.5},   # yüksek enerji + orta/düşük varyasyon
    "Gergin":   {"valence": -0.4, "arousal": +0.6},   # orta enerji + yüksek varyasyon
    "Dengeli":  {"valence": +0.1, "arousal": +0.2},   # orta enerji + orta varyasyon
    "Sakin":    {"valence": -0.1, "arousal": -0.2},   # düşük/orta enerji + düşük varyasyon
}

_EPS = 1e-8

# Eşikler — gerçek veri aralıklarından kalibre edildi
_RMS_HIGH_DBFS  = -25.0   # > -25 dBFS → yüksek enerji
_RMS_MED_DBFS   = -40.0   # > -40 dBFS → orta enerji; <= -40 → düşük
_F0STD_HIGH_HZ  = 50.0    # > 50 Hz → yüksek varyasyon
_F0STD_MED_HZ   = 20.0    # > 20 Hz → orta; <= 20 → düşük


def _rms_dbfs(audio: np.ndarray) -> float:
    """RMS değerini dBFS cinsinden hesaplar."""
    rms = float(np.sqrt(np.mean(audio ** 2) + _EPS))
    return 20.0 * np.log10(rms)


def _classify_energy(rms_dbfs_val: float) -> str:
    if rms_dbfs_val > _RMS_HIGH_DBFS:
        return "high"
    if rms_dbfs_val > _RMS_MED_DBFS:
        return "medium"
    return "low"


def _classify_variation(f0_std_val: float) -> str:
    if f0_std_val > _F0STD_HIGH_HZ:
        return "high"
    if f0_std_val > _F0STD_MED_HZ:
        return "medium"
    return "low"


def _get_voice_profile(energy: str, variation: str) -> str:
    """
    Enerji + varyasyon kombinasyonundan ses profili belirler.
    """
    if energy == "high" and variation == "high":
        return "Canlı"
    if energy == "high":
        return "Kararlı"   # yüksek enerji, orta/düşük varyasyon
    if variation == "high":
        return "Gergin"    # orta enerji, yüksek varyasyon
    if variation == "medium":
        return "Dengeli"   # orta enerji, orta varyasyon
    return "Sakin"         # düşük/orta enerji, düşük varyasyon


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
    Fiziksel ses özellikleri (f0 + enerji) tabanlı ses profili → audio signal timeline üretir.
    SER modeli kaldırıldı; dil bağımsız kural sistemi kullanılıyor.
    """

    def __init__(self):
        print(f"[AudioSignalFusion] Başlatılıyor (f0+enerji kural sistemi)...")
        self.device = "cpu"
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

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        y = waveform.squeeze(0).numpy().astype(np.float32)

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

            # Ses profili: f0_std + rms_dbfs kural sistemi
            rms_db = _rms_dbfs(chunk)
            _, f0_std_chunk, _ = _extract_pitch_stats(chunk, sr)
            energy = _classify_energy(rms_db)
            variation = _classify_variation(f0_std_chunk)
            profile = _get_voice_profile(energy, variation)
            valence_raw = VOICE_PROFILES[profile]["valence"]
            arousal_raw = VOICE_PROFILES[profile]["arousal"]

            # EMA yumuşatma (ani sinyal değişimlerini azaltır)
            if val_ema is None:
                val_ema = valence_raw
            else:
                val_ema = (alpha * valence_raw) + ((1.0 - alpha) * val_ema)
            if aro_ema is None:
                aro_ema = arousal_raw
            else:
                aro_ema = (alpha * arousal_raw) + ((1.0 - alpha) * aro_ema)

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
                    "debug": {
                        "voice_profile":     profile,
                        "voice_energy":      energy,
                        "voice_variation":   variation,
                        "rms_dbfs":          round(rms_db, 2),
                        "f0_std":            round(f0_std_chunk, 2),
                        "valence_score_raw": round(valence_raw, 3),
                        "arousal_score_raw": round(arousal_raw, 3),
                        "valence_score_ema": round(float(val_ema), 3),
                        "arousal_score_ema": round(float(aro_ema), 3),
                    },
                }
            )

            cursor += STEP_SEC

        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        """
        
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
