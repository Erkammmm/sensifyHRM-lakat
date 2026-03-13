"""
Ses Analizi Modülü (FAZ-4)

torchaudio tabanlı, per-saniye ses özelliği çıkarımı.
Referans: stajyer_kodlari/voice_test_said.py

Değişiklikler:
  - tkinter kaldırıldı; wav_path parametre olarak alınır
  - Sınıf tabanlı arayüz: VoiceAnalyzer.analyze_audio(wav_path)
  - Çıktı: per-second timeline listesi (LLM / contextual aggregator için)
  - pandas kaldırıldı; saf Python / numpy / torch
  - Dosyaya yazma yok; veri doğrudan döndürülür
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio

# ── Analiz sabitleri ──────────────────────────────────────────────────────────
TARGET_SR       = 16000
SEGMENT_SEC     = 1.0        # her segment 1 saniye

FFT_SIZE        = 1024       # STFT pencere boyutu
HOP_SIZE        = 256        # STFT atlama boyutu
MEL_BANDS       = 64         # Mel filtre sayısı

FRAME_MS        = 30         # kısa-term RMS/ZCR frame süresi (ms)
FRAME_HOP_MS    = 10         # kısa-term frame atlama (ms)

PITCH_FRAME_TIME = 0.01      # F0 tespiti frame süresi (sn)
PITCH_WIN_LEN    = 11        # F0 tespiti pencere uzunluğu
F0_MIN          = 70.0       # Hz
F0_MAX          = 420.0      # Hz

MIN_SILENCE_ACTIVE_RATIO = 0.06
MIN_REAL_SPEECH_RATIO    = 0.28
MIN_VOICED_RATIO         = 0.12
SPEECH_CONF_THRESHOLD    = 0.58

EPS              = 1e-8
ABS_SILENCE_RMS  = 1e-4
ABS_SILENCE_PEAK = 5e-4


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def _rms_to_dbfs(rms: float) -> float:
    """RMS genliği dBFS'e çevirir."""
    return 20.0 * math.log10(max(float(rms), EPS))


def _linear_score(value: float, low: float, high: float) -> float:
    """Değeri [low, high] aralığında [0, 1]'e doğrusal normalize eder."""
    if not np.isfinite(value) or high <= low:
        return 0.0
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _smooth_mask(mask: np.ndarray, window: int = 5) -> np.ndarray:
    """Tek tük kopuk frame'leri convolution ile toparlayan bool maske yumuşatması."""
    if mask.size == 0:
        return mask.astype(bool)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    counts = np.convolve(mask.astype(np.float32), np.ones(window, dtype=np.float32), mode="same")
    return counts >= int(math.ceil(window / 2.0))


def _frame_rms(mono: torch.Tensor, frame_len: int, hop_len: int) -> np.ndarray:
    """
    1-D mono tensor → frame-level RMS dizisi.

    Formül: RMS_t = sqrt( (1/N) * sum_n x_t[n]^2 )
    """
    x = mono.squeeze(0).cpu()
    if x.numel() < frame_len:
        x = torch.nn.functional.pad(x, (0, frame_len - x.numel()))
    frames = x.unfold(0, frame_len, hop_len)             # (T, frame_len)
    return torch.sqrt(torch.mean(frames ** 2, dim=1) + EPS).numpy()


def _frame_zcr(mono: torch.Tensor, frame_len: int, hop_len: int) -> np.ndarray:
    """
    1-D mono tensor → frame-level ZCR dizisi.

    Formül: ZCR_t = (1/(N-1)) * sum_{n=1}^{N-1} 1[x_t[n]*x_t[n-1] < 0]
    """
    x = mono.squeeze(0).cpu()
    if x.numel() < frame_len:
        x = torch.nn.functional.pad(x, (0, frame_len - x.numel()))
    frames = x.unfold(0, frame_len, hop_len)
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float32)
    return ((frames[:, 1:] * frames[:, :-1]) < 0).float().mean(dim=1).numpy()


def _segment_zcr(segment: torch.Tensor) -> float:
    """1 saniyelik segment için tek ZCR değeri."""
    x = segment.squeeze(0)
    if x.numel() < 2:
        return 0.0
    return float(((x[1:] * x[:-1]) < 0).float().mean().item())


def _spectral_features(
    segment: torch.Tensor,
    sr: int,
    spec_transform: torchaudio.transforms.Spectrogram,
    mel_transform: torchaudio.transforms.MelSpectrogram,
) -> Tuple[float, float, float]:
    """
    (spectral_centroid, spectral_flatness, mel_energy) döndürür.

    Centroid:  C_t = sum_k f_k * S_t[k] / sum_k S_t[k]
    Flatness:  F_t = geom_mean(S_t[k]) / arith_mean(S_t[k])
    Mel energy: ortalama mel-band enerjisi
    """
    psd = spec_transform(segment.cpu()).squeeze(0)         # (freq, T)
    total = float(psd.sum().item())
    if total <= EPS:
        return float("nan"), float("nan"), 0.0

    freqs = torch.linspace(0, sr / 2, psd.shape[0], dtype=psd.dtype).unsqueeze(1)
    frame_e = psd.sum(dim=0, keepdim=True).clamp_min(EPS)
    centroid = ((freqs * psd).sum(dim=0, keepdim=True) / frame_e).mean().item()

    safe_psd = psd.clamp_min(EPS)
    geo = torch.exp(torch.log(safe_psd).mean(dim=0))
    ari = safe_psd.mean(dim=0)
    flatness = (geo / ari.clamp_min(EPS)).mean().item()

    mel_energy = float(mel_transform(segment.cpu()).squeeze(0).sum(dim=0).mean().item())
    return float(centroid), float(flatness), float(mel_energy)


def _global_pitch_trace(mono: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Tüm kayıt için frame-level F0 izi.

    f0 = fs / T0  →  torchaudio.functional.detect_pitch_frequency
    """
    try:
        return torchaudio.functional.detect_pitch_frequency(
            mono.cpu(),
            sample_rate=sr,
            frame_time=PITCH_FRAME_TIME,
            win_length=PITCH_WIN_LEN,
            freq_low=F0_MIN,
            freq_high=F0_MAX,
        ).squeeze(0).cpu()
    except Exception as e:
        print(f"[VoiceAnalyzer] Pitch hesaplanamadı: {e}")
        return torch.empty(0, dtype=torch.float32)


def _segment_pitch_stats(
    pitch_trace: torch.Tensor,
    rms_frames: np.ndarray,
    zcr_frames: np.ndarray,
    start_sec: float,
    end_sec: float,
    silence_thresh: float,
) -> Tuple[float, float, float]:
    """
    Bir saniye için (f0_mean, f0_std, voiced_ratio).

    Sesli frame: geçerli F0 AND RMS yeterli AND ZCR düşük
    """
    n = min(pitch_trace.numel(), len(rms_frames), len(zcr_frames))
    if n == 0:
        return float("nan"), float("nan"), 0.0

    i_s = max(0, min(int(start_sec / PITCH_FRAME_TIME), n))
    i_e = max(i_s, min(int(math.ceil(end_sec / PITCH_FRAME_TIME)), n))
    if i_e <= i_s:
        return float("nan"), float("nan"), 0.0

    p = pitch_trace[i_s:i_e].numpy()
    r = rms_frames[i_s:i_e]
    z = zcr_frames[i_s:i_e]
    n2 = min(len(p), len(r), len(z))
    p, r, z = p[:n2], r[:n2], z[:n2]

    voiced = (
        np.isfinite(p) & (p >= F0_MIN) & (p <= F0_MAX)
        & (r >= max(silence_thresh * 1.4, ABS_SILENCE_RMS))
        & (z <= 0.18)
    )
    voiced_ratio = float(np.mean(voiced)) if len(voiced) else 0.0
    valid = p[voiced]

    if valid.size == 0:
        return float("nan"), float("nan"), voiced_ratio
    return float(np.mean(valid)), float(np.std(valid)), voiced_ratio


def _background_score(flatness: float, zcr: float, centroid: float, rms_dbfs: float) -> float:
    """
    0–1 arka plan gürültü skoru.

    bg = 0.30 * score(flatness) + 0.25 * score(zcr)
       + 0.25 * score(centroid) + 0.20 * score(-rms_dbfs)
    """
    return float(np.clip(
        0.30 * _linear_score(flatness,   0.03,  0.11)
        + 0.25 * _linear_score(zcr,      0.07,  0.18)
        + 0.25 * _linear_score(centroid, 900.0, 2600.0)
        + 0.20 * _linear_score(-rms_dbfs, 36.0,  54.0),
        0.0, 1.0,
    ))


def _speech_confidence_score(
    speech_ratio: float,
    voiced_ratio: float,
    flatness: float,
    centroid: float,
    rms_dbfs: float,
    bg_score: float,
) -> float:
    """
    0–1 konuşma güven skoru (gerçek konuşma olasılığı).

    conf = 0.28*f(speech_ratio) + 0.24*f(voiced) + 0.14*(1-f(flatness))
         + 0.10*band_check       + 0.14*f(level)  + 0.10*(1-bg)
    """
    band_ok = 1.0 if (np.isfinite(centroid) and 120.0 <= centroid <= 2400.0) else 0.0
    return float(np.clip(
        0.28 * _linear_score(speech_ratio, 0.12, 0.65)
        + 0.24 * _linear_score(voiced_ratio, 0.08, 0.50)
        + 0.14 * (1.0 - _linear_score(flatness, 0.05, 0.16))
        + 0.10 * band_ok
        + 0.14 * _linear_score(rms_dbfs, -42.0, -22.0)
        + 0.10 * (1.0 - bg_score),
        0.0, 1.0,
    ))


def _speech_energy_label(rms_dbfs: float, centroid: float, f0_std: float) -> Tuple[float, str]:
    """
    (energy_score, label) — label ∈ "düşük" | "orta" | "yüksek".

    score = 0.55*f(rms) + 0.20*f(centroid) + 0.25*f(f0_std)
    """
    score = float(np.clip(
        0.55 * _linear_score(rms_dbfs, -38.0, -20.0)
        + 0.20 * _linear_score(centroid, 250.0, 1200.0)
        + 0.25 * _linear_score(f0_std, 6.0, 28.0),
        0.0, 1.0,
    ))
    if not np.isfinite(score):
        return 0.0, "belirsiz"
    label = "yüksek" if score > 0.67 else ("orta" if score >= 0.33 else "düşük")
    return score, label


def _speech_style_label(
    energy_score: float, bg_score: float, f0_std: float, voiced_ratio: float
) -> str:
    """
    Akustik stil etiketi: sakin | dengeli | canlı | karışık | belirsiz.

    sakin   = 0.55*(1-e) + 0.30*clean + 0.15*(1-tone_movement)
    dengeli = 0.50*(orta_e) + 0.30*clean + 0.20*(orta_tone)
    canlı   = 0.55*e + 0.25*tone_movement + 0.20*clean
    """
    if voiced_ratio < 0.10:
        return "belirsiz"
    tone = _linear_score(f0_std, 8.0, 24.0)
    clean = 1.0 - bg_score
    sakin   = 0.55 * (1.0 - energy_score) + 0.30 * clean + 0.15 * (1.0 - tone)
    dengeli = (
        0.50 * (1.0 - abs(energy_score - 0.5) / 0.5)
        + 0.30 * clean
        + 0.20 * (1.0 - abs(tone - 0.5))
    )
    canli   = 0.55 * energy_score + 0.25 * tone + 0.20 * clean
    scores  = {"sakin": sakin, "dengeli": dengeli, "canlı": canli}
    label, best = max(scores.items(), key=lambda x: x[1])
    return label if best >= 0.52 else "karışık"


def _safe_f(v) -> float:
    """nan / inf → 0.0 dönüşümü."""
    if v is None:
        return 0.0
    try:
        f = float(v)
        return 0.0 if not math.isfinite(f) else f
    except Exception:
        return 0.0


# ── VoiceAnalyzer sınıfı ─────────────────────────────────────────────────────

class VoiceAnalyzer:
    """
    torchaudio tabanlı per-saniye ses analizi.

    analyze_audio(wav_path) → {
        "per_second_timeline": [
            {
                "start_sec"        : int,
                "end_sec"          : float,
                "segment_type"     : "konuşma" | "sessiz" | "konuşma_dışı",
                "is_speech"        : bool,
                "rms_dbfs"         : float,
                "f0_mean"          : float,
                "f0_std"           : float,
                "spectral_centroid": float,
                "spectral_flatness": float,
                "mel_energy"       : float,
                "konusma_guveni"   : float,   # 0–1
                "konusma_stili"    : str,     # sakin|dengeli|canlı|karışık|belirsiz
                "konusma_enerjisi" : str,     # düşük|orta|yüksek|belirsiz
            }, ...
        ],
        "raw_voice_features": {}   # geriye dönük uyumluluk (raporlama modülü)
    }
    """

    def __init__(self):
        print(f"[{self.__class__.__name__}] Hazır!")

    def analyze_audio(self, wav_path: str) -> Dict:
        """
        WAV dosyasını per-saniye analiz eder.

        Args:
            wav_path: Pipeline tarafından çıkarılmış 16kHz mono WAV yolu.

        Returns:
            {"per_second_timeline": List[Dict], "raw_voice_features": {}}
        """
        if not os.path.exists(wav_path):
            print(f"[{self.__class__.__name__}] Ses dosyası bulunamadı: {wav_path}")
            return {"per_second_timeline": [], "raw_voice_features": {}}

        # ── 1) Yükleme ve yeniden örnekleme ──────────────────────────────────
        waveform, sr = torchaudio.load(wav_path)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            sr = TARGET_SR

        if float(waveform.abs().max().item()) < 1e-7:
            print(f"[{self.__class__.__name__}] Uyarı: ses neredeyse tamamen sessiz.")
            return {"per_second_timeline": [], "raw_voice_features": {}}

        # ── 2) Mono — en yüksek RMS kanalını seç ────────────────────────────
        if waveform.shape[0] > 1:
            ch_rms = torch.sqrt(torch.mean(waveform ** 2, dim=1) + EPS)
            best_ch = int(torch.argmax(ch_rms).item())
            mono = waveform[best_ch : best_ch + 1].contiguous()
        else:
            mono = waveform

        total_sec = mono.shape[1] / sr
        n_segments = int(math.ceil(total_sec / SEGMENT_SEC))

        print(
            f"[{self.__class__.__name__}] Analiz başladı: "
            f"{total_sec:.1f} sn, {n_segments} segment, SR={sr}"
        )

        # ── 3) Global frame özellikleri ──────────────────────────────────────
        frame_len = int(sr * FRAME_MS / 1000)
        hop_len   = int(sr * FRAME_HOP_MS / 1000)

        rms_frames = _frame_rms(mono, frame_len, hop_len)
        zcr_frames = _frame_zcr(mono, frame_len, hop_len)

        nonzero = rms_frames[rms_frames > 1e-8]
        if nonzero.size == 0:
            print(f"[{self.__class__.__name__}] Tüm frame RMS sıfır; ses analiz edilemiyor.")
            return {"per_second_timeline": [], "raw_voice_features": {}}

        # Adaptif eşikler
        # silence_thresh = max(abs_min, 0.5 * p15(RMS))
        # speech_thresh  = max(2.2 * silence_thresh, p40(RMS))
        silence_thresh = max(ABS_SILENCE_RMS, float(np.percentile(nonzero, 15)) * 0.5)
        speech_thresh  = max(silence_thresh * 2.2, float(np.percentile(nonzero, 40)))

        active_mask = rms_frames >= silence_thresh

        # ── 4) Pitch izi ─────────────────────────────────────────────────────
        pitch_trace = _global_pitch_trace(mono, sr)

        # Tüm dizileri ortak uzunluğa hizala
        if pitch_trace.numel() > 0:
            n_common = min(len(rms_frames), len(zcr_frames), pitch_trace.numel())
            pitch_np = pitch_trace.numpy()[:n_common]
        else:
            n_common = min(len(rms_frames), len(zcr_frames))
            pitch_np = np.full(n_common, float("nan"), dtype=np.float32)

        rms_frames  = rms_frames[:n_common]
        zcr_frames  = zcr_frames[:n_common]
        active_mask = active_mask[:n_common]

        # Sesli (voiced) ve konuşma-benzeri (speech-like) frame maskeleri
        raw_voiced = (
            np.isfinite(pitch_np) & (pitch_np >= F0_MIN) & (pitch_np <= F0_MAX)
            & (rms_frames >= max(silence_thresh * 1.4, ABS_SILENCE_RMS))
            & (zcr_frames <= 0.18)
        )
        raw_speech = (
            (rms_frames >= speech_thresh)
            & (zcr_frames >= 0.01) & (zcr_frames <= 0.16)
            & (raw_voiced | ((rms_frames >= speech_thresh * 1.25) & (zcr_frames <= 0.10)))
        )

        voiced_mask = _smooth_mask(raw_voiced, 3) & active_mask
        speech_mask = _smooth_mask(raw_speech, 3) & active_mask

        pitch_tensor = torch.from_numpy(pitch_np.copy())

        # ── 5) Spektral dönüşüm objeleri (bir kez oluştur) ───────────────────
        spec_tf = torchaudio.transforms.Spectrogram(
            n_fft=FFT_SIZE, hop_length=HOP_SIZE, power=2.0
        )
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            n_mels=MEL_BANDS, power=2.0,
        )

        frame_hop_sec = hop_len / sr

        # ── 6) Per-saniye analiz döngüsü ─────────────────────────────────────
        timeline: List[Dict] = []

        for i in range(n_segments):
            s_sample = int(i * SEGMENT_SEC * sr)
            e_sample = min(int((i + 1) * SEGMENT_SEC * sr), mono.shape[1])
            segment  = mono[:, s_sample:e_sample]

            # Çok kısa son segment (< 250ms) atlanır
            if segment.shape[1] < sr // 4:
                continue

            s_sec = s_sample / sr
            e_sec = e_sample / sr

            # Anlık seviye
            seg_rms      = float(torch.sqrt(torch.mean(segment ** 2) + EPS).item())
            seg_rms_dbfs = _rms_to_dbfs(seg_rms)
            seg_peak     = float(torch.max(torch.abs(segment)).item())
            seg_zcr      = _segment_zcr(segment)

            # Frame indeksleri
            f_s = max(0, min(int(round(s_sec / frame_hop_sec)), n_common))
            f_e = max(f_s, min(int(math.ceil(e_sec / frame_hop_sec)), n_common))

            if f_e > f_s:
                active_ratio = float(np.mean(active_mask[f_s:f_e]))
                speech_ratio = float(np.mean(speech_mask[f_s:f_e]))
                voiced_ratio = float(np.mean(voiced_mask[f_s:f_e]))
            else:
                active_ratio = speech_ratio = voiced_ratio = 0.0

            # Tam sessizlik kararı
            is_silence = bool(
                active_ratio < MIN_SILENCE_ACTIVE_RATIO
                and seg_rms < max(silence_thresh * 1.5, ABS_SILENCE_RMS * 2.0)
                and seg_peak < max(ABS_SILENCE_PEAK * 2.0, silence_thresh * 6.0)
            )

            if is_silence:
                segment_type  = "sessiz"
                centroid = flatness = float("nan")
                mel_energy    = 0.0
                f0_mean = f0_std = float("nan")
                speech_conf   = 0.0
                energy_label  = "belirsiz"
                style_label   = "sessiz"

            else:
                centroid, flatness, mel_energy = _spectral_features(
                    segment, sr, spec_tf, mel_tf
                )
                f0_mean, f0_std, _ = _segment_pitch_stats(
                    pitch_tensor, rms_frames, zcr_frames, s_sec, e_sec, silence_thresh
                )

                bg_score   = _background_score(flatness, seg_zcr, centroid, seg_rms_dbfs)
                speech_conf = _speech_confidence_score(
                    speech_ratio, voiced_ratio, flatness, centroid, seg_rms_dbfs, bg_score
                )

                # Gerçek konuşma kararı
                # speech if (ratio≥0.28 AND voiced≥0.12 AND conf≥0.58)
                #         OR (ratio≥0.40 AND conf≥0.52)
                is_speech_seg = bool(
                    (
                        speech_ratio >= MIN_REAL_SPEECH_RATIO
                        and voiced_ratio >= MIN_VOICED_RATIO
                        and speech_conf >= SPEECH_CONF_THRESHOLD
                    )
                    or (speech_ratio >= 0.40 and speech_conf >= 0.52)
                )
                segment_type = "konuşma" if is_speech_seg else "konuşma_dışı"

                if is_speech_seg:
                    f0_std_safe = f0_std if np.isfinite(f0_std) else 0.0
                    energy_score, energy_label = _speech_energy_label(
                        seg_rms_dbfs, centroid, f0_std_safe
                    )
                    style_label = _speech_style_label(
                        energy_score, bg_score, f0_std_safe, voiced_ratio
                    )
                else:
                    energy_label = "belirsiz"
                    style_label  = "konuşma_dışı"

            timeline.append(
                {
                    "start_sec":         i,
                    "end_sec":           round(float(e_sec), 3),
                    "segment_type":      segment_type,
                    "is_speech":         segment_type == "konuşma",
                    "rms_dbfs":          round(seg_rms_dbfs, 2),
                    "f0_mean":           round(_safe_f(f0_mean), 2),
                    "f0_std":            round(_safe_f(f0_std), 2),
                    "spectral_centroid": round(_safe_f(centroid), 2),
                    "spectral_flatness": round(_safe_f(flatness), 4),
                    "mel_energy":        round(_safe_f(mel_energy), 4),
                    "konusma_guveni":    round(float(speech_conf), 3),
                    "konusma_stili":     style_label,
                    "konusma_enerjisi":  energy_label,
                }
            )

        print(
            f"[{self.__class__.__name__}] Ses analizi tamamlandı: "
            f"{len(timeline)} saniye segmenti."
        )
        return {
            "per_second_timeline": timeline,
            # raw_voice_features boş döner; raporlama modülü graceful handle eder
            "raw_voice_features": {},
        }


if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    print("VoiceAnalyzer modülü hazır.")
    print("Kullanım: analyzer.analyze_audio('audio.wav')")
