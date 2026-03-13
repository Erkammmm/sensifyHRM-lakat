"""
FAZ-4 Contextual Aggregator

Her Whisper/ThoughtUnit segmenti için üç sinyal kaynağını zaman bazlı hizalar:
  1. face_timeline      — FaceAnalyzer (UniFace): emotion_label, gaze_pitch_deg, gaze_yaw_deg
  2. voice_timeline     — VoiceAnalyzer (torchaudio): konusma_stili, konusma_guveni, f0_mean, rms_dbfs
  3. audio_signal_timeline — AudioSignalFusion (HuBERT): valence/arousal (numeric, debug dict)

Çıktı: LLM'e gidecek segment_signal_packages listesi (flat dict).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Küçük yardımcılar
# ---------------------------------------------------------------------------

def _mode(values: List[str], default: str) -> str:
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


def _mean(values: list, default: float = 0.0) -> float:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return default
    return sum(nums) / len(nums)


def _format_mmss(seconds: float) -> str:
    try:
        s = max(0.0, float(seconds))
    except Exception:
        s = 0.0
    m = int(s // 60)
    sec = int(round(s - (m * 60)))
    if sec >= 60:
        m += 1
        sec = 0
    return f"{m:02d}:{sec:02d}"


_CONFIDENCE_THRESHOLD = 0.55   # frame-level: below this → treat as Neutral
_FREQ_THRESHOLD       = 0.80   # segment-level: emotion dominates >80% of frames…
_FREQ_CONF_THRESHOLD  = 0.65   # …but avg confidence < this → demote to Neutral


def _filtered_emotion_labels(frames: List[Dict]) -> List[str]:
    """
    Per-frame confidence filter: if emotion_confidence < 0.55 the frame is
    reclassified as "Neutral" before any aggregation is done.
    """
    out: List[str] = []
    for f in frames:
        label = f.get("emotion_label") or "Neutral"
        conf  = float(f.get("emotion_confidence") or 0.0)
        out.append(label if conf >= _CONFIDENCE_THRESHOLD else "Neutral")
    return out


def _dominant_with_freq_filter(labels: List[str], frames: List[Dict]) -> str:
    """
    Compute dominant emotion from filtered labels, then apply frequency filter:
    if the dominant emotion covers >80% of frames but its average raw confidence
    is below 0.65, demote it to "Neutral".
    """
    if not labels:
        return "Neutral"
    counter = Counter(labels)
    dominant, count = counter.most_common(1)[0]
    freq = count / len(labels)
    if freq > _FREQ_THRESHOLD and dominant != "Neutral":
        dom_confidences = [
            float(f.get("emotion_confidence") or 0.0)
            for f, lab in zip(frames, labels)
            if lab == dominant
        ]
        avg_conf = sum(dom_confidences) / len(dom_confidences) if dom_confidences else 0.0
        if avg_conf < _FREQ_CONF_THRESHOLD:
            return "Neutral"
    return dominant


_GAZE_PITCH_THRESHOLD = 8.0   # normalized pitch (after offset removal)
_GAZE_YAW_THRESHOLD   = 12.0  # raw yaw


def _gaze_direction(pitch_deg: float, yaw_deg: float) -> str:
    """
    Pitch is expected to be already normalized (offset-subtracted).
    Thresholds: |pitch| > 8° → up/down, |yaw| > 12° → right/left.
    Pitch is priority: if both thresholds exceeded, pitch wins.
    """
    if abs(pitch_deg) > _GAZE_PITCH_THRESHOLD:
        return "up" if pitch_deg > 0 else "down"
    if abs(yaw_deg) > _GAZE_YAW_THRESHOLD:
        return "right" if yaw_deg > 0 else "left"
    return "center"


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def build_segment_signal_packages(
    text_segments: List[Dict],
    audio_signal_timeline: List[Dict],
    # FAZ-3 compat: eski parametre adı visual_signal_timeline, yeni adı face_timeline
    visual_signal_timeline: Optional[List[Dict]] = None,
    face_timeline: Optional[List[Dict]] = None,
    voice_timeline: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Args:
      text_segments           : STT / ThoughtUnit segmentleri [{start, end, text}, ...]
      audio_signal_timeline   : AudioSignalFusion çıktısı [{start, end, valence_state,
                                 arousal_state, ..., debug:{valence_score_ema,...}}, ...]
      visual_signal_timeline  : (FAZ-3 compat) FaceAnalyzer eski format — ileride kaldırılacak
      face_timeline           : FaceAnalyzer FAZ-4 format [{timestamp_sec, emotion_label,
                                 emotion_confidence, gaze_pitch_deg, gaze_yaw_deg, face_detected}, ...]
      voice_timeline          : VoiceAnalyzer FAZ-4 format [{start_sec, end_sec, konusma_stili,
                                 konusma_guveni, f0_mean, rms_dbfs, ...}, ...]

    Returns:
      LLM segment_signal_packages listesi.
    """
    # face_timeline yoksa eski ismi dene (pipeline.py henüz güncellenmemişse)
    effective_face = face_timeline if face_timeline is not None else (visual_signal_timeline or [])
    effective_voice = voice_timeline or []
    effective_audio = audio_signal_timeline or []

    # Per-video pitch offset: MobileGaze produces a systematic positive pitch bias
    # due to camera placement. Subtract the video-wide mean before thresholding.
    _all_pitches = [
        float(f.get("gaze_pitch_deg") or 0.0)
        for f in effective_face
        if f.get("face_detected", True) and f.get("gaze_pitch_deg") is not None
    ]
    _pitch_center = _mean(_all_pitches, 0.0)

    packages: List[Dict] = []
    if not text_segments:
        return packages

    for seg_id, seg in enumerate(text_segments):
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end   = float(seg.get("end", seg_start) or seg_start)
        text      = (seg.get("text") or "").strip()

        # ------------------------------------------------------------------
        # 1. Yüz / Duygu / Bakış — FaceAnalyzer FAZ-4
        # ------------------------------------------------------------------
        face_in_range = [
            f for f in effective_face
            if f.get("face_detected", True)  # face_detected=False kayıtları atla
            and seg_start <= float(f.get("timestamp_sec", f.get("timestamp", 0.0)) or 0.0) <= seg_end
        ]

        if face_in_range:
            # Step B filtering: per-frame confidence filter, then frequency+confidence filter
            filtered_labels    = _filtered_emotion_labels(face_in_range)
            dominant_emotion   = _dominant_with_freq_filter(filtered_labels, face_in_range)
            emotion_confidence = round(_mean([f.get("emotion_confidence") for f in face_in_range], 0.0), 3)
            avg_gaze_pitch     = round(_mean([f.get("gaze_pitch_deg") for f in face_in_range], 0.0), 2)
            avg_gaze_yaw       = round(_mean([f.get("gaze_yaw_deg") for f in face_in_range], 0.0), 2)
        else:
            dominant_emotion   = "Neutral"
            emotion_confidence = 0.0
            avg_gaze_pitch     = 0.0
            avg_gaze_yaw       = 0.0

        # Per-frame gaze direction with frequency-based segment label.
        # Pitch normalized by video-wide offset before thresholding.
        # If >25% of frames are non-center → use most common non-center direction.
        if face_in_range:
            per_frame_dirs = [
                _gaze_direction(
                    float(f.get("gaze_pitch_deg") or 0.0) - _pitch_center,
                    float(f.get("gaze_yaw_deg") or 0.0),
                )
                for f in face_in_range
            ]
            dir_counts = Counter(per_frame_dirs)
            n_total = len(per_frame_dirs)
            n_noncenter = n_total - dir_counts.get("center", 0)
            if n_noncenter / n_total > 0.25:
                # Majority or near-majority non-center: pick most common non-center direction
                noncenter_counts = {d: c for d, c in dir_counts.items() if d != "center"}
                gaze_dir = max(noncenter_counts, key=noncenter_counts.get)
            else:
                gaze_dir = "center"
        else:
            gaze_dir = "center"

        # ------------------------------------------------------------------
        # 2. Konuşma özellikleri — VoiceAnalyzer FAZ-4
        # ------------------------------------------------------------------
        voice_in_range = [
            v for v in effective_voice
            if float(v.get("start_sec", v.get("start", 0.0)) or 0.0) < seg_end
            and float(v.get("end_sec",   v.get("end",   0.0)) or 0.0) > seg_start
        ]

        if voice_in_range:
            speech_style      = _mode([v.get("konusma_stili", "") for v in voice_in_range], "sakin")
            speech_confidence = round(_mean([v.get("konusma_guveni") for v in voice_in_range], 0.0), 3)
            f0_mean_val       = round(_mean([v.get("f0_mean") for v in voice_in_range], 0.0), 2)
            f0_std_val        = round(_mean([v.get("f0_std")  for v in voice_in_range], 0.0), 2)
            rms_dbfs_val      = round(_mean([v.get("rms_dbfs") for v in voice_in_range], -60.0), 2)
        else:
            speech_style      = "sakin"
            speech_confidence = 0.0
            f0_mean_val       = 0.0
            f0_std_val        = 0.0
            rms_dbfs_val      = -60.0

        # ------------------------------------------------------------------
        # 3. HuBERT valence / arousal — AudioSignalFusion
        # ------------------------------------------------------------------
        audio_in_range = [
            a for a in effective_audio
            if float(a.get("start", 0.0) or 0.0) < seg_end
            and float(a.get("end",   0.0) or 0.0) > seg_start
        ]

        if audio_in_range:
            hubert_valence = round(
                _mean([a.get("debug", {}).get("valence_score_ema", a.get("valence_score", 0.0))
                       for a in audio_in_range], 0.0), 3
            )
            hubert_arousal = round(
                _mean([a.get("debug", {}).get("arousal_score_ema", a.get("arousal_score", 0.0))
                       for a in audio_in_range], 0.0), 3
            )
        else:
            hubert_valence = 0.0
            hubert_arousal = 0.0

        # ------------------------------------------------------------------
        # Türetilmiş davranışsal göstergeler
        # ------------------------------------------------------------------
        # gaze_away: bakış merkezden uzakta mı?
        gaze_away = gaze_dir != "center"

        # voice_stress: yüksek pitch varyasyonu + düşük konuşma güveni
        voice_stress = f0_std_val > 30.0 and speech_confidence < 0.6

        # incongruence: valence ile yüz duygusu çelişiyor mu?
        # (yalnızca emotion_confidence >= 0.5 ise güvenilir)
        _stress_emotions = {"Fear", "Angry", "Disgust", "Sad"}
        _positive_emotions = {"Happy", "Surprise"}
        if emotion_confidence >= 0.5:
            if hubert_valence > 0.0 and dominant_emotion in _stress_emotions:
                incongruence = True   # pozitif valence + negatif yüz
            elif hubert_valence < 0.0 and dominant_emotion in _positive_emotions:
                incongruence = True   # negatif valence + pozitif yüz
            else:
                incongruence = False
        else:
            incongruence = False  # düşük güven → yorum yapma

        # tension_score: 0-1 ağırlıklı bileşik stres sinyali
        _emotion_stress = 1.0 if dominant_emotion in _stress_emotions else 0.0
        _speech_inv     = 1.0 - speech_confidence              # 0-1
        _f0_norm        = min(f0_std_val / 50.0, 1.0)          # 0-1 (50 Hz üstü tam stres)
        _gaze_num       = 1.0 if gaze_away else 0.0
        tension_score   = round(
            _emotion_stress * 0.4
            + _speech_inv   * 0.3
            + _f0_norm      * 0.2
            + _gaze_num     * 0.1,
            3,
        )

        # is_critical_moment: yüksek gerilim VEYA tutarsızlık + ses stresi
        is_critical_moment = tension_score > 0.6 or (incongruence and voice_stress)

        # ------------------------------------------------------------------
        # Paket
        # ------------------------------------------------------------------
        packages.append(
            {
                "segment_id":        seg_id,
                "timestamp":         f"{_format_mmss(seg_start)} - {_format_mmss(seg_end)}",
                "start":             seg_start,
                "end":               seg_end,
                "text":              text,
                # Yüz / duygu
                "dominant_emotion":  dominant_emotion,
                "emotion_confidence": emotion_confidence,
                # Bakış
                "avg_gaze_pitch":    avg_gaze_pitch,
                "avg_gaze_yaw":      avg_gaze_yaw,
                "gaze_direction":    gaze_dir,
                # Konuşma
                "speech_style":      speech_style,
                "speech_confidence": speech_confidence,
                "f0_mean":           f0_mean_val,
                "f0_std":            f0_std_val,
                "rms_dbfs":          rms_dbfs_val,
                # HuBERT valence / arousal
                "hubert_valence":    hubert_valence,
                "hubert_arousal":    hubert_arousal,
                # Türetilmiş davranışsal göstergeler
                "gaze_away":         gaze_away,
                "voice_stress":      voice_stress,
                "incongruence":      incongruence,
                "tension_score":     tension_score,
                "is_critical_moment": is_critical_moment,
            }
        )

    return packages


def generate_analysis(
    text_segments: List[Dict],
    audio_signal_timeline: List[Dict],
    visual_signal_timeline: Optional[List[Dict]] = None,
    face_timeline: Optional[List[Dict]] = None,
    voice_timeline: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Public entry point — thin wrapper around build_segment_signal_packages."""
    return build_segment_signal_packages(
        text_segments=text_segments,
        audio_signal_timeline=audio_signal_timeline,
        visual_signal_timeline=visual_signal_timeline,
        face_timeline=face_timeline,
        voice_timeline=voice_timeline,
    )
