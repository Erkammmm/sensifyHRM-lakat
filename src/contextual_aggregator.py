"""
FAZ-3 Contextual Aggregator

Her Whisper segmenti için:
  - text
  - audio_signal (valence/arousal + fiziksel state'ler)
  - visual_signal (facial_state/attention_state/stress_indicator)

zaman bazlı hizalanarak LLM'ye gidecek tek paket üretilir.

Not: Emotion/Sentiment etiketleri burada yoktur.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple


def _mode(values: List[str], default: str) -> str:
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


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


def build_segment_signal_packages(
    text_segments: List[Dict],
    audio_signal_timeline: List[Dict],
    visual_signal_timeline: List[Dict],
    voice_analysis: Optional[Dict] = None,
) -> List[Dict]:
    """
    Args:
      text_segments: FAZ-3 STT segmentleri [{start,end,text}, ...]
      audio_signal_timeline: [{start,end,valence_state,arousal_state,speech_energy,speech_rate,pitch_stability,...}, ...]
      visual_signal_timeline: [{timestamp,facial_state,attention_state,stress_indicator,...}, ...]

    Returns:
      Segment Signal Package listesi (LLM input).
    """
    packages: List[Dict] = []
    if not text_segments:
        return packages

    # --- helpers for max-pooling summaries (365Aspects inspired) ---
    def _max_series_in_range(series: Dict, start: float, end: float) -> Tuple[float, float]:
        """
        series: {"times":[...], "values":[...]}
        returns: (max_val, max_time) or (0.0, start) if empty
        """
        try:
            times = series.get("times", []) if isinstance(series, dict) else []
            vals = series.get("values", []) if isinstance(series, dict) else []
            if not times or not vals:
                return 0.0, start
            m = 0.0
            mt = start
            for t, v in zip(times, vals):
                try:
                    tf = float(t)
                    vf = float(v)
                except Exception:
                    continue
                if tf < start or tf > end:
                    continue
                if vf > m:
                    m = vf
                    mt = tf
            return float(m), float(mt)
        except Exception:
            return 0.0, start

    def _max_debug(audio_chunks: List[Dict], key: str) -> float:
        m = 0.0
        for a in audio_chunks or []:
            dbg = a.get("debug", {}) if isinstance(a, dict) else {}
            if not isinstance(dbg, dict):
                continue
            try:
                v = float(dbg.get(key, 0.0) or 0.0)
            except Exception:
                continue
            if v > m:
                m = v
        return float(m)

    def _ratio(values: List[str], target: str) -> float:
        vals = [v for v in values if isinstance(v, str) and v.strip()]
        if not vals:
            return 0.0
        hit = sum(1 for v in vals if v == target)
        return float(hit) / float(len(vals))

    raw_voice = (voice_analysis or {}).get("raw_voice_features", {}) if isinstance(voice_analysis, dict) else {}
    rms_series = ((raw_voice.get("energy_rms") or {}).get("series") or {}) if isinstance(raw_voice, dict) else {}
    pitch_series = ((raw_voice.get("pitch_f0") or {}).get("series") or {}) if isinstance(raw_voice, dict) else {}

    for seg in text_segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end = float(seg.get("end", seg_start) or seg_start)
        text = (seg.get("text") or "").strip()

        # --- audio overlap (interval intersection) ---
        audio_in_range = [
            a
            for a in (audio_signal_timeline or [])
            if float(a.get("start", 0.0) or 0.0) < seg_end
            and float(a.get("end", 0.0) or 0.0) > seg_start
        ]

        audio_signal = {
            "valence": _mode([a.get("valence_state", "") for a in audio_in_range], "NEUTRAL"),
            "arousal": _mode([a.get("arousal_state", "") for a in audio_in_range], "MEDIUM"),
            "speech_energy": _mode([a.get("speech_energy", "") for a in audio_in_range], "MEDIUM"),
            "speech_rate": _mode([a.get("speech_rate", "") for a in audio_in_range], "NORMAL"),
            "pitch_stability": _mode([a.get("pitch_stability", "") for a in audio_in_range], "STABLE"),
        }

        # --- visual overlap (point-in-interval) ---
        visual_in_range = [
            v
            for v in (visual_signal_timeline or [])
            if seg_start <= float(v.get("timestamp", 0.0) or 0.0) <= seg_end
        ]

        visual_signal = {
            "facial_state": _mode([v.get("facial_state", "") for v in visual_in_range], "NEUTRAL"),
            "attention_state": _mode([v.get("attention_state", "") for v in visual_in_range], "FOCUSED"),
            "stress_indicator": _mode([v.get("stress_indicator", "") for v in visual_in_range], "LOW"),
        }

        # --- 365Aspects-style max-pooling fusion summary (small + explainable) ---
        rms_max, rms_t = _max_series_in_range(rms_series, seg_start, seg_end)
        pitch_max, pitch_t = _max_series_in_range(pitch_series, seg_start, seg_end)
        val_raw_max = _max_debug(audio_in_range, "valence_score_raw")
        aro_raw_max = _max_debug(audio_in_range, "arousal_score_raw")
        ser_top_max = _max_debug(audio_in_range, "ser_top_score")

        stress_score_max = 0.0
        smile_score_max = 0.0
        for v in visual_in_range or []:
            try:
                stress_score_max = max(stress_score_max, float(v.get("stress_score", 0.0) or 0.0))
            except Exception:
                pass
            try:
                smile_score_max = max(smile_score_max, float(v.get("smile_score", 0.0) or 0.0))
            except Exception:
                pass
        attention_averted_ratio = _ratio([v.get("attention_state", "") for v in visual_in_range], "AVERTED")

        fusion_summary = {
            "max_pool": {
                "voice": {
                    "rms_max": round(float(rms_max), 4),
                    "rms_peak_time": round(float(rms_t), 2),
                    "pitch_max_hz": round(float(pitch_max), 1),
                    "pitch_peak_time": round(float(pitch_t), 2),
                },
                "audio_ser": {
                    "valence_raw_max": round(float(val_raw_max), 3),
                    "arousal_raw_max": round(float(aro_raw_max), 3),
                    "ser_top_score_max": round(float(ser_top_max), 3),
                },
                "visual": {
                    "stress_score_max": round(float(stress_score_max), 4),
                    "smile_score_max": round(float(smile_score_max), 4),
                    "attention_averted_ratio": round(float(attention_averted_ratio), 3),
                },
            }
        }

        packages.append(
            {
                "timestamp": f"{_format_mmss(seg_start)} - {_format_mmss(seg_end)}",
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "audio_signal": audio_signal,
                "visual_signal": visual_signal,
                "fusion_summary": fusion_summary,
            }
        )

    return packages

