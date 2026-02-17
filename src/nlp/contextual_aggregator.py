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
from typing import Dict, List, Optional


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

        packages.append(
            {
                "timestamp": f"{_format_mmss(seg_start)} - {_format_mmss(seg_end)}",
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "audio_signal": audio_signal,
                "visual_signal": visual_signal,
            }
        )

    return packages


def generate_analysis(
    text_segments: List[Dict],
    audio_signal_timeline: List[Dict],
    visual_signal_timeline: List[Dict],
) -> List[Dict]:
    """
    Public entry: build segment signal packages for LLM input.
    Alias for `build_segment_signal_packages` to provide a single clear entry point.
    """
    return build_segment_signal_packages(text_segments, audio_signal_timeline, visual_signal_timeline)
