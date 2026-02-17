"""
FAZ-3 Thought Unit Merger

Amaç:
  - STT'nin ürettiği çok kısa (1-2sn) parçaları "anlamlı konu blokları" haline getirmek.
  - LLM'ye ve rapora giden segment sayısını düşürmek (ürünleşme).

Heuristics:
  - Yakın zamanlı segmentleri (gap <= max_gap_sec) birleştir
  - Çok kısa segmentler mutlaka komşusuna kaynaştırılmaya çalışılır
  - Blokları hedef süre aralığında tut (min_block_sec ... max_block_sec)
"""

from __future__ import annotations

import os
import re
from typing import Dict, List


MIN_SEG_SEC = float(os.getenv("SENSIFYHR_MIN_SEG_SEC", "1.2"))
MAX_GAP_SEC = float(os.getenv("SENSIFYHR_MAX_GAP_SEC", "0.8"))
MIN_BLOCK_SEC = float(os.getenv("SENSIFYHR_MIN_BLOCK_SEC", "15.0"))
MAX_BLOCK_SEC = float(os.getenv("SENSIFYHR_MAX_BLOCK_SEC", "35.0"))
TARGET_BLOCK_SEC = float(os.getenv("SENSIFYHR_TARGET_BLOCK_SEC", "20.0"))


_END_PUNCT_RE = re.compile(r"[.!?…]+$")


def merge_into_thought_units(stt_segments: List[Dict]) -> List[Dict]:
    """
    Input: [{start,end,text}, ...]
    Output: merged thought units [{start,end,text}, ...]
    """
    segs = [s for s in (stt_segments or []) if isinstance(s, dict)]
    if not segs:
        return []

    # sort just in case
    segs.sort(key=lambda x: float(x.get("start", 0.0) or 0.0))

    out: List[Dict] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_text_parts: List[str] = []

    for i, seg in enumerate(segs):
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", s) or s)
        t = " ".join((seg.get("text", "") or "").strip().split())
        if not t:
            continue

        seg_dur = max(0.0, e - s)
        is_short = seg_dur < MIN_SEG_SEC or len(t) < 8

        if cur_start is None:
            cur_start, cur_end = s, e
            cur_text_parts = [t]
            continue

        gap = max(0.0, s - float(cur_end))
        cur_dur = max(0.0, float(cur_end) - float(cur_start))

        can_join = gap <= MAX_GAP_SEC and (cur_dur + gap + seg_dur) <= (MAX_BLOCK_SEC + 1.0)

        # kısa segmentleri agresif şekilde birleştir
        if is_short and gap <= (MAX_GAP_SEC * 2.0):
            can_join = True

        if can_join:
            cur_text_parts.append(t)
            cur_end = max(float(cur_end), e)
            cur_dur = max(0.0, float(cur_end) - float(cur_start))

            # hedefe ulaştıysa ve cümle bitişi varsa blok kapat
            if cur_dur >= TARGET_BLOCK_SEC and bool(_END_PUNCT_RE.search(t.strip())):
                text = " ".join(cur_text_parts).strip()
                if text:
                    out.append({"start": float(cur_start), "end": float(cur_end), "text": text})
                cur_start, cur_end, cur_text_parts = None, None, []
            continue

        # join edemiyorsak: blok çok kısa kalmasın diye flush kuralı
        if cur_dur < MIN_BLOCK_SEC and gap <= (MAX_GAP_SEC * 2.0):
            # kısa blok + yakın segment: zorla birleştir
            cur_text_parts.append(t)
            cur_end = max(float(cur_end), e)
            continue

        # flush
        if cur_start is not None and cur_end is not None:
            text = " ".join(cur_text_parts).strip()
            if text:
                out.append({"start": float(cur_start), "end": float(cur_end), "text": text})
        cur_start, cur_end = s, e
        cur_text_parts = [t]

    # final flush
    if cur_start is not None and cur_end is not None:
        text = " ".join(cur_text_parts).strip()
        if text:
            out.append({"start": float(cur_start), "end": float(cur_end), "text": text})
    return out

