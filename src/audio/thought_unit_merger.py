# Amaç:
# STT'nin ürettiği kısa segmentleri anlamlı thought unit bloklarına birleştirirken
# speaker_id, speaker_role ve role_confidence bilgisini korumak.

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Optional


MIN_SEG_SEC = float(os.getenv("SENSIFYHR_MIN_SEG_SEC", "1.2"))
MAX_GAP_SEC = float(os.getenv("SENSIFYHR_MAX_GAP_SEC", "0.8"))
MIN_BLOCK_SEC = float(os.getenv("SENSIFYHR_MIN_BLOCK_SEC", "15.0"))
MAX_BLOCK_SEC = float(os.getenv("SENSIFYHR_MAX_BLOCK_SEC", "35.0"))
TARGET_BLOCK_SEC = float(os.getenv("SENSIFYHR_TARGET_BLOCK_SEC", "20.0"))

_END_PUNCT_RE = re.compile(r"[.!?…]+$")


def _build_merged_block(start: float, end: float, parts: List[str], segments: List[Dict]) -> Dict:
    # Amaç:
    # Birleştirilen blok için baskın speaker ve role bilgisini hesaplamak.
    speaker_ids = [s.get("speaker_id", "Speaker_0") for s in segments]
    speaker_roles = [s.get("speaker_role", "Bilinmiyor") for s in segments]
    role_confidences = [float(s.get("role_confidence", 0.0)) for s in segments]

    dominant_speaker_id = Counter(speaker_ids).most_common(1)[0][0] if speaker_ids else "Speaker_0"
    dominant_speaker_role = Counter(speaker_roles).most_common(1)[0][0] if speaker_roles else "Bilinmiyor"
    avg_role_confidence = round(sum(role_confidences) / max(len(role_confidences), 1), 3)

    return {
        "start": float(start),
        "end": float(end),
        "text": " ".join(parts).strip(),
        "speaker_id": dominant_speaker_id,
        "speaker_role": dominant_speaker_role,
        "role_confidence": avg_role_confidence,
    }


def merge_into_thought_units(stt_segments: List[Dict]) -> List[Dict]:
    """
    Input:
      [{start,end,text,speaker_id?,speaker_role?,role_confidence?}, ...]

    Output:
      [{start,end,text,speaker_id,speaker_role,role_confidence}, ...]
    """
    segs = [s for s in (stt_segments or []) if isinstance(s, dict)]
    if not segs:
        return []

    # Amaç:
    # Segmentleri zamana göre sıralamak.
    segs.sort(key=lambda x: float(x.get("start", 0.0) or 0.0))

    out: List[Dict] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_text_parts: List[str] = []
    cur_segments: List[Dict] = []

    for seg in segs:
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
            cur_segments = [seg]
            continue

        gap = max(0.0, s - float(cur_end))
        cur_dur = max(0.0, float(cur_end) - float(cur_start))

        last_speaker_id = cur_segments[-1].get("speaker_id", "Speaker_0")
        last_speaker_role = cur_segments[-1].get("speaker_role", "Bilinmiyor")
        this_speaker_id = seg.get("speaker_id", "Speaker_0")
        this_speaker_role = seg.get("speaker_role", "Bilinmiyor")

        same_speaker = last_speaker_id == this_speaker_id
        same_role = last_speaker_role == this_speaker_role

        # Amaç:
        # Normalde yalnızca yakın ve süre limiti uygun segmentleri birleştir.
        can_join = gap <= MAX_GAP_SEC and (cur_dur + gap + seg_dur) <= (MAX_BLOCK_SEC + 1.0)

        # Amaç:
        # Speaker veya role değiştiyse birleşmeyi zorlaştır.
        if not same_speaker or not same_role:
            can_join = False

        # Amaç:
        # Çok kısa segmentleri daha agresif bağla ama yalnızca speaker aynıysa.
        if is_short and same_speaker and gap <= (MAX_GAP_SEC * 2.0):
            can_join = True

        if can_join:
            cur_text_parts.append(t)
            cur_segments.append(seg)
            cur_end = max(float(cur_end), e)
            cur_dur = max(0.0, float(cur_end) - float(cur_start))

            # Amaç:
            # Hedef süreye geldiysek ve son cümle kapanmışsa thought unit'i kapatmak.
            if cur_dur >= TARGET_BLOCK_SEC and bool(_END_PUNCT_RE.search(t.strip())):
                out.append(_build_merged_block(cur_start, cur_end, cur_text_parts, cur_segments))
                cur_start, cur_end, cur_text_parts, cur_segments = None, None, [], []

            continue

        # Amaç:
        # Çok kısa thought unit oluşmasını önlemek için, yakın segmenti zorla birleştirmek.
        if cur_dur < MIN_BLOCK_SEC and gap <= (MAX_GAP_SEC * 2.0):
            cur_text_parts.append(t)
            cur_segments.append(seg)
            cur_end = max(float(cur_end), e)
            continue

        # Amaç:
        # Mevcut thought unit'i kapatıp yenisine geçmek.
        out.append(_build_merged_block(cur_start, cur_end, cur_text_parts, cur_segments))
        cur_start, cur_end = s, e
        cur_text_parts = [t]
        cur_segments = [seg]

    # Amaç:
    # Son bloğu da çıktı listesine eklemek.
    if cur_start is not None and cur_end is not None and cur_text_parts:
        out.append(_build_merged_block(cur_start, cur_end, cur_text_parts, cur_segments))

    return out