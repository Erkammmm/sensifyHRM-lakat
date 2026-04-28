# Amaç:
# Speaker_0..Speaker_N etiketlerini panel mülakat mantığına göre
# 1 Aday + N Mülakatçı rolüne çevirmek.
#
# Bu sürüm alan bağımsızdır:
# - soru içeriğine özel keyword listeleri kullanmaz
# - "kim kime soru soruyor, kim cevap veriyor" akışına bakar
# - yanlış sahiplenilmiş kısa soru segmentlerini interaction-driven olarak düzeltir

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


# Amaç:
# Selamlaşma / nezaket cümleleri role seçiminde güçlü sinyal olmasın.
COURTESY_PATTERNS = [
    "nasılsınız",
    "iyi misiniz",
    "hoş geldiniz",
    "hoş bulduk",
    "teşekkür ederim",
    "teşekkürler",
    "sağ olun",
    "merhaba",
    "iyi akşamlar",
    "iyi günler",
    "iyi çalışmalar",
]

# Amaç:
# Aday tarafında daha sık görülen self-reporting kalıpları.
# Bunlar alan bağımlı değil; kişinin kendinden bahsettiğini gösterir.
FIRST_PERSON_PATTERNS = [
    " ben ",
    " benim ",
    " bana ",
    " bence ",
    " yaptım",
    " çalıştım",
    " öğrendim",
    " geliştirdim",
    " kullandım",
    " sorumluydum",
    " deneyim",
    " tecrübe",
]


def _normalize_text(text: str) -> str:
    return " " + " ".join((text or "").lower().strip().split()) + " "


def _contains_any(text: str, patterns: List[str]) -> bool:
    return any(p in text for p in patterns)


def _count_any(text: str, patterns: List[str]) -> int:
    return sum(text.count(p) for p in patterns)


# Amaç:
# Nezaket / selamlaşma odaklı kısa segmentleri belirlemek.
def _is_courtesy_segment(text: str) -> bool:
    t = _normalize_text(text)
    if not t.strip():
        return False
    word_count = len(t.split())
    return word_count <= 8 and _contains_any(t, COURTESY_PATTERNS)


# Amaç:
# İçeriğe değil, biçime göre "kısa soru segmenti" tanımlamak.
def _is_question_segment(text: str, duration: float) -> bool:
    t = (text or "").strip()
    if not t or "?" not in t:
        return False

    if _is_courtesy_segment(t):
        return False

    word_count = len(t.split())

    # Kısa ve yönlendirici soru olma eğilimi
    return duration <= 8.0 or word_count <= 14


# Amaç:
# İçeriğe değil, yapıya göre "cevap benzeri" segment tanımlamak.
def _is_answer_like_segment(text: str, duration: float) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    if _is_courtesy_segment(t):
        return False

    word_count = len(t.split())

    # Çok kısa ve sadece soru olan şeyleri cevap sayma
    if "?" in t and word_count <= 4:
        return False

    return duration >= 1.2 and word_count >= 3


# Amaç:
# Role map çıkarıldıktan sonra, aday üstüne yanlış düşmüş kısa soru segmentlerini
# en uygun mülakatçıya geri atamak.
#
# Burada içerik kullanılmaz.
# Kural:
# - segment kısa soruysa
# - şu an aday üstündeyse
# - hemen ardından aynı speaker üzerinde cevap akışı başlıyorsa
# bu segment muhtemelen interviewer'a aittir.
def refine_question_ownership(segments: List[Dict], role_map: Dict[str, str]) -> List[Dict]:
    if not segments:
        return []

    interviewer_ids = [spk for spk, role in role_map.items() if str(role).startswith("Mülakatçı")]
    if not interviewer_ids:
        return [dict(seg) for seg in segments]

    out = [dict(seg) for seg in segments]

    def find_context_interviewer(idx: int) -> str | None:
        seg_start = float(out[idx].get("start", 0.0))

        best_speaker = None
        best_distance = float("inf")

        for j, other in enumerate(out):
            if j == idx:
                continue
            other_spk = other.get("speaker_id")
            if other_spk not in interviewer_ids:
                continue

            other_mid = (float(other.get("start", 0.0)) + float(other.get("end", 0.0))) / 2.0
            dist = abs(seg_start - other_mid)

            if dist < best_distance:
                best_distance = dist
                best_speaker = other_spk

        return best_speaker

    for i, seg in enumerate(out):
        seg_role = seg.get("speaker_role") or role_map.get(seg.get("speaker_id"), "Bilinmiyor")
        if seg_role != "Aday":
            continue

        seg_text = seg.get("text", "") or ""
        seg_duration = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))

        if not _is_question_segment(seg_text, seg_duration):
            continue

        current_speaker = seg.get("speaker_id")

        # Amaç:
        # Bu kısa sorudan hemen sonra aynı speaker üzerinde cevap akışı başlıyorsa,
        # soru büyük ihtimalle yanlış speaker'a yapışmıştır.
        answer_like_followup = False

        for j in range(i + 1, min(i + 4, len(out))):
            nxt = out[j]
            nxt_speaker = nxt.get("speaker_id")

            # Başka speaker geldiyse bu pencereyi bırak
            if nxt_speaker != current_speaker:
                break

            nxt_text = nxt.get("text", "") or ""
            nxt_dur = max(0.0, float(nxt.get("end", 0.0)) - float(nxt.get("start", 0.0)))

            if _is_answer_like_segment(nxt_text, nxt_dur):
                answer_like_followup = True
                break

        if not answer_like_followup:
            continue

        target_interviewer = find_context_interviewer(i)
        if target_interviewer and target_interviewer != current_speaker:
            out[i]["speaker_id"] = target_interviewer
            out[i]["speaker_role"] = role_map.get(target_interviewer, "Mülakatçı")

    return out


class RoleMapper:
    # Amaç:
    # Speaker bazında temel istatistikleri çıkarmak.
    def _speaker_stats(self, segments: List[Dict]) -> Dict[str, Dict]:
        grouped = defaultdict(list)
        for seg in segments:
            grouped[seg.get("speaker_id", "Speaker_0")].append(seg)

        total_all_duration = sum(
            max(0.0, float(s["end"]) - float(s["start"])) for s in segments
        )

        segments_sorted = sorted(segments, key=lambda x: float(x["start"]))
        first_segment_speaker = segments_sorted[0].get("speaker_id", "Speaker_0") if segments_sorted else "Speaker_0"
        first_segment_text = (segments_sorted[0].get("text", "") or "").lower() if segments_sorted else ""

        stats = {}
        for speaker_id, segs in grouped.items():
            segs_sorted = sorted(segs, key=lambda x: float(x["start"]))
            total_dur = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segs)
            full_text = _normalize_text(" ".join((s.get("text", "") or "") for s in segs))

            question_count = sum(1 for s in segs if "?" in (s.get("text", "") or ""))
            long_turn_count = sum(1 for s in segs if (float(s["end"]) - float(s["start"])) >= 6.0)
            short_turn_count = sum(1 for s in segs if (float(s["end"]) - float(s["start"])) <= 4.0)

            courtesy_hits = sum(1 for s in segs if _is_courtesy_segment(s.get("text", "") or ""))
            first_person_hits = _count_any(full_text, FIRST_PERSON_PATTERNS)

            starts_conversation = 1 if speaker_id == first_segment_speaker else 0
            first_segment_has_question = 1 if (speaker_id == first_segment_speaker and "?" in first_segment_text) else 0
            first_start = float(segs_sorted[0]["start"]) if segs_sorted else 999999.0

            stats[speaker_id] = {
                "segment_count": len(segs),
                "total_duration": total_dur,
                "duration_ratio": (total_dur / total_all_duration) if total_all_duration > 0 else 0.0,
                "question_count": question_count,
                "question_ratio": (question_count / max(len(segs), 1)),
                "long_turn_count": long_turn_count,
                "short_turn_count": short_turn_count,
                "courtesy_hits": courtesy_hits,
                "first_person_hits": first_person_hits,
                "starts_conversation": starts_conversation,
                "first_segment_has_question": first_segment_has_question,
                "first_start": first_start,
                "answer_turn_count": 0,
                "asked_others_count": 0,
                "received_question_answers": 0,
                "question_outgoing_targets": set(),
                "full_text": full_text,
            }

        # Amaç:
        # Soru-cevap yönünü konuşma akışından çıkarmak.
        # İçeriğe değil, "kısa soru → başka speaker'dan cevap" yapısına bakar.
        for i in range(1, len(segments_sorted)):
            prev_seg = segments_sorted[i - 1]
            curr_seg = segments_sorted[i]

            prev_spk = prev_seg.get("speaker_id", "Speaker_0")
            curr_spk = curr_seg.get("speaker_id", "Speaker_0")

            if prev_spk == curr_spk:
                continue

            prev_text = prev_seg.get("text", "") or ""
            curr_text = curr_seg.get("text", "") or ""
            prev_dur = max(0.0, float(prev_seg["end"]) - float(prev_seg["start"]))
            curr_dur = max(0.0, float(curr_seg["end"]) - float(curr_seg["start"]))

            if _is_question_segment(prev_text, prev_dur) and _is_answer_like_segment(curr_text, curr_dur):
                if prev_spk in stats:
                    stats[prev_spk]["asked_others_count"] += 1
                    stats[prev_spk]["question_outgoing_targets"].add(curr_spk)

                if curr_spk in stats:
                    stats[curr_spk]["answer_turn_count"] += 1
                    stats[curr_spk]["received_question_answers"] += 1

        for speaker_id in stats:
            stats[speaker_id]["question_outgoing_targets_count"] = len(
                stats[speaker_id]["question_outgoing_targets"]
            )
            del stats[speaker_id]["question_outgoing_targets"]

        return stats

    # Amaç:
    # Her speaker için interviewer ve candidate skorları üretmek.
    # Bu sürüm içerik domainine değil, etkileşim yapısına daha çok yaslanır.
    def _score_roles(self, stats: Dict[str, Dict]) -> Dict[str, Dict]:
        if not stats:
            return {}

        max_total_duration = max((v["total_duration"] for v in stats.values()), default=1.0) or 1.0

        scored = {}
        for speaker_id, item in stats.items():
            duration_norm = item["total_duration"] / max_total_duration

            effective_question_count = max(0, item["question_count"] - item["courtesy_hits"])
            effective_question_ratio = effective_question_count / max(item["segment_count"], 1)

            interviewer_score = (
                effective_question_ratio * 2.8
                + effective_question_count * 0.7
                + item["asked_others_count"] * 3.0
                + item["question_outgoing_targets_count"] * 2.2
                + item["short_turn_count"] * 0.25
                + item["starts_conversation"] * 0.2
                - duration_norm * 2.4
                - item["answer_turn_count"] * 2.4
                - item["received_question_answers"] * 1.8
            )

            candidate_score = (
                item["first_person_hits"] * 1.8
                + item["long_turn_count"] * 1.2
                + duration_norm * 4.0
                + item["answer_turn_count"] * 3.1
                + item["received_question_answers"] * 3.0
                - item["asked_others_count"] * 2.9
                - item["question_outgoing_targets_count"] * 2.0
                - effective_question_ratio * 0.5
            )

            candidate_advantage = candidate_score - interviewer_score

            scored[speaker_id] = {
                **item,
                "effective_question_count": effective_question_count,
                "effective_question_ratio": round(effective_question_ratio, 3),
                "interviewer_score": round(interviewer_score, 3),
                "candidate_score": round(candidate_score, 3),
                "candidate_advantage": round(candidate_advantage, 3),
            }

        return scored

    # Amaç:
    # Panel mülakatta tek adayı seçmek.
    # En çok soru alan / cevap veren / uzun açıklayan speaker öne çıkar.
    def _select_candidate(self, scored: Dict[str, Dict]) -> str:
        return max(
            scored.keys(),
            key=lambda spk: (
                scored[spk]["candidate_advantage"],
                scored[spk]["received_question_answers"],
                scored[spk]["answer_turn_count"],
                scored[spk]["total_duration"],
                -scored[spk]["asked_others_count"],
                -scored[spk]["question_outgoing_targets_count"],
            ),
        )

    # Amaç:
    # Nihai role map üretmek:
    # 1 aday, kalan herkes mülakatçı.
    def assign_roles(self, segments: List[Dict]) -> Tuple[List[Dict], Dict[str, str], Dict[str, Dict], Dict]:
        if not segments:
            return [], {}, {}, {
                "mapping_confidence": 0.0,
                "reason": "No segments available",
            }

        stats = self._speaker_stats(segments)
        scored = self._score_roles(stats)

        speakers = list(scored.keys())
        if not speakers:
            return [], {}, {}, {
                "mapping_confidence": 0.0,
                "reason": "No speakers scored",
            }

        candidate_id = self._select_candidate(scored)

        interviewer_ids = [spk for spk in speakers if spk != candidate_id]
        interviewer_ids = sorted(interviewer_ids, key=lambda spk: scored[spk]["first_start"])

        role_map = {candidate_id: "Aday"}
        for idx, spk in enumerate(interviewer_ids, start=1):
            role_map[spk] = f"Mülakatçı{idx}"

        candidate_adv = scored[candidate_id]["candidate_advantage"]
        strongest_interviewer_adv = max(
            [scored[spk]["candidate_advantage"] for spk in interviewer_ids],
            default=0.0
        )
        mapping_confidence = round(abs(candidate_adv - strongest_interviewer_adv), 3)

        diagnostics = {
            "mapping_confidence": mapping_confidence,
            "candidate_id": candidate_id,
            "interviewer_count": len(interviewer_ids),
            "reason": "Dynamic panel interview role mapping applied.",
        }

        enriched = []
        for seg in segments:
            item = dict(seg)
            spk = item.get("speaker_id", "Speaker_0")
            item["speaker_role"] = role_map.get(spk, "Aday")
            item["role_confidence"] = abs(scored.get(spk, {}).get("candidate_advantage", 0.0))
            enriched.append(item)

        return enriched, role_map, scored, diagnostics