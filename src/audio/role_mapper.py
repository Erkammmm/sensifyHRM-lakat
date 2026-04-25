# Amaç:
# Speaker_0..Speaker_N etiketlerini panel mülakat mantığına göre
# 1 Aday + N Mülakatçı rolüne çevirmek.
# Bu sürümde "kim kime soru soruyor, kim sorulara cevap veriyor"
# davranışını daha güçlü kullanıyoruz.

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


# Amaç:
# Gerçekten mülakatçı tarafında daha anlamlı olan kalıplar.
INTERVIEWER_PATTERNS = [
    "kendinizi tanıtır mısınız",
    "kendinizden bahseder misiniz",
    "bu projede ne yaptınız",
    "neden bu şirket",
    "neden bizi seçtiniz",
    "güçlü yönleriniz",
    "zayıf yönleriniz",
    "örnek verebilir misiniz",
    "bana biraz bahseder misiniz",
    "deneyiminizden bahseder misiniz",
]

# Amaç:
# Aday tarafında daha sık görülen deneyim/cevap kalıpları.
CANDIDATE_PATTERNS = [
    "ben",
    "çalıştım",
    "yaptım",
    "geliştirdim",
    "projemde",
    "stajımda",
    "kullandım",
    "sorumluydum",
    "öğrendim",
    "görev aldım",
    "tecrübe",
    "deneyim",
]

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
]

# Amaç:
# Aday üstüne yanlış düşen mülakatçı sorularını tespit etmek için ipuçları.
INTERVIEW_QUESTION_HINTS = [
    "kendinizi",
    "tanıtır mısınız",
    "bahseder misiniz",
    "hangi alanda",
    "hangi şehirde",
    "neden yazılım",
    "neden bu şirket",
    "makine öğrenmesinin ne olduğunu",
    "iş hayatınızda",
    "derin öğrenme nedir",
    "ann nasıl çalışır",
    "yurt dışına gittin mi",
    "gitmeyi düşünüyor musun",
    "görme fırsatı buldunuz mu",
    "hangisini daha çok sevdiniz",
    "örnek verebilir misiniz",
]

# Amaç:
# Adayın kendi sorduğu soruları yanlışlıkla mülakatçıya taşımamak için istisnalar.
INTERVIEW_QUESTION_EXCLUDES = [
    "başka soru sormak ister misiniz bana",
    "bana?",
]


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
            full_text = " ".join((s.get("text", "") or "") for s in segs).lower()

            question_count = sum(1 for s in segs if "?" in (s.get("text", "") or ""))
            long_turn_count = sum(1 for s in segs if (float(s["end"]) - float(s["start"])) >= 6.0)
            short_turn_count = sum(1 for s in segs if (float(s["end"]) - float(s["start"])) <= 4.0)

            interviewer_hits = sum(full_text.count(p) for p in INTERVIEWER_PATTERNS)
            candidate_hits = sum(full_text.count(p) for p in CANDIDATE_PATTERNS)
            courtesy_hits = sum(full_text.count(p) for p in COURTESY_PATTERNS)
            first_person_hits = sum(full_text.count(p) for p in [" ben ", " yaptım", " çalıştım", " kullandım"])

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
                "interviewer_hits": interviewer_hits,
                "candidate_hits": candidate_hits,
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
        # Soru-cevap yönünü çıkarmak.
        # Eğer A speaker soru soruyor ve hemen sonra B uzun/gerçek bir cevap veriyorsa:
        # - A, başkasına soru sormuş sayılır
        # - B, başkasının sorusuna cevap vermiş sayılır
        for i in range(1, len(segments_sorted)):
            prev_seg = segments_sorted[i - 1]
            curr_seg = segments_sorted[i]

            prev_spk = prev_seg.get("speaker_id", "Speaker_0")
            curr_spk = curr_seg.get("speaker_id", "Speaker_0")

            if prev_spk == curr_spk:
                continue

            prev_text = (prev_seg.get("text", "") or "").lower()
            curr_text = (curr_seg.get("text", "") or "").lower()
            curr_dur = max(0.0, float(curr_seg["end"]) - float(curr_seg["start"]))

            # Amaç:
            # Selamlaşma sorularını ana soru akışından düşürmek.
            prev_is_courtesy = any(p in prev_text for p in COURTESY_PATTERNS)

            # Amaç:
            # Önceki segment gerçek bir soruysa ve mevcut segment anlamlı bir cevapsa
            # etkileşimi speaker graph olarak say.
            if "?" in prev_text and not prev_is_courtesy and curr_dur >= 1.2 and len(curr_text.split()) >= 3:
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
    def _score_roles(self, stats: Dict[str, Dict]) -> Dict[str, Dict]:
        if not stats:
            return {}

        max_total_duration = max((v["total_duration"] for v in stats.values()), default=1.0) or 1.0

        scored = {}
        for speaker_id, item in stats.items():
            duration_norm = item["total_duration"] / max_total_duration

            # Amaç:
            # Nezaket sorularını etkisizleştir.
            effective_question_count = max(0, item["question_count"] - item["courtesy_hits"])
            effective_question_ratio = effective_question_count / max(item["segment_count"], 1)

            interviewer_score = (
                effective_question_ratio * 2.6
                + effective_question_count * 0.7
                + item["interviewer_hits"] * 3.2
                + item["asked_others_count"] * 2.8
                + item["question_outgoing_targets_count"] * 2.0
                + item["short_turn_count"] * 0.3
                + item["first_segment_has_question"] * 0.3
                - duration_norm * 2.8
                - item["answer_turn_count"] * 2.2
                - item["candidate_hits"] * 0.7
            )

            candidate_score = (
                item["candidate_hits"] * 2.7
                + item["first_person_hits"] * 1.6
                + item["long_turn_count"] * 1.1
                + duration_norm * 4.2
                + item["answer_turn_count"] * 3.1
                + item["received_question_answers"] * 2.6
                - item["asked_others_count"] * 2.8
                - item["question_outgoing_targets_count"] * 1.8
                - item["interviewer_hits"] * 0.6
                - effective_question_ratio * 0.4
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
    # Yeni mantık:
    # - soru alan / cevap veren kişi yukarı çıkar
    # - başkalarına soru yönelten kişi aşağı iner
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