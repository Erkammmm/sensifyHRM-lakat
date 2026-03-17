"""
Rapor Oluşturucu
HTML + JSON rapor üretir, matplotlib grafikleri embed eder.
"""

import os
import re
import json
import math
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from .plot import (
    generate_report_charts,
    plot_face_emotion_timeline_b64,
    plot_voice_valence_timeline_b64,
    plot_gaze_timeline_b64,
    plot_speech_confidence_timeline_b64,
    plot_voice_energy_timeline_b64,
)


class ReportGenerator:
    """HTML ve JSON raporları oluşturur."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        # Jinja template environment (templates live in src/reporting/templates/)
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.template_env = Environment(loader=FileSystemLoader(templates_dir))

    def generate_report(self, report: Dict) -> Dict[str, str]:
        """
        Tam rapor oluşturur (JSON + HTML).

        Args:
            report: pipeline.process_interview() çıktısı

        Returns:
            {"json": json_path, "html": html_path}
        """
        interview_id = report.get("interview_id", "unknown")

        # 1) Grafikleri üret
        charts_dir = os.path.join(self.reports_dir, "charts")
        charts = generate_report_charts(report, charts_dir)
        charts_html = self._embed_charts(charts)

        # 2) HTML rapor oluştur
        html_path = self._create_html_report(report, charts_html)

        # 3) JSON rapor kaydet
        json_path = self._save_json_report(report)

        print(f"[Report] Raporlar oluşturuldu: {interview_id}")
        return {"json": json_path, "html": html_path}

    @staticmethod
    def _embed_charts(charts: List[Dict]) -> str:
        """Grafik dosyalarını HTML base64 <img> olarak embed eder."""
        if not charts:
            return ""
        blocks = []
        for chart in charts:
            path = chart.get("path", "")
            if not path or not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            title = chart.get("title", "Grafik")
            blocks.append(f'<h3>{title}</h3>')
            blocks.append(
                f'<div class="chart-container">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;" />'
                f'</div>'
            )
        return "\n".join(blocks)

    def _create_html_report(self, report: Dict, charts_html: str) -> str:
        """HTML raporu oluşturur."""
        interview_id = report.get("interview_id", "unknown")
        duration = report.get("duration_seconds", 0)
        video_info = report.get("video_info", {})
        phase = (report.get("phase", "v2") or "v2").lower()
        is_phase3 = phase == "v3"

        # Özetler
        text_summary = report.get("text_analysis", {}).get("summary", {})
        text_segments = report.get("text_analysis", {}).get("segments", [])
        audio_summary = report.get("audio_emotion_analysis", {}).get("summary", {})
        face_summary = report.get("face_analysis", {}).get("summary", {})
        audio_signal_summary = report.get("audio_signal_analysis", {}).get("summary", {})
        visual_signal_summary = report.get("visual_signal_analysis", {}).get("summary", {})
        voice_raw = report.get("voice_analysis", {}).get("raw_voice_features", {})
        anomalies = report.get("anomalies", [])
        ai_text = report.get("ai_analysis", {}).get("analysis", "") if isinstance(report.get("ai_analysis"), dict) else ""
        segment_packages = report.get("segment_signal_packages", []) if is_phase3 else []
        thought_units = report.get("text_analysis", {}).get("thought_units", []) if is_phase3 else []
        time_blocks = report.get("time_blocks", []) if is_phase3 else []

        # v3 ürün KPI skorları (basit heuristik)
        v3_scores = {"confidence": 50, "stress_control": 50, "communication": 50}
        if is_phase3:
            v3_scores = _compute_v3_product_scores(report)

        # Ses metrikleri
        ss = voice_raw.get("speech_silence", {})
        pp = voice_raw.get("preprocessing", {})

        # Metin segmentleri HTML tablosu
        text_table_rows = ""
        for seg in text_segments[:50]:
            if is_phase3 or "sentiment" not in seg:
                text_table_rows += (
                    f'<tr>'
                    f'<td>{seg.get("start", 0):.1f}s - {seg.get("end", 0):.1f}s</td>'
                    f'<td>{seg.get("text", "")}</td>'
                    f'</tr>'
                )
            else:
                color = "#2ecc71" if seg.get("sentiment") == "positive" else "#e74c3c"
                text_table_rows += (
                    f'<tr>'
                    f'<td>{seg.get("start", 0):.1f}s - {seg.get("end", 0):.1f}s</td>'
                    f'<td>{seg.get("text", "")}</td>'
                    f'<td style="color:{color};font-weight:bold;">{seg.get("sentiment", "")}</td>'
                    f'<td>{seg.get("confidence", 0):.2f}</td>'
                    f'</tr>'
                )

        # Anomali tablosu
        anomaly_rows = ""
        for a in anomalies:
            anomaly_rows += (
                f'<tr>'
                f'<td>{a.get("time_range", "")}</td>'
                f'<td>{a.get("text", "")[:80]}</td>'
                f'<td>{a.get("text_sentiment", "")}</td>'
                f'<td>{a.get("face_emotion", "")}</td>'
                f'<td style="color:#e74c3c;font-weight:bold;">{a.get("result", "")}</td>'
                f'</tr>'
            )

        # AI metni formatla (tam metin)
        formatted_ai = ""
        if ai_text:
            escaped = ai_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            paragraphs = [p.strip() for p in escaped.replace("\r\n", "\n").split("\n\n") if p.strip()]
            formatted_ai = "</p><p>".join(p.replace("\n", "<br>") for p in paragraphs)

        # FAZ-3: AI metnini başlıklara göre parçala (Dashboard kartları için)
        ai_sections = {}
        critical_cards_html = ""
        exec_summary_html = ""
        soft_skill_cards_html = ""
        if is_phase3 and ai_text:
            ai_sections = _split_markdown_sections(ai_text)
            critical = ai_sections.get("kritik_anlar", "").strip()
            executive = ai_sections.get("yonetici_ozeti", "").strip()
            soft_skill = ai_sections.get("soft_skill", "").strip()

            critical_cards_html = _render_bullets_as_cards(critical, max_items=6)
            exec_summary_html = _render_bullets_as_list(executive, max_items=6)
            soft_skill_cards_html = _render_soft_skill_cards(soft_skill, max_items=8)

        # FAZ-3: Thought Units tablosu (ürünleşme)
        thought_rows = ""
        if is_phase3 and thought_units:
            for tu in thought_units[:20]:
                thought_rows += (
                    "<tr>"
                    f"<td>{tu.get('start',0):.1f}s - {tu.get('end',0):.1f}s</td>"
                    f"<td>{(tu.get('text','') or '')[:280]}</td>"
                    "</tr>"
                )

        # FAZ-3 segment sinyal tablosu (explainability) -> Detaylar sekmesine
        segment_rows = ""
        if is_phase3 and segment_packages:
            for pkg in segment_packages[:30]:
                a = pkg.get("audio_signal", {}) or {}
                v = pkg.get("visual_signal", {}) or {}
                segment_rows += (
                    "<tr>"
                    f"<td>{pkg.get('timestamp','')}</td>"
                    f"<td>{(pkg.get('text','') or '')[:180]}</td>"
                    f"<td>{_state_badge(_tr_valence(a.get('valence','')), _sev_valence(a.get('valence','')))}</td>"
                    f"<td>{_state_badge(_tr_arousal(a.get('arousal','')), _sev_arousal(a.get('arousal','')))}</td>"
                    f"<td>{_state_badge(_tr_level(a.get('speech_energy','')), _sev_level(a.get('speech_energy','')))}</td>"
                    f"<td>{_state_badge(_tr_rate(a.get('speech_rate','')), _sev_rate(a.get('speech_rate','')))}</td>"
                    f"<td>{_state_badge(_tr_pitch(a.get('pitch_stability','')), _sev_pitch(a.get('pitch_stability','')))}</td>"
                    f"<td>{_state_badge(_tr_facial(v.get('facial_state','')), _sev_facial(v.get('facial_state','')))}</td>"
                    f"<td>{_state_badge(_tr_attention(v.get('attention_state','')), _sev_attention(v.get('attention_state','')))}</td>"
                    f"<td>{_state_badge(_tr_stress(v.get('stress_indicator','')), _sev_stress(v.get('stress_indicator','')))}</td>"
                    "</tr>"
                )

        if is_phase3:
            face_timeline = report.get("face_analysis", {}).get("timeline", []) or []
            faz4 = _compute_faz4_dashboard_data(segment_packages, face_timeline=face_timeline)
            emo_dist = _compute_emotion_distribution(face_timeline)
            audio_signal_tl = report.get("audio_signal_analysis", {}).get("timeline", []) or []
            voice_emo_dist = _compute_voice_emotion_distribution(audio_signal_tl)

            # 5 matplotlib timeline charts (base64 PNG)
            chart_face_b64    = plot_face_emotion_timeline_b64(segment_packages)
            chart_valence_b64 = plot_voice_valence_timeline_b64(segment_packages)
            chart_gaze_b64    = plot_gaze_timeline_b64(segment_packages)
            chart_speech_b64  = plot_speech_confidence_timeline_b64(segment_packages)
            chart_energy_b64  = plot_voice_energy_timeline_b64(segment_packages)

            # "Baskın Duygu" KPI card — top emotion from confidence-filtered distribution
            _neg_emos = {"Sad", "Fear", "Angry", "Disgust"}
            if not emo_dist["no_data"] and emo_dist["labels"]:
                top_idx = emo_dist["values"].index(max(emo_dist["values"]))
                baskin_duygu_label = emo_dist["labels"][top_idx]
                baskin_duygu_pct = f"{emo_dist['values'][top_idx]:.1f}%"
                baskin_duygu_card_class = "danger" if baskin_duygu_label in _neg_emos else "accent2"
            else:
                baskin_duygu_label = "—"
                baskin_duygu_pct = "—"
                baskin_duygu_card_class = "warn"
            speech_stats = _compute_speech_stats(
                text_segments,
                float(video_info.get("duration_seconds") or 0.0),
            )
            template = self.template_env.get_template("report_v3.html")
            html = template.render(
                interview_id=interview_id,
                analysis_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                video_duration=f'{video_info.get("duration_seconds", 0):.1f}',
                # Kart 1 — Göz Teması
                gaze_label=faz4["gaze_label"],
                gaze_card_class=faz4["gaze_card_class"],
                gaze_focus_pct=faz4["gaze_focus_pct"],
                gaze_aversion_ts=faz4["gaze_aversion_ts"],
                # Kart 2 — Ses Güveni
                speech_label=faz4["speech_label"],
                speech_card_class=faz4["speech_card_class"],
                avg_speech_confidence=faz4["avg_speech_confidence"],
                conf_drop_ts=faz4["conf_drop_ts"],
                baskin_ses_str=voice_emo_dist["top2_str"],
                # Kart 3 — Duygusal Denge
                emotion_label=faz4["emotion_label"],
                emotion_card_class=faz4["emotion_card_class"],
                emotion_neg_pct=faz4["emotion_neg_pct"],
                crit_count=faz4["crit_count"],
                crit_timestamps=faz4["crit_timestamps"],
                # Kart 4 — Baskın Duygu
                baskin_duygu_label=baskin_duygu_label,
                baskin_duygu_pct=baskin_duygu_pct,
                baskin_duygu_card_class=baskin_duygu_card_class,
                # 5 matplotlib zaman çizelgesi grafikleri (base64 PNG)
                chart_face_b64=chart_face_b64,
                chart_valence_b64=chart_valence_b64,
                chart_gaze_b64=chart_gaze_b64,
                chart_speech_b64=chart_speech_b64,
                chart_energy_b64=chart_energy_b64,
                # Konuşma istatistikleri (Step F)
                speech_dur_str=speech_stats["speech_dur_str"],
                silence_dur_str=speech_stats["silence_dur_str"],
                long_silence_count=speech_stats["long_silence_count"],
                longest_silence_dur=speech_stats["longest_silence_dur"],
                longest_silence_ts=speech_stats["longest_silence_ts"],
                avg_words=speech_stats["avg_words"],
                pace_label=speech_stats["pace_label"],
                # Duygu dağılımı — yüz (Step D)
                emo_dist_no_data=emo_dist["no_data"],
                emo_dist_labels_json=json.dumps(emo_dist["labels"], ensure_ascii=False),
                emo_dist_values_json=json.dumps(emo_dist["values"]),
                emo_dist_colors_json=json.dumps(emo_dist["colors"]),
                # Ses duygu dağılımı — SER (Step 4)
                voice_emo_dist_no_data=voice_emo_dist["no_data"],
                voice_emo_dist_labels_json=json.dumps(voice_emo_dist["labels"], ensure_ascii=False),
                voice_emo_dist_values_json=json.dumps(voice_emo_dist["values"]),
                voice_emo_dist_colors_json=json.dumps(voice_emo_dist["colors"]),
                # Konuşma Yapısı — Thought Units
                thought_units=thought_units[:20],
                thought_unit_count=len(thought_units),
                # Zaman Bloğu Paragrafları
                time_blocks=time_blocks,
                time_block_count=len(time_blocks),
                # LLM raporu: injected by _write_ai_to_reports after generation
            )
        else:
            template = self.template_env.get_template("report_v2.html")
            html = template.render(
                interview_id=interview_id,
                phase=phase,
                is_phase3=False,
                analysis_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                duration_seconds=f"{duration:.2f}",
                video_fps=video_info.get("fps", 0),
                video_resolution=f'{video_info.get("width", 0)}x{video_info.get("height", 0)}',
                video_duration=f'{video_info.get("duration_seconds", 0):.1f}',
                focus_score=f'{face_summary.get("focus_score", 0):.1f}',
                blink_rate=f'{face_summary.get("blink_rate_per_min", 0):.1f}',
                dominant_face_emotion=face_summary.get("dominant_emotion", "?"),
                total_blinks=face_summary.get("total_blinks", 0),
                dominant_valence=audio_signal_summary.get("dominant_valence", "?"),
                dominant_arousal=audio_signal_summary.get("dominant_arousal", "?"),
                dominant_facial_state=face_summary.get("dominant_emotion", "?"),
                total_sentences=text_summary.get("total_sentences", 0),
                dominant_sentiment=text_summary.get("dominant_sentiment", "?"),
                positive_pct=text_summary.get("sentiment_percentages", {}).get("positive", 0),
                negative_pct=text_summary.get("sentiment_percentages", {}).get("negative", 0),
                dominant_audio_emotion=audio_summary.get("dominant_emotion", "?"),
                audio_avg_confidence=f'{audio_summary.get("avg_confidence", 0):.2f}',
                audio_total_chunks=audio_summary.get("total_chunks", 0),
                speech_seconds=f'{ss.get("total_speech_seconds", 0):.1f}',
                silence_seconds=f'{ss.get("total_silence_seconds", 0):.1f}',
                speech_ratio=f'{ss.get("speech_silence_ratio", 0):.2f}',
                avg_silence=f'{ss.get("average_silence_seconds", 0):.2f}',
                rms_mean=f'{voice_raw.get("energy_rms", {}).get("mean", 0):.4f}',
                pitch_mean=f'{voice_raw.get("pitch_f0", {}).get("mean", 0):.1f}',
                pitch_var=f'{voice_raw.get("pitch_f0", {}).get("variability", 0):.1f}',
                anomaly_count=len(anomalies),
                anomaly_rows=anomaly_rows,
                text_table_rows=text_table_rows,
                text_table_phase3=False,
                charts_html=charts_html,
                ai_hr_text=formatted_ai,
                critical_cards_html="",
                exec_summary_html="",
                soft_skill_cards_html="",
                thought_rows="",
                thought_count=0,
                v3_confidence=50,
                v3_stress_control=50,
                v3_communication=50,
                segment_rows="",
                segment_count=0,
            )

        html_path = os.path.join(self.reports_dir, f"report_{interview_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return html_path

    def _save_json_report(self, report: Dict) -> str:
        """JSON raporu kaydeder (zaman serileri kırpılmış)."""
        interview_id = report.get("interview_id", "unknown")

        # Derin kopyalama ve seriler kırpma
        clean = _sanitize_for_json(report)

        # Büyük serileri kaldır
        voice = clean.get("voice_analysis", {}).get("raw_voice_features", {})
        if isinstance(voice, dict):
            voice.pop("waveform", None)
            voice.pop("mel_spectrogram", None)
            voice.pop("windowed_features", None)
            for key in ("energy_rms", "pitch_f0"):
                sub = voice.get(key)
                if isinstance(sub, dict):
                    sub.pop("series", None)
            ss = voice.get("speech_silence")
            if isinstance(ss, dict):
                ss.pop("speech_segments", None)

        json_path = os.path.join(self.reports_dir, f"report_{interview_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        return json_path


# =====================================================================
# Yardımcılar
# =====================================================================
def _sanitize_for_json(obj: Any) -> Any:
    """NumPy/NaN gibi tipleri JSON'a uygun hale getirir."""
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [_sanitize_for_json(v) for v in obj.tolist()]
        if isinstance(obj, np.generic):
            obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


# =====================================================================
# (HTML templates moved to `src/reporting/templates/` - kept there to keep this module concise)


def _state_badge(text: str, sev: str) -> str:
    sev = sev if sev in ("good", "warn", "bad") else "warn"
    safe = (text or "").strip() or "-"
    return f'<span class="badge sev-{sev}"><span class="dot"></span>{safe}</span>'


def _tr_valence(v: str) -> str:
    m = {"NEGATIVE": "Düşük (ciddi/odaklı olabilir)", "NEUTRAL": "Nötr", "POSITIVE": "Yüksek"}
    return m.get((v or "").strip().upper(), (v or "").strip() or "-")


def _tr_arousal(a: str) -> str:
    m = {"LOW": "Düşük", "MEDIUM": "Orta", "HIGH": "Yüksek"}
    return m.get((a or "").strip().upper(), (a or "").strip() or "-")


def _tr_level(x: str) -> str:
    return _tr_arousal(x)


def _tr_rate(r: str) -> str:
    m = {"SLOW": "Yavaş", "NORMAL": "Normal", "FAST": "Hızlı"}
    return m.get((r or "").strip().upper(), (r or "").strip() or "-")


def _tr_pitch(p: str) -> str:
    m = {"STABLE": "Kararlı", "UNSTABLE": "Dalgalı"}
    return m.get((p or "").strip().upper(), (p or "").strip() or "-")


def _tr_facial(s: str) -> str:
    m = {"NEUTRAL": "Nötr", "POSITIVE": "Pozitif", "TENSE": "Gergin"}
    return m.get((s or "").strip().upper(), (s or "").strip() or "-")


def _tr_attention(a: str) -> str:
    m = {"FOCUSED": "Odaklı", "AVERTED": "Göz kaçırma/dalınma"}
    return m.get((a or "").strip().upper(), (a or "").strip() or "-")


def _tr_stress(s: str) -> str:
    m = {"LOW": "Düşük", "ELEVATED": "Yükselmiş", "HIGH": "Yüksek"}
    return m.get((s or "").strip().upper(), (s or "").strip() or "-")


def _sev_valence(v: str) -> str:
    u = (v or "").strip().upper()
    if u == "POSITIVE":
        return "good"
    if u == "NEGATIVE":
        return "warn"
    return "warn"


def _sev_arousal(a: str) -> str:
    u = (a or "").strip().upper()
    if u == "HIGH":
        return "warn"
    if u == "LOW":
        return "good"
    return "warn"


def _sev_level(x: str) -> str:
    return _sev_arousal(x)


def _sev_rate(r: str) -> str:
    u = (r or "").strip().upper()
    if u in ("NORMAL",):
        return "good"
    if u in ("FAST",):
        return "warn"
    if u in ("SLOW",):
        return "warn"
    return "warn"


def _sev_pitch(p: str) -> str:
    u = (p or "").strip().upper()
    return "good" if u == "STABLE" else "warn"


def _sev_facial(s: str) -> str:
    u = (s or "").strip().upper()
    if u == "TENSE":
        return "warn"
    if u == "POSITIVE":
        return "good"
    return "warn"


def _sev_attention(a: str) -> str:
    u = (a or "").strip().upper()
    return "good" if u == "FOCUSED" else "warn"


def _sev_stress(s: str) -> str:
    u = (s or "").strip().upper()
    if u == "HIGH":
        return "bad"
    if u == "ELEVATED":
        return "warn"
    return "good"


def _clamp_int(v: float, lo: int = 0, hi: int = 100) -> int:
    try:
        x = int(round(float(v)))
    except Exception:
        x = 0
    return max(lo, min(hi, x))


def _compute_v3_product_scores(report: Dict) -> Dict[str, int]:
    """
    Basit ürün KPI skorları:
      - confidence: odak + pitch stability + enerji
      - stress_control: stress_indicator dağılımı + blink_rate (uçlar)
      - communication: speech/silence ratio + speech_rate (normal tercih)
    """
    face_sum = report.get("face_analysis", {}).get("summary", {}) or {}
    voice = report.get("voice_analysis", {}).get("raw_voice_features", {}) or {}
    ss = voice.get("speech_silence", {}) or {}
    audio_sig = report.get("audio_signal_analysis", {}).get("summary", {}) or {}
    visual_tl = report.get("visual_signal_analysis", {}).get("timeline", []) or []
    audio_tl = report.get("audio_signal_analysis", {}).get("timeline", []) or []

    focus = float(face_sum.get("focus_score", 0) or 0)
    blink = float(face_sum.get("blink_rate_per_min", 0) or 0)
    ratio = float(ss.get("speech_silence_ratio", 0) or 0)

    # stress indicator distribution
    stress_vals = [v.get("stress_indicator") for v in visual_tl if isinstance(v, dict)]
    stress_high = sum(1 for s in stress_vals if s == "HIGH")
    stress_elev = sum(1 for s in stress_vals if s == "ELEVATED")
    total = len(stress_vals) if stress_vals else 1
    stress_score = 100.0 - ((stress_high * 2.0 + stress_elev * 1.0) / total) * 35.0  # lower is worse

    # pitch stability / energy from audio signal timeline
    pitch_unstable = sum(1 for a in audio_tl if isinstance(a, dict) and a.get("pitch_stability") == "UNSTABLE")
    energy_low = sum(1 for a in audio_tl if isinstance(a, dict) and a.get("speech_energy") == "LOW")
    a_total = len(audio_tl) if audio_tl else 1

    confidence = (0.55 * focus) + (0.25 * (100.0 - (pitch_unstable / a_total) * 100.0)) + (0.20 * (100.0 - (energy_low / a_total) * 100.0))

    # communication: ratio ideal ~ 3-7; penalize extremes
    if ratio <= 0:
        comm = 30.0
    else:
        # map ratio to 0..100 with peak at 5
        comm = 100.0 - min(70.0, abs(ratio - 5.0) * 12.0)
    # blink extremes penalize stress control
    blink_penalty = 0.0
    if blink > 35:
        blink_penalty = min(25.0, (blink - 35) * 1.0)
    if blink < 8 and blink > 0:
        blink_penalty = min(15.0, (8 - blink) * 1.5)

    stress_control = stress_score - blink_penalty

    return {
        "confidence": _clamp_int(confidence),
        "stress_control": _clamp_int(stress_control),
        "communication": _clamp_int(comm),
    }


# =====================================================================
# FAZ-3 AI metnini parse etme yardımcıları
# =====================================================================
def _normalize_heading(s: str) -> str:
    s = (s or "").strip().lower()
    # küçük normalize: emoji ve ekstra karakterleri temizlemeye çalış
    for ch in ["✅", "⚠️", "🤖", "🎯", "📊", "📌", "🔍", "🧠", "📈", "👁️", "🎤", "📝"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s


def _split_markdown_sections(text: str) -> Dict[str, str]:
    """
    LLM çıktısını '##' başlıklarına göre böler ve ana bölümleri döndürür.
    Beklenen (prompt_phase3.txt):
      - Yönetici Özeti
      - Sinyal Tutarlılığı ve Örüntüler
      - Kritik Anlar / Olası Red Flags
      - Soft Skill Karnesi (Olasılıksal)
      - Önerilen Takip Soruları (Doğrulama)
    """
    lines = (text or "").replace("\r\n", "\n").split("\n")
    current = None
    buf: Dict[str, List[str]] = {}

    def key_for_heading(h: str) -> Optional[str]:
        h2 = _normalize_heading(h)
        if "yönetici özeti" in h2 or "yonetici ozeti" in h2:
            return "yonetici_ozeti"
        if "sinyal tutarlılığı" in h2 or "orunt" in h2 or "örünt" in h2:
            return "oruntuler"
        if "kritik anlar" in h2 or "red flags" in h2 or "risk sinyali" in h2:
            return "kritik_anlar"
        if "soft skill" in h2 or "karnesi" in h2:
            return "soft_skill"
        if "takip soruları" in h2 or "dogrulama" in h2 or "doğrulama" in h2:
            return "takip_sorulari"
        return None

    for line in lines:
        if line.strip().startswith("##"):
            heading = line.strip().lstrip("#").strip()
            k = key_for_heading(heading)
            current = k
            if current and current not in buf:
                buf[current] = []
            continue
        if current:
            buf[current].append(line)

    return {k: "\n".join(v).strip() for k, v in buf.items()}


def _extract_bullets(section_text: str, max_items: int = 6) -> List[str]:
    """
    Basit bullet/numaralı satır çıkarımı.
    """
    items: List[str] = []
    for raw in (section_text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("-", "•", "*")):
            line = line.lstrip("-•*").strip()
        # "1) ..." veya "1. ..." gibi
        if len(line) >= 2 and (line[0].isdigit() and (line[1] in ".)")):
            line = line[2:].strip()
        if line:
            items.append(line)
        if len(items) >= max_items:
            break
    return items


def _escape_html(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_bullets_as_cards(section_text: str, max_items: int = 6) -> str:
    cards = []
    for item in _extract_bullets(section_text, max_items=max_items):
        safe = _escape_html(item).replace("\n", "<br>")
        cards.append(
            '<div class="card-mini">'
            '<h4>Olası kritik an</h4>'
            f'<div class="body">{safe}</div>'
            "</div>"
        )
    return "\n".join(cards)


def _render_bullets_as_list(section_text: str, max_items: int = 6) -> str:
    lis = []
    for item in _extract_bullets(section_text, max_items=max_items):
        safe = _escape_html(item)
        lis.append(f"<li>{safe}</li>")
    return "\n".join(lis)


def _parse_key_value_lines(section_text: str, max_items: int = 8) -> List[Dict[str, str]]:
    """
    Soft skill karnesi gibi 'Başlık: açıklama' satırlarını parse eder.
    Bullet/numaralı formatları da tolere eder.
    """
    rows: List[Dict[str, str]] = []
    for raw in (section_text or "").replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("-", "•", "*")):
            line = line.lstrip("-•*").strip()
        if len(line) >= 2 and (line[0].isdigit() and (line[1] in ".)")):
            line = line[2:].strip()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        rows.append({"key": key, "value": val})
        if len(rows) >= max_items:
            break
    return rows


def _render_soft_skill_cards(section_text: str, max_items: int = 8) -> str:
    cards = []
    rows = _parse_key_value_lines(section_text, max_items=max_items)
    for row in rows:
        k = _escape_html(row.get("key", ""))
        v = _escape_html(row.get("value", "")).replace("\n", "<br>")
        cards.append(
            '<div class="card-mini">'
            f"<h4>{k}</h4>"
            f'<div class="body">{v}</div>'
            "</div>"
        )
    return "\n".join(cards)


# =====================================================================
# FAZ-4 Dashboard Yardımcıları
# =====================================================================

def _compute_faz4_dashboard_data(segment_packages: List[Dict], face_timeline: List[Dict] = None) -> Dict:
    """3 davranışsal kart ve Chart.js veri dizilerini segment paketlerinden hesaplar."""
    _empty = {
        "gaze_label": "Veri Yok", "gaze_card_class": "warn",
        "gaze_focus_pct": "-%", "gaze_aversion_ts": "—",
        "speech_label": "Veri Yok", "speech_card_class": "warn",
        "avg_speech_confidence": "-", "conf_drop_ts": "—",
        "emotion_label": "Veri Yok", "emotion_card_class": "warn",
        "emotion_neg_pct": "—",
        "crit_count": 0, "crit_timestamps": "—",
        "chart_labels": [], "chart_emotion_colors": [],
        "chart_emotion_band": [], "chart_gaze": [],
        "chart_speech": [], "chart_critical": [],
    }
    if not segment_packages:
        return _empty

    n = len(segment_packages)

    # ── Kart 1: Göz Teması ──────────────────────────────────────────────
    # Primary: frame-level focus using offset-corrected gaze (more granular & reliable)
    # Fallback: segment-level gaze_direction from aggregator
    def _is_away(p: Dict) -> bool:
        if "gaze_away" in p:
            return bool(p["gaze_away"])
        return p.get("gaze_direction", "center") != "center"

    if face_timeline and len(face_timeline) >= 5:
        # Frame-level focus: offset-correct pitch/yaw with median, then threshold
        detected_frames = [f for f in face_timeline if f.get("face_detected", True)]
        if detected_frames:
            raw_pitches = sorted([float(f.get("gaze_pitch_deg") or 0.0) for f in detected_frames])
            raw_yaws    = sorted([float(f.get("gaze_yaw_deg") or 0.0) for f in detected_frames])
            p_median = raw_pitches[len(raw_pitches) // 2]
            y_median = raw_yaws[len(raw_yaws) // 2]
            focus_count = sum(
                1 for f in detected_frames
                if abs(float(f.get("gaze_pitch_deg") or 0.0) - p_median) <= 20.0
                and abs(float(f.get("gaze_yaw_deg") or 0.0) - y_median) <= 22.0
            )
            gaze_focus_num = round(focus_count / max(1, len(face_timeline)) * 100)
        else:
            gaze_focus_num = 0
    else:
        center_count = sum(1 for p in segment_packages if not _is_away(p))
        gaze_focus_num = round(center_count / n * 100)

    gaze_focus_pct = f"{gaze_focus_num}%"

    if gaze_focus_num > 70:
        gaze_label, gaze_card_class = "Yüksek", "accent2"
    elif gaze_focus_num >= 40:
        gaze_label, gaze_card_class = "Orta", "warn"
    else:
        gaze_label, gaze_card_class = "Düşük", "danger"

    # First sustained aversion: 3+ consecutive gaze_away segments
    gaze_aversion_ts = "—"
    run_start = None
    run_len = 0
    for i, p in enumerate(segment_packages):
        if _is_away(p):
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len == 3:
                ts_raw = segment_packages[run_start].get("timestamp", "")
                gaze_aversion_ts = ts_raw.split(" - ")[0] if ts_raw else "—"
                break
        else:
            run_start = None
            run_len = 0

    # ── Kart 2: Ses Güveni ──────────────────────────────────────────────
    conf_vals = [float(p.get("speech_confidence", 0)) for p in segment_packages]
    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0

    if avg_conf > 0.75:
        speech_label, speech_card_class = "Kararlı", "accent2"
    elif avg_conf >= 0.5:
        speech_label, speech_card_class = "Orta", "warn"
    else:
        speech_label, speech_card_class = "Zayıf", "danger"

    # Biggest single-step confidence drop
    conf_drop_ts = "—"
    if len(conf_vals) >= 2:
        max_drop = 0.0
        for i in range(1, len(conf_vals)):
            drop = conf_vals[i - 1] - conf_vals[i]
            if drop > max_drop:
                max_drop = drop
                ts_raw = segment_packages[i].get("timestamp", "")
                conf_drop_ts = ts_raw.split(" - ")[0] if ts_raw else "—"

    # ── Kart 3: Duygusal Denge ──────────────────────────────────────────
    # Use negative emotion % from face_timeline (confidence >= 0.55)
    _negative_emotions = {"Sad", "Fear", "Angry", "Disgust"}
    if face_timeline:
        valid_face = [
            f for f in face_timeline
            if f.get("face_detected", True)
            and float(f.get("emotion_confidence") or 0.0) >= 0.55
        ]
        if valid_face:
            neg_count = sum(
                1 for f in valid_face
                if (f.get("emotion_label") or "Neutral") in _negative_emotions
            )
            negative_pct = neg_count / len(valid_face) * 100.0
        else:
            negative_pct = 0.0
    else:
        negative_pct = 0.0

    if negative_pct > 60.0:
        emotion_label, emotion_card_class = "Gergin", "danger"
    elif negative_pct > 35.0:
        emotion_label, emotion_card_class = "Dikkat", "warn"
    else:
        emotion_label, emotion_card_class = "Dengeli", "accent2"

    # Keep crit_count for the sub-metric line
    critical_segs = [p for p in segment_packages if p.get("is_critical_moment", False)]
    crit_count = len(critical_segs)
    crit_tss = [p.get("timestamp", "").split(" - ")[0] for p in critical_segs[:5] if p.get("timestamp")]
    crit_timestamps = ", ".join(crit_tss) if crit_tss else "—"
    emotion_neg_pct = f"{negative_pct:.1f}%"

    # ── Chart.js veri dizileri ──────────────────────────────────────────
    _emotion_color = {
        "Happy":   "rgba(52,211,153,0.7)",
        "Surprise":"rgba(52,211,153,0.7)",
        "Neutral": "rgba(107,114,128,0.5)",
        "Sad":     "rgba(96,165,250,0.7)",
        "Fear":    "rgba(251,113,133,0.7)",
        "Angry":   "rgba(251,113,133,0.7)",
        "Disgust": "rgba(251,113,133,0.7)",
    }

    chart_labels         = [p.get("timestamp", f"{float(p.get('start',0)):.0f}s").split(" - ")[0]
                            for p in segment_packages]
    chart_emotion_colors = [_emotion_color.get(p.get("dominant_emotion", "Neutral"), "rgba(107,114,128,0.5)")
                            for p in segment_packages]
    chart_emotion_band   = [1.0] * n           # constant height; color comes from backgroundColor array
    chart_gaze           = [0 if _is_away(p) else 1 for p in segment_packages]
    chart_speech         = [round(float(p.get("speech_confidence", 0)), 2) for p in segment_packages]
    chart_critical       = [1 if p.get("is_critical_moment", False) else 0 for p in segment_packages]

    return {
        "gaze_label": gaze_label, "gaze_card_class": gaze_card_class,
        "gaze_focus_pct": gaze_focus_pct, "gaze_aversion_ts": gaze_aversion_ts,
        "speech_label": speech_label, "speech_card_class": speech_card_class,
        "avg_speech_confidence": f"{avg_conf:.2f}", "conf_drop_ts": conf_drop_ts,
        "emotion_label": emotion_label, "emotion_card_class": emotion_card_class,
        "emotion_neg_pct": emotion_neg_pct,
        "crit_count": crit_count, "crit_timestamps": crit_timestamps,
        "chart_labels": chart_labels,
        "chart_emotion_colors": chart_emotion_colors,
        "chart_emotion_band": chart_emotion_band,
        "chart_gaze": chart_gaze,
        "chart_speech": chart_speech,
        "chart_critical": chart_critical,
    }


def _compute_speech_stats(segments: List[Dict], video_duration: float) -> Dict:
    """
    Whisper segmentlerinden konuşma istatistiklerini hesaplar (Step F).

    Args:
        segments:       text_analysis.segments — [{start, end, text}, ...]
        video_duration: video_info.duration_seconds

    Returns dict with keys:
        speech_dur_str, silence_dur_str,
        long_silence_count, longest_silence_dur, longest_silence_ts,
        avg_words, pace_label
    """
    if not segments:
        return {
            "speech_dur_str": "—", "silence_dur_str": "—",
            "long_silence_count": 0,
            "longest_silence_dur": "—", "longest_silence_ts": "—",
            "avg_words": 0.0, "pace_label": "—",
        }

    def _fmt(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = int(round(sec - m * 60))
        if s >= 60:
            m += 1; s = 0
        return f"{m}dk {s:02d}sn"

    def _mmss(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = int(round(sec - m * 60))
        if s >= 60:
            m += 1; s = 0
        return f"{m:02d}:{s:02d}"

    # Sort segments by start time
    segs = sorted(segments, key=lambda s: float(s.get("start") or 0.0))

    # Total speech duration
    speech_dur = sum(
        max(0.0, float(s.get("end") or 0.0) - float(s.get("start") or 0.0))
        for s in segs
    )
    silence_dur = max(0.0, float(video_duration) - speech_dur)

    # Gaps between consecutive segments
    long_silence_count = 0
    longest_gap = 0.0
    longest_gap_ts = 0.0
    for i in range(1, len(segs)):
        gap_start = float(segs[i - 1].get("end") or 0.0)
        gap_end   = float(segs[i].get("start") or 0.0)
        gap = gap_end - gap_start
        if gap > longest_gap:
            longest_gap = gap
            longest_gap_ts = gap_start
        if gap > 5.0:
            long_silence_count += 1

    # Average words per segment
    word_counts = [
        len((s.get("text") or "").split())
        for s in segs if (s.get("text") or "").strip()
    ]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0

    pace_label = "Yavaş" if avg_words < 8 else ("Hızlı" if avg_words > 15 else "Normal")

    return {
        "speech_dur_str":     _fmt(speech_dur),
        "silence_dur_str":    _fmt(silence_dur),
        "long_silence_count": long_silence_count,
        "longest_silence_dur": f"{longest_gap:.1f}sn" if longest_gap > 0 else "—",
        "longest_silence_ts":  _mmss(longest_gap_ts) if longest_gap > 0 else "—",
        "avg_words":           round(avg_words, 1),
        "pace_label":          pace_label,
    }


_EMOTION_COLORS = {
    "Happy":   "rgba(52,211,153,0.85)",
    "Surprise":"rgba(52,211,153,0.85)",
    "Neutral": "rgba(107,114,128,0.65)",
    "Sad":     "rgba(96,165,250,0.85)",
    "Fear":    "rgba(251,113,133,0.85)",
    "Angry":   "rgba(251,113,133,0.85)",
    "Disgust": "rgba(251,113,133,0.85)",
}
_EMOTION_ORDER = ["Happy", "Surprise", "Neutral", "Sad", "Fear", "Angry", "Disgust"]
_EMOTION_CONF_THRESHOLD = 0.55


def _compute_emotion_distribution(face_timeline: List[Dict]) -> Dict:
    """
    Yüz timeline'ından güven filtreli duygu dağılımını hesaplar.

    Kurallar (Step D):
      - Sadece face_detected=True ve emotion_confidence >= 0.55 çerçeveler sayılır.
      - Geçerli çerçeve sayısı, toplam çerçevelerin %10'undan azsa → no_data=True.
      - Yalnızca %0'ın üzerinde duygu gösterilir.

    Döndürür:
      {
        "no_data": bool,
        "labels":  [str, ...],   # sadece pct > 0 olanlar, _EMOTION_ORDER sırasıyla
        "values":  [float, ...], # yüzde (0-100, 1 ondalık)
        "colors":  [str, ...],   # rgba stringleri
      }
    """
    _empty = {"no_data": True, "labels": [], "values": [], "colors": []}
    if not face_timeline:
        return _empty

    total = len(face_timeline)
    valid_frames = [
        f for f in face_timeline
        if f.get("face_detected", True)
        and float(f.get("emotion_confidence") or 0.0) >= _EMOTION_CONF_THRESHOLD
    ]
    if len(valid_frames) < max(1, total * 0.10):
        return _empty

    counts: Dict[str, int] = {}
    for f in valid_frames:
        label = (f.get("emotion_label") or "Neutral").strip()
        counts[label] = counts.get(label, 0) + 1

    n = len(valid_frames)
    labels, values, colors = [], [], []
    for emo in _EMOTION_ORDER:
        cnt = counts.get(emo, 0)
        if cnt == 0:
            continue
        pct = round(cnt / n * 100, 1)
        labels.append(emo)
        values.append(pct)
        colors.append(_EMOTION_COLORS.get(emo, "rgba(107,114,128,0.65)"))

    # Also include any unexpected labels not in _EMOTION_ORDER
    for emo, cnt in counts.items():
        if emo not in _EMOTION_ORDER:
            pct = round(cnt / n * 100, 1)
            labels.append(emo)
            values.append(pct)
            colors.append("rgba(107,114,128,0.65)")

    if not labels:
        return _empty
    return {"no_data": False, "labels": labels, "values": values, "colors": colors}


_VOICE_PROFILE_COLORS = {
    # torchaudio f0+enerji ses profili renkleri
    "canlı":    "rgba(52,211,153,0.85)",    # yeşil — enerjik ve pozitif
    "kararlı":  "rgba(96,165,250,0.85)",    # mavi — güçlü ve stabil
    "dengeli":  "rgba(107,114,128,0.65)",   # gri — nötr
    "sakin":    "rgba(147,197,253,0.65)",   # açık mavi — sakin
    "gergin":   "rgba(251,146,60,0.85)",    # turuncu — stresli
}
_VOICE_PROFILE_ORDER = ["canlı", "kararlı", "dengeli", "sakin", "gergin"]


def _compute_voice_emotion_distribution(audio_signal_timeline: List[Dict]) -> Dict:
    """
    Ses profili dağılımını hesaplar (torchaudio f0+enerji kural sistemi).

    debug.voice_profile alanını okur: "Canlı"|"Kararlı"|"Dengeli"|"Sakin"|"Gergin"
    Sessiz chunk'lar (voice_profile eksik) atlanır.

    Döndürür:
      {
        "no_data": bool,
        "labels":  [str, ...],   # capitalize, pct > 0 olanlar
        "values":  [float, ...], # yüzde (0-100, 1 ondalık)
        "colors":  [str, ...],
        "dominant_label": str,
        "dominant_pct":   float,
        "top2_str":       str,
      }
    """
    _empty = {
        "no_data": True, "labels": [], "values": [], "colors": [],
        "dominant_label": "—", "dominant_pct": 0.0, "top2_str": "—",
    }
    if not audio_signal_timeline:
        return _empty

    profile_counts: Dict[str, int] = {}
    valid_chunks = 0

    for chunk in audio_signal_timeline:
        dbg = chunk.get("debug", {})
        profile = (dbg.get("voice_profile") or "").strip()
        if not profile:
            continue
        key = profile.lower()
        profile_counts[key] = profile_counts.get(key, 0) + 1
        valid_chunks += 1

    if valid_chunks == 0 or not profile_counts:
        return _empty

    total = sum(profile_counts.values())
    if total == 0:
        return _empty

    labels, values, colors = [], [], []
    for profile in _VOICE_PROFILE_ORDER:
        count = profile_counts.get(profile, 0)
        if count == 0:
            continue
        pct = round(count / total * 100, 1)
        labels.append(profile.capitalize())
        values.append(pct)
        colors.append(_VOICE_PROFILE_COLORS.get(profile, "rgba(107,114,128,0.65)"))

    # Beklenmeyen profiller
    for profile, count in profile_counts.items():
        if profile not in _VOICE_PROFILE_ORDER and count > 0:
            pct = round(count / total * 100, 1)
            labels.append(profile.capitalize())
            values.append(pct)
            colors.append("rgba(107,114,128,0.65)")

    if not labels:
        return _empty

    sorted_pairs = sorted(zip(values, labels), reverse=True)
    dominant_label = sorted_pairs[0][1]
    dominant_pct = sorted_pairs[0][0]
    top2_str = " | ".join(f"{lbl} %{pct}" for pct, lbl in sorted_pairs[:2])

    return {
        "no_data": False, "labels": labels, "values": values, "colors": colors,
        "dominant_label": dominant_label, "dominant_pct": dominant_pct,
        "top2_str": top2_str,
    }


def _md_bold(s: str) -> str:
    """**metin** → <strong>metin</strong>"""
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)


def _format_ai_report_html(ai_text: str) -> str:
    """LLM markdown çıktısını temiz HTML'e dönüştürür (## başlıklar, - maddeler, paragraflar)."""
    if not ai_text:
        return "<p>Analiz metni mevcut değil.</p>"

    parts: List[str] = []
    in_list = False

    for raw in ai_text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("## "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            heading = _escape_html(stripped[3:].strip())
            parts.append(f'<h2 class="ai-section">{heading}</h2>')

        elif stripped.startswith("### "):
            if in_list:
                parts.append("</ul>")
                in_list = False
            heading = _escape_html(stripped[4:].strip())
            parts.append(f'<h3 class="ai-subsection">{heading}</h3>')

        elif stripped.startswith(("- ", "* ", "• ")):
            if not in_list:
                parts.append('<ul class="ai-list">')
                in_list = True
            item = _md_bold(_escape_html(stripped[2:].strip()))
            parts.append(f"<li>{item}</li>")

        elif stripped == "":
            if in_list:
                parts.append("</ul>")
                in_list = False

        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            paragraph = _md_bold(_escape_html(stripped))
            parts.append(f"<p>{paragraph}</p>")

    if in_list:
        parts.append("</ul>")

    return "\n".join(parts)


if __name__ == "__main__":
    print("ReportGenerator modülü hazır.")
