"""
Rapor Oluşturucu
HTML + JSON rapor üretir, matplotlib grafikleri embed eder.
"""

import os
import json
import math
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
from jinja2 import Template
from .plot import generate_report_charts


class ReportGenerator:
    """HTML ve JSON raporları oluşturur."""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

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

        template = Template(HTML_TEMPLATE_V3 if is_phase3 else HTML_TEMPLATE)
        html = template.render(
            interview_id=interview_id,
            phase=phase,
            is_phase3=is_phase3,
            analysis_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            duration_seconds=f"{duration:.2f}",
            video_fps=video_info.get("fps", 0),
            video_resolution=f'{video_info.get("width", 0)}x{video_info.get("height", 0)}',
            video_duration=f'{video_info.get("duration_seconds", 0):.1f}',
            # Yüz
            focus_score=f'{face_summary.get("focus_score", 0):.1f}',
            blink_rate=f'{face_summary.get("blink_rate_per_min", 0):.1f}',
            dominant_face_emotion=face_summary.get("dominant_emotion", "?"),
            total_blinks=face_summary.get("total_blinks", 0),
            # FAZ-3 sinyaller
            dominant_valence=audio_signal_summary.get("dominant_valence", "?"),
            dominant_arousal=audio_signal_summary.get("dominant_arousal", "?"),
            dominant_facial_state=face_summary.get("dominant_emotion", "?"),
            # Metin
            total_sentences=text_summary.get("total_sentences", 0),
            dominant_sentiment=text_summary.get("dominant_sentiment", "?"),
            positive_pct=text_summary.get("sentiment_percentages", {}).get("positive", 0),
            negative_pct=text_summary.get("sentiment_percentages", {}).get("negative", 0),
            # Ses duygu
            dominant_audio_emotion=audio_summary.get("dominant_emotion", "?"),
            audio_avg_confidence=f'{audio_summary.get("avg_confidence", 0):.2f}',
            audio_total_chunks=audio_summary.get("total_chunks", 0),
            # Ses özellikleri
            speech_seconds=f'{ss.get("total_speech_seconds", 0):.1f}',
            silence_seconds=f'{ss.get("total_silence_seconds", 0):.1f}',
            speech_ratio=f'{ss.get("speech_silence_ratio", 0):.2f}',
            avg_silence=f'{ss.get("average_silence_seconds", 0):.2f}',
            rms_mean=f'{voice_raw.get("energy_rms", {}).get("mean", 0):.4f}',
            pitch_mean=f'{voice_raw.get("pitch_f0", {}).get("mean", 0):.1f}',
            pitch_var=f'{voice_raw.get("pitch_f0", {}).get("variability", 0):.1f}',
            # Anomaliler
            anomaly_count=len(anomalies),
            anomaly_rows=anomaly_rows,
            # Metin tablosu
            text_table_rows=text_table_rows,
            text_table_phase3=is_phase3,
            # Grafikler
            charts_html=charts_html,
            # AI
            ai_hr_text=formatted_ai,
            # FAZ-3 parsed AI sections (dashboard)
            critical_cards_html=critical_cards_html,
            exec_summary_html=exec_summary_html,
            soft_skill_cards_html=soft_skill_cards_html,
            thought_rows=thought_rows,
            thought_count=len(thought_units) if thought_units else 0,
            v3_confidence=v3_scores.get("confidence", 50),
            v3_stress_control=v3_scores.get("stress_control", 50),
            v3_communication=v3_scores.get("communication", 50),
            # FAZ-3 explainability table
            segment_rows=segment_rows,
            segment_count=len(segment_packages) if segment_packages else 0,
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
# HTML TEMPLATE
# =====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mülakat Analiz Raporu - {{ interview_id }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            color: #333;
            background: #f0f2f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            border-radius: 12px;
        }
        .header {
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            font-size: 2.2em;
            margin-bottom: 5px;
        }
        .header .subtitle {
            color: #7f8c8d;
            font-size: 1.1em;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .meta-item {
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
        }
        .section {
            margin-bottom: 35px;
        }
        .section h2 {
            color: #2c3e50;
            font-size: 1.6em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }
        .section h3 {
            color: #34495e;
            font-size: 1.2em;
            margin-top: 15px;
            margin-bottom: 8px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
        }
        .stat-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .stat-card.orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
        .stat-card.red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
        .stat-card.blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
        .stat-card .label {
            font-size: 0.85em;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .stat-card .value {
            font-size: 1.8em;
            font-weight: bold;
        }
        .chart-container {
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }
        th {
            background: #2c3e50;
            color: white;
            font-weight: 600;
        }
        tr:hover { background: #f8f9fa; }
        .anomaly-badge {
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .ai-analysis {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            margin-top: 20px;
        }
        .ai-analysis p {
            margin-bottom: 12px;
            text-align: justify;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.85em;
        }
    </style>
</head>
<body>
<div class="container">

    <!-- HEADER -->
    <div class="header">
        <h1>🎯 SensifyHR Mülakat Analiz Raporu</h1>
        <p class="subtitle">Multimodal AI Mülakat Değerlendirme Sistemi</p>
        <div class="meta-grid">
            <div class="meta-item"><strong>Mülakat ID:</strong> {{ interview_id }}</div>
            <div class="meta-item"><strong>Tarih:</strong> {{ analysis_date }}</div>
            <div class="meta-item"><strong>Analiz Süresi:</strong> {{ duration_seconds }}s</div>
            <div class="meta-item"><strong>Video:</strong> {{ video_resolution }} @ {{ video_fps }} FPS</div>
            <div class="meta-item"><strong>Video Süresi:</strong> {{ video_duration }}s</div>
        </div>
    </div>

    <!-- KPI KARTLARI -->
    <div class="section">
        <h2>📊 Genel Özet</h2>
        <div class="stats-grid">
            <div class="stat-card green">
                <div class="label">👁️ Odak Skoru</div>
                <div class="value">%{{ focus_score }}</div>
            </div>
            <div class="stat-card blue">
                <div class="label">👁️ Göz Kırpma/dk</div>
                <div class="value">{{ blink_rate }}</div>
            </div>
            <div class="stat-card">
                <div class="label">😊 Baskın Yüz Duygusu</div>
                <div class="value">{{ dominant_face_emotion }}</div>
            </div>
            <div class="stat-card orange">
                <div class="label">🎤 Baskın Ses Duygusu</div>
                <div class="value">{{ dominant_audio_emotion }}</div>
            </div>
            <div class="stat-card green">
                <div class="label">📝 Baskın Metin Duygusu</div>
                <div class="value">{{ dominant_sentiment }}</div>
            </div>
            <div class="stat-card red">
                <div class="label">⚠️ Tutarsızlık</div>
                <div class="value">{{ anomaly_count }}</div>
            </div>
        </div>
    </div>

    <!-- YÜZ ANALİZİ -->
    <div class="section">
        <h2>😊 Yüz Analizi (MediaPipe)</h2>
        <div class="stats-grid">
            <div class="stat-card green">
                <div class="label">Baskın Duygu</div>
                <div class="value">{{ dominant_face_emotion }}</div>
            </div>
            <div class="stat-card blue">
                <div class="label">Toplam Göz Kırpma</div>
                <div class="value">{{ total_blinks }}</div>
            </div>
            <div class="stat-card">
                <div class="label">Odak Skoru</div>
                <div class="value">%{{ focus_score }}</div>
            </div>
        </div>
    </div>

    <!-- SES DUYGU ANALİZİ -->
    <div class="section">
        <h2>🎤 Ses Duygu Analizi (HuBERT SER)</h2>
        <div class="stats-grid">
            <div class="stat-card orange">
                <div class="label">Baskın Duygu</div>
                <div class="value">{{ dominant_audio_emotion }}</div>
            </div>
            <div class="stat-card blue">
                <div class="label">Ort. Güven</div>
                <div class="value">{{ audio_avg_confidence }}</div>
            </div>
            <div class="stat-card">
                <div class="label">Analiz Parçası</div>
                <div class="value">{{ audio_total_chunks }}</div>
            </div>
        </div>
    </div>

    <!-- METİN ANALİZİ -->
    <div class="section">
        <h2>📝 Metin Analizi (Whisper + BERT)</h2>
        <div class="stats-grid">
            <div class="stat-card green">
                <div class="label">Toplam Cümle</div>
                <div class="value">{{ total_sentences }}</div>
            </div>
            <div class="stat-card green">
                <div class="label">Pozitif</div>
                <div class="value">%{{ positive_pct }}</div>
            </div>
            <div class="stat-card red">
                <div class="label">Negatif</div>
                <div class="value">%{{ negative_pct }}</div>
            </div>
        </div>

        {% if text_table_rows %}
        <h3>Cümle Detayları</h3>
        <table>
            <thead>
                <tr><th>Zaman</th><th>Cümle</th><th>Duygu</th><th>Güven</th></tr>
            </thead>
            <tbody>
                {{ text_table_rows | safe }}
            </tbody>
        </table>
        {% endif %}
    </div>

    <!-- SES ÖZELLİKLERİ -->
    <div class="section">
        <h2>🔊 Ses Özellikleri (Librosa)</h2>
        <div class="stats-grid">
            <div class="stat-card green">
                <div class="label">Konuşma Süresi</div>
                <div class="value">{{ speech_seconds }}s</div>
            </div>
            <div class="stat-card red">
                <div class="label">Sessizlik Süresi</div>
                <div class="value">{{ silence_seconds }}s</div>
            </div>
            <div class="stat-card blue">
                <div class="label">Konuşma/Sessizlik</div>
                <div class="value">{{ speech_ratio }}</div>
            </div>
            <div class="stat-card orange">
                <div class="label">Ort. Duraklama</div>
                <div class="value">{{ avg_silence }}s</div>
            </div>
            <div class="stat-card">
                <div class="label">RMS Enerji (Ort)</div>
                <div class="value">{{ rms_mean }}</div>
            </div>
            <div class="stat-card blue">
                <div class="label">Pitch (Ort)</div>
                <div class="value">{{ pitch_mean }} Hz</div>
            </div>
        </div>
    </div>

    <!-- TUTARSIZLIK -->
    {% if anomaly_count > 0 %}
    <div class="section">
        <h2>⚠️ Tutarsızlık Analizi</h2>
        <p><span class="anomaly-badge">{{ anomaly_count }} tutarsızlık tespit edildi</span></p>
        <table>
            <thead>
                <tr><th>Zaman</th><th>Cümle</th><th>Metin Duygusu</th><th>Yüz Duygusu</th><th>Sonuç</th></tr>
            </thead>
            <tbody>
                {{ anomaly_rows | safe }}
            </tbody>
        </table>
    </div>
    {% else %}
    <div class="section">
        <h2>✅ Tutarsızlık Analizi</h2>
        <p>Metin ve yüz duyguları arasında anlamlı bir tutarsızlık tespit edilmedi.</p>
    </div>
    {% endif %}

    <!-- GRAFİKLER -->
    <div class="section">
        <h2>📈 Görselleştirmeler</h2>
        {{ charts_html | safe }}
    </div>

    <!-- AI DEĞERLENDİRME -->
    {% if ai_hr_text %}
    <div class="section">
        <h2>🤖 AI Destekli İK Değerlendirmesi</h2>
        <div class="ai-analysis">
            <p>{{ ai_hr_text | safe }}</p>
        </div>
    </div>
    {% endif %}

    <!-- FOOTER -->
    <div class="footer">
        <p>🎯 SensifyHR Mülakat Analiz Sistemi v2.0 - {{ analysis_date }}</p>
        <p>Bu rapor yapay zekâ destekli analiz sonuçlarına dayalıdır ve karar destek amaçlıdır.</p>
    </div>

</div>
</body>
</html>
"""


# =====================================================================
# HTML TEMPLATE (FAZ-3 / Product Mode)
# =====================================================================
HTML_TEMPLATE_V3 = """
<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SensifyHR v3.0 Raporu - {{ interview_id }}</title>
  <style>
    :root{
      --bg:#0b1220;
      --card:#0f1a33;
      --card2:#0c152b;
      --text:#e8eefc;
      --muted:#a9b6d3;
      --accent:#7c5cff;
      --accent2:#34d399;
      --danger:#fb7185;
      --warn:#fbbf24;
      --border:rgba(255,255,255,.08);
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Segoe UI Emoji";
      background: radial-gradient(1200px 800px at 10% 10%, rgba(124,92,255,.18), transparent 60%),
                  radial-gradient(1000px 700px at 90% 20%, rgba(52,211,153,.14), transparent 55%),
                  var(--bg);
      color:var(--text);
      padding:28px;
    }
    .container{max-width:1200px;margin:0 auto}
    .header{
      display:flex; align-items:flex-start; justify-content:space-between; gap:16px;
      margin-bottom:18px;
    }
    .title h1{margin:0;font-size:26px;letter-spacing:.2px}
    .title p{margin:6px 0 0;color:var(--muted)}
    .badge{
      display:inline-flex;align-items:center;gap:8px;
      padding:8px 12px;border:1px solid var(--border); border-radius:999px;
      background:rgba(255,255,255,.03); color:var(--muted); font-size:13px;
    }
    .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
    .card{
      background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
      border:1px solid var(--border);
      border-radius:14px;
      padding:16px;
      box-shadow:0 10px 30px rgba(0,0,0,.25);
    }
    .kpi{display:flex;flex-direction:column;gap:8px}
    .kpi .label{color:var(--muted);font-size:12px}
    .kpi .value{font-size:22px;font-weight:700}
    .kpi .hint{color:var(--muted);font-size:12px}
    .pill{display:inline-block;padding:4px 10px;border-radius:999px;border:1px solid var(--border);font-size:12px;color:var(--muted)}
    .pill.ok{color:#b7f7d6;border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.08)}
    .pill.warn{color:#fde68a;border-color:rgba(251,191,36,.35);background:rgba(251,191,36,.08)}
    .pill.bad{color:#fecdd3;border-color:rgba(251,113,133,.35);background:rgba(251,113,133,.08)}

    /* Tabs (no JS) */
    .tabs{margin-top:10px}
    .tabs input{display:none}
    .tab-labels{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
    .tab-labels label{
      cursor:pointer;
      padding:10px 12px;border-radius:10px;border:1px solid var(--border);
      background:rgba(255,255,255,.03); color:var(--muted);
      font-size:13px; user-select:none;
    }
    #tab1:checked ~ .tab-labels label[for="tab1"],
    #tab2:checked ~ .tab-labels label[for="tab2"],
    #tab3:checked ~ .tab-labels label[for="tab3"]{
      color:var(--text);
      border-color:rgba(124,92,255,.5);
      background:rgba(124,92,255,.14);
    }
    .tab-content{display:none}
    #tab1:checked ~ .contents #content1{display:block}
    #tab2:checked ~ .contents #content2{display:block}
    #tab3:checked ~ .contents #content3{display:block}

    h2{margin:0 0 10px;font-size:16px}
    h3{margin:14px 0 8px;font-size:14px;color:var(--muted);font-weight:600}
    .muted{color:var(--muted)}
    table{width:100%;border-collapse:collapse;font-size:13px}
    th,td{padding:10px 10px;border-bottom:1px solid rgba(255,255,255,.06);vertical-align:top}
    th{color:var(--muted);font-weight:600;text-align:left}
    tr:hover td{background:rgba(255,255,255,.03)}
    .ai{
      background:linear-gradient(180deg, rgba(124,92,255,.10), rgba(255,255,255,.02));
      border:1px solid rgba(124,92,255,.25);
    }
    .cards{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
    .card-mini{
      grid-column:span 6;
      padding:14px;
      border-radius:14px;
      border:1px solid var(--border);
      background:rgba(255,255,255,.03);
    }
    .card-mini h4{margin:0 0 8px;font-size:13px;color:var(--muted);font-weight:700}
    .card-mini .body{font-size:13px;line-height:1.65;color:var(--text)}
    .list{margin:0;padding-left:18px}
    .list li{margin:6px 0;color:var(--text)}
    .footer{margin-top:16px;color:var(--muted);font-size:12px}
    .chart-container{margin:12px 0}
    .chart-container img{border-radius:12px;border:1px solid var(--border)}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace}
    .badge{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid var(--border);background:rgba(255,255,255,.03);font-size:12px;white-space:nowrap}
    .dot{width:10px;height:10px;border-radius:999px;display:inline-block}
    .sev-good .dot{background:#2ecc71}
    .sev-warn .dot{background:#f1c40f}
    .sev-bad  .dot{background:#e74c3c}

    /* Ring KPI */
    .rings{display:flex;gap:14px;flex-wrap:wrap}
    .ring{
      width:160px;height:160px;border-radius:999px;
      display:grid;place-items:center;
      background:conic-gradient(var(--accent) calc(var(--p) * 1%), rgba(255,255,255,.06) 0);
      border:1px solid var(--border);
      position:relative;
    }
    .ring::after{
      content:"";
      position:absolute; inset:14px;
      background:linear-gradient(180deg, rgba(0,0,0,.35), rgba(255,255,255,.02));
      border-radius:999px;
      border:1px solid rgba(255,255,255,.06);
    }
    .ring .inner{position:relative; z-index:2; text-align:center}
    .ring .pct{font-size:28px;font-weight:800}
    .ring .name{font-size:12px;color:var(--muted);margin-top:4px}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">
        <h1>SensifyHR v3.0 – Davranışsal Sinyal Raporu</h1>
        <p class="muted">Mülakat ID: <span class="mono">{{ interview_id }}</span> • Süre: {{ duration_seconds }}s • Tarih: {{ analysis_date }}</p>
      </div>
      <div class="badge">
        <span>Phase:</span>
        <strong>v3</strong>
      </div>
    </div>

    <div class="tabs">
      <input type="radio" id="tab1" name="tabs" checked>
      <input type="radio" id="tab2" name="tabs">
      <input type="radio" id="tab3" name="tabs">

      <div class="tab-labels">
        <label for="tab1">Dashboard</label>
        <label for="tab2">LLM Değerlendirme</label>
        <label for="tab3">Detaylar</label>
      </div>

      <div class="contents">
        <!-- DASHBOARD -->
        <div id="content1" class="tab-content">
          <div class="grid">
            <div class="card kpi" style="grid-column: span 3;">
              <div class="label">Odak Skoru</div>
              <div class="value">%{{ focus_score }}</div>
              <div class="hint">Gaze üzerinden FOCUSED/AVERTED türetilir</div>
            </div>
            <div class="card kpi" style="grid-column: span 3;">
              <div class="label">Göz Kırpma / dk</div>
              <div class="value">{{ blink_rate }}</div>
              <div class="hint">Aşırı uçlar stres/yorgunlukla ilişkili olabilir</div>
            </div>
            <div class="card kpi" style="grid-column: span 3;">
              <div class="label">Baskın Valence</div>
              <div class="value">{{ dominant_valence }}</div>
              <div class="hint">SER projeksiyonu (zayıf sinyal) + fiziksel metriklerle birlikte yorumlanır</div>
            </div>
            <div class="card kpi" style="grid-column: span 3;">
              <div class="label">Baskın Arousal</div>
              <div class="value">{{ dominant_arousal }}</div>
              <div class="hint">Konuşma enerjisi/hız/pitch stabilitesi ile birlikte değerlendirilir</div>
            </div>

            <div class="card" style="grid-column: span 12;">
              <h2>Öne Çıkan Sinyal Özetleri</h2>
              <div class="muted" style="margin-bottom:10px">
                Bu fazda “duygu etiketi” üretilmez. Aşağıdaki alanlar karar destek amaçlı sinyallerdir.
              </div>
              <div style="display:flex;gap:10px;flex-wrap:wrap">
                <span class="pill">Facial state (dominant): <strong>{{ dominant_facial_state }}</strong></span>
                <span class="pill">Konuşma/Sessizlik: <strong>{{ speech_ratio }}</strong></span>
                <span class="pill">Ort. Duraklama: <strong>{{ avg_silence }}s</strong></span>
                <span class="pill">RMS (ort): <strong>{{ rms_mean }}</strong></span>
                <span class="pill">Pitch (ort): <strong>{{ pitch_mean }} Hz</strong></span>
              </div>
            </div>

            <div class="card" style="grid-column: span 12;">
              <h2>Ürün KPI’ları (Heuristik • v3)</h2>
              <div class="muted" style="margin-bottom:10px">Bu skorlar kesin hüküm değildir; sinyallere dayalı hızlı özet göstergelerdir.</div>
              <div class="rings">
                <div class="ring" style="--p: {{ v3_confidence }};">
                  <div class="inner">
                    <div class="pct">{{ v3_confidence }}%</div>
                    <div class="name">Özgüven</div>
                  </div>
                </div>
                <div class="ring" style="--p: {{ v3_stress_control }}; background:conic-gradient(var(--accent2) calc(var(--p) * 1%), rgba(255,255,255,.06) 0);">
                  <div class="inner">
                    <div class="pct">{{ v3_stress_control }}%</div>
                    <div class="name">Stres Kontrolü</div>
                  </div>
                </div>
                <div class="ring" style="--p: {{ v3_communication }}; background:conic-gradient(var(--warn) calc(var(--p) * 1%), rgba(255,255,255,.06) 0);">
                  <div class="inner">
                    <div class="pct">{{ v3_communication }}%</div>
                    <div class="name">İletişim Akıcılığı</div>
                  </div>
                </div>
              </div>
            </div>

            <div class="card" style="grid-column: span 12;">
              <h2>Yönetici Özeti (LLM)</h2>
              {% if exec_summary_html %}
                <ul class="list">{{ exec_summary_html | safe }}</ul>
              {% else %}
                <div class="muted">LLM yönetici özeti bulunamadı (llm_provider=none veya çıktı formatı farklı).</div>
              {% endif %}
            </div>

            <div class="card" style="grid-column: span 12;">
              <h2>Kritik Anlar / Olası Risk Sinyalleri (LLM)</h2>
              <div class="muted" style="margin-bottom:10px">Bu bölüm olasılıksal yorum içerir; kesin hüküm değildir.</div>
              {% if critical_cards_html %}
                <div class="cards">{{ critical_cards_html | safe }}</div>
              {% else %}
                <div class="muted">Kritik anlar bölümü bulunamadı veya boş.</div>
              {% endif %}
            </div>

            <div class="card" style="grid-column: span 12;">
              <h2>Soft Skill Karnesi (LLM • olasılıksal)</h2>
              <div class="muted" style="margin-bottom:10px">Bu değerlendirme sinyal tabanlıdır; kesin hüküm değildir.</div>
              {% if soft_skill_cards_html %}
                <div class="cards">{{ soft_skill_cards_html | safe }}</div>
              {% else %}
                <div class="muted">Soft skill bölümü bulunamadı veya boş.</div>
              {% endif %}
            </div>

            <div class="card" style="grid-column: span 12;">
              <h2>Konu Blokları (Thought Units)</h2>
              <div class="muted" style="margin-bottom:10px">Aşırı kısa STT parçaları birleştirilerek HR için okunabilir bloklar üretilir. Toplam blok: <strong>{{ thought_count }}</strong>.</div>
              {% if thought_rows %}
              <div style="overflow:auto">
                <table>
                  <thead>
                    <tr><th>Zaman</th><th>Metin (blok)</th></tr>
                  </thead>
                  <tbody>
                    {{ thought_rows | safe }}
                  </tbody>
                </table>
              </div>
              {% else %}
              <div class="muted">Konu blokları bulunamadı.</div>
              {% endif %}
            </div>
          </div>
        </div>

        <!-- LLM -->
        <div id="content2" class="tab-content">
          <div class="card ai">
            <h2>AI Destekli İK Değerlendirmesi (Signal Reasoning)</h2>
            {% if ai_hr_text %}
              <div style="line-height:1.75">
                <p>{{ ai_hr_text | safe }}</p>
              </div>
            {% else %}
              <div class="muted">LLM çıktısı yok (llm_provider=none veya bağlantı/hata).</div>
            {% endif %}
          </div>
          <div class="footer">
            Not: Bu çıktı karar destek amaçlıdır; kesin hüküm veya klinik çıkarım içermez.
          </div>
        </div>

        <!-- DETAILS -->
        <div id="content3" class="tab-content">
          <div class="card">
            <h2>Detaylar (Teknik Grafikler)</h2>
            <div class="muted">Grafikler “Detaylar” sekmesinde tutulur; ana ekranda HR odaklı özet önceliklidir.</div>
            {{ charts_html | safe }}
          </div>
          <div class="card" style="margin-top:14px">
            <h2>Segment Bazlı Sinyal Tablosu (Explainability)</h2>
            <div class="muted" style="margin-bottom:10px">
              İlk 30 paket gösterilir. Toplam paket: <strong>{{ segment_count }}</strong>.
            </div>
            {% if segment_rows %}
            <div style="overflow:auto">
              <table>
                <thead>
                  <tr>
                    <th>Zaman Aralığı</th>
                    <th>Metin</th>
                    <th>Duygusal Ton<br><span class="muted">(Olumluluk)</span></th>
                    <th>Enerji/Heyecan</th>
                    <th>Konuşma Enerjisi</th>
                    <th>Konuşma Hızı</th>
                    <th>Ses Perdesi<br><span class="muted">(Kararlılık)</span></th>
                    <th>Yüz İfadesi</th>
                    <th>Odak Durumu</th>
                    <th>Stres Sinyali</th>
                  </tr>
                </thead>
                <tbody>
                  {{ segment_rows | safe }}
                </tbody>
              </table>
            </div>
            {% else %}
            <div class="muted">Segment paketleri bulunamadı.</div>
            {% endif %}
          </div>
          <div class="footer">
            Video: {{ video_resolution }} @ {{ video_fps }} FPS • Video Süresi: {{ video_duration }}s
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


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


if __name__ == "__main__":
    print("ReportGenerator modülü hazır.")
