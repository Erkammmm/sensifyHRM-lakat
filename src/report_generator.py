"""
Rapor Oluşturucu
Görselleştirme (Plotly) ve HTML/PDF rapor oluşturma.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import base64
from jinja2 import Template
from xhtml2pdf import pisa
from .plot import generate_report_charts


class ReportGenerator:
    """
    Görselleştirme ve HTML/PDF rapor oluşturur.
    """

    def __init__(self, reports_dir: str = "reports"):
        """
        Args:
            reports_dir: Raporların kaydedileceği klasör
        """
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate_report(
        self,
        interview_id: str,
        frame_analysis: List[Dict],
        frame_summary: Dict,
        voice_analysis: Dict,
        ai_analysis: Dict,
        video_info: Dict,
        duration_seconds: float,
        pyfeat_summary: Optional[Dict] = None,
        pyfeat_frame_analysis: Optional[List[Dict]] = None,
    ) -> Dict[str, str]:
        """
        Tam rapor oluşturur (JSON, HTML, PDF).

        Args:
            interview_id: Mülakat ID
            frame_analysis: Ham frame analizi (grafik için)
            frame_summary: Frame özeti
            voice_analysis: Ses analizi
            ai_analysis: Gemini AI analizi
            pyfeat_summary: Py-Feat özet (opsiyonel)
            pyfeat_frame_analysis: Py-Feat frame analizi (opsiyonel)
            video_info: Video bilgileri
            duration_seconds: Video süresi

        Returns:
            Oluşturulan dosya yolları: {"json": "...", "html": "...", "pdf": "..."}
        """
        # 1. Grafikleri oluştur
        charts_html = self._create_charts(
            frame_analysis=frame_analysis,
            frame_summary=frame_summary,
            voice_analysis=voice_analysis,
            pyfeat_summary=pyfeat_summary,
            pyfeat_frame_analysis=pyfeat_frame_analysis,
            video_info=video_info,
            duration_seconds=duration_seconds,
        )

        # 2. HTML raporu oluştur
        html_path = self._create_html_report(
            interview_id=interview_id,
            frame_summary=frame_summary,
            voice_analysis=voice_analysis,
            ai_analysis=ai_analysis,
            video_info=video_info,
            duration_seconds=duration_seconds,
            charts_html=charts_html,
        )

        # 3. PDF'e dönüştür
        pdf_path = self._html_to_pdf(html_path, interview_id)

        # 4. JSON raporu kaydet (rafine versiyon)
        json_path = self._save_json_report(
            interview_id=interview_id,
            frame_summary=frame_summary,
            voice_analysis=voice_analysis,
            ai_analysis=ai_analysis,
            pyfeat_summary=pyfeat_summary,
            video_info=video_info,
            duration_seconds=duration_seconds,
        )

        return {
            "json": json_path,
            "html": html_path,
            "pdf": pdf_path,
        }

    def _create_charts(
        self,
        frame_analysis: List[Dict],
        frame_summary: Dict,
        voice_analysis: Dict,
        pyfeat_summary: Optional[Dict],
        pyfeat_frame_analysis: Optional[List[Dict]],
        video_info: Dict,
        duration_seconds: float,
    ) -> str:
        """Matplotlib/Seaborn grafiklerini üretir ve HTML string olarak döndürür."""
        charts_dir = os.path.join(self.reports_dir, "charts")
        charts = generate_report_charts(
            frame_analysis=frame_analysis,
            frame_summary=frame_summary,
            voice_analysis=voice_analysis,
            pyfeat_summary=pyfeat_summary,
            pyfeat_frame_analysis=pyfeat_frame_analysis,
            video_info=video_info,
            duration_seconds=duration_seconds,
            output_dir=charts_dir,
        )

        if not charts:
            return ""

        html_blocks = []
        for idx, chart in enumerate(charts):
            chart_path = chart.get("path", "")
            if not chart_path or not os.path.exists(chart_path):
                continue
            with open(chart_path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            title = chart.get("title", f"Chart {idx + 1}")
            html_blocks.append(f"<h3>{title}</h3>")
            html_blocks.append(
                f"<div class=\"chart-container\"><img src=\"data:image/png;base64,{img_b64}\" style=\"width:100%;\" /></div>"
            )

        return "\n".join(html_blocks)

    def _create_html_report(
        self,
        interview_id: str,
        frame_summary: Dict,
        voice_analysis: Dict,
        ai_analysis: Dict,
        video_info: Dict,
        duration_seconds: float,
        charts_html: str,
    ) -> str:
        """HTML raporu oluşturur."""
        
        # Jinja2 template
        html_template = """
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
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header .meta {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }
        .section h3 {
            color: #34495e;
            font-size: 1.3em;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-card .label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .stat-card .value {
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: bold;
        }
        .chart-container {
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .ai-analysis {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            margin-top: 20px;
        }
        .ai-analysis p {
            margin-bottom: 15px;
            text-align: justify;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Mülakat Analiz Raporu</h1>
            <div class="meta">
                <p><strong>Mülakat ID:</strong> {{ interview_id }}</p>
                <p><strong>Analiz Tarihi:</strong> {{ analysis_date }}</p>
                <p><strong>Video Süresi:</strong> {{ duration_seconds }} saniye</p>
            </div>
        </div>

        <div class="section">
            <h2>Genel İstatistikler</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">Göz Merkeziyet</div>
                    <div class="value">{{ gaze_center_percentage }}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Konuşma Süresi</div>
                    <div class="value">{{ speaking_percentage }}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Thinking/Reading</div>
                    <div class="value">{{ thinking_reading_percentage }}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Speech Rate</div>
                    <div class="value">{{ speech_rate }}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Görselleştirmeler</h2>
            {{ charts_html | safe }}
        </div>

        {% if ai_hr_text %}
        <div class="section">
            <h2>AI Destekli İK Değerlendirmesi</h2>
            <div class="ai-analysis">
                <p>{{ ai_hr_text }}</p>
            </div>
        </div>
        {% endif %}

        <div class="footer">
            <p>SensifyHR Mülakat Analiz Sistemi - {{ analysis_date }}</p>
        </div>
    </div>
</body>
</html>
        """

        # Verileri hazırla
        gen_stats = frame_summary.get("general_statistics", {})
        cog_load = frame_summary.get("cognitive_load_score", {})
        voice_data = voice_analysis.get("raw_voice_features", voice_analysis)

        def _escape_html(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        ai_text = ""
        if isinstance(ai_analysis, dict):
            ai_text = str(ai_analysis.get("analysis", "")).strip() if ai_analysis else ""
        else:
            ai_text = str(ai_analysis).strip() if ai_analysis else ""

        template = Template(html_template)
        html_content = template.render(
            interview_id=interview_id,
            analysis_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            duration_seconds=f"{duration_seconds:.2f}",
            gaze_center_percentage=f"{gen_stats.get('gaze_center_percentage', 0.0):.1f}",
            speaking_percentage=f"{cog_load.get('speaking_percentage', 0.0):.1f}",
            thinking_reading_percentage=f"{cog_load.get('thinking_reading_percentage', 0.0):.1f}",
            speech_rate=f"{voice_data.get('speech_rate', {}).get('value', 0.0):.2f}",
            charts_html=charts_html,
            ai_hr_text=_escape_html(ai_text) if ai_text else "",
        )

        # HTML dosyasını kaydet
        html_path = os.path.join(self.reports_dir, f"report_{interview_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path


    def _html_to_pdf(self, html_path: str, interview_id: str) -> str:
        """HTML'i PDF'e dönüştürür."""
        pdf_path = os.path.join(self.reports_dir, f"report_{interview_id}.pdf")

        with open(html_path, "r", encoding="utf-8") as html_file:
            html_content = html_file.read()

        # PDF oluştur
        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(html_content, dest=pdf_file, encoding="utf-8")

        if pisa_status.err:
            raise Exception(f"PDF oluşturma hatası: {pisa_status.err}")

        return pdf_path

    def _save_json_report(
        self,
        interview_id: str,
        frame_summary: Dict,
        voice_analysis: Dict,
        ai_analysis: Dict,
        pyfeat_summary: Optional[Dict],
        video_info: Dict,
        duration_seconds: float,
    ) -> str:
        """Rafine JSON raporu kaydeder (ham frame_analysis olmadan)."""
        def _strip_voice_series(data: Dict) -> Dict:
            if not isinstance(data, dict):
                return data
            cleaned = dict(data)
            raw = cleaned.get("raw_voice_features", cleaned)
            if isinstance(raw, dict):
                raw = dict(raw)
                raw.pop("rms_energy_series", None)
                raw.pop("pitch_series", None)
                raw.pop("pitch_histogram", None)
                if "raw_voice_features" in cleaned:
                    cleaned["raw_voice_features"] = raw
                else:
                    cleaned = raw
            return cleaned

        report = {
            "interview_id": interview_id,
            "duration_seconds": duration_seconds,
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "video_info": video_info,
            "frame_summary": frame_summary,
            "voice_analysis": _strip_voice_series(voice_analysis),
            "ai_analysis": ai_analysis,
            "pyfeat_summary": pyfeat_summary if pyfeat_summary else {},
        }

        json_path = os.path.join(self.reports_dir, f"report_{interview_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return json_path


if __name__ == "__main__":
    print("ReportGenerator modülü hazır.")
