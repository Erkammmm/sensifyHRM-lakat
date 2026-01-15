"""
Rapor Oluşturucu
Görselleştirme (Plotly) ve HTML/PDF rapor oluşturma.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from jinja2 import Template
from xhtml2pdf import pisa


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
        pyfeat_summary: Optional[Dict] = None,
        video_info: Dict,
        duration_seconds: float,
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
            video_info: Video bilgileri
            duration_seconds: Video süresi

        Returns:
            Oluşturulan dosya yolları: {"json": "...", "html": "...", "pdf": "..."}
        """
        # 1. Grafikleri oluştur
        charts_html = self._create_charts(frame_analysis, frame_summary)

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

    def _create_charts(self, frame_analysis: List[Dict], frame_summary: Dict) -> str:
        """Plotly grafiklerini oluşturur ve HTML string olarak döndürür."""
        valid_frames = [f for f in frame_analysis if f is not None]
        if not valid_frames:
            return ""

        # 1. Zaman eksenli Ağız ve Göz Hareketleri grafiği
        timestamps = [f.get("timestamp", 0.0) for f in valid_frames]
        mouth_widths = [
            f.get("raw_features", {}).get("mouth_width_norm", 0.0)
            for f in valid_frames
        ]
        mouth_heights = [
            f.get("raw_features", {}).get("mouth_height_norm", 0.0)
            for f in valid_frames
        ]
        eye_openings = [
            f.get("raw_features", {}).get("eye_opening_norm", 0.0)
            for f in valid_frames
        ]

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=timestamps,
                y=mouth_widths,
                mode="lines",
                name="Ağız Genişliği",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=timestamps,
                y=mouth_heights,
                mode="lines",
                name="Ağız Yüksekliği",
                line=dict(color="#ff7f0e", width=2),
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=timestamps,
                y=eye_openings,
                mode="lines",
                name="Göz Açıklığı",
                line=dict(color="#2ca02c", width=2),
            )
        )
        fig1.update_layout(
            title="Ağız ve Göz Hareketleri (Zaman Eksenli)",
            xaxis_title="Zaman (saniye)",
            yaxis_title="Normalize Değer",
            template="plotly_white",
            height=400,
        )

        # 2. Göz Bakış Dağılımı pasta grafiği
        gaze_summary = frame_summary.get("gaze_summary", {})
        dir_perc = gaze_summary.get("direction_percentages", {})
        
        if dir_perc:
            labels = list(dir_perc.keys())
            values = list(dir_perc.values())
            colors = px.colors.qualitative.Set3[: len(labels)]

            fig2 = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3,
                        marker=dict(colors=colors),
                    )
                ]
            )
            fig2.update_layout(
                title="Göz Bakış Dağılımı",
                template="plotly_white",
                height=400,
            )
        else:
            fig2 = go.Figure()
            fig2.add_annotation(
                text="Göz bakış verisi mevcut değil",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            fig2.update_layout(height=400)

        # HTML string olarak döndür (PDF için base64 image olarak da kaydedilebilir)
        # Şimdilik HTML embed olarak döndürüyoruz
        chart1_html = fig1.to_html(include_plotlyjs='cdn', div_id='chart1', full_html=False)
        chart2_html = fig2.to_html(include_plotlyjs=False, div_id='chart2', full_html=False)
        
        charts_html = f"""
        <div class="chart-container">
            {chart1_html}
        </div>
        <div class="chart-container">
            {chart2_html}
        </div>
        """

        return charts_html

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

        <div class="section">
            <h2>AI Analizi (Gemini)</h2>
            <div class="ai-analysis">
                {{ ai_analysis_text | safe }}
            </div>
        </div>

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
            ai_analysis_text=self._format_ai_analysis(ai_analysis),
        )

        # HTML dosyasını kaydet
        html_path = os.path.join(self.reports_dir, f"report_{interview_id}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def _format_ai_analysis(self, ai_analysis: Dict) -> str:
        """AI analizini HTML formatına çevirir."""
        analysis_text = ai_analysis.get("analysis", "")
        if not analysis_text:
            return "<p>Analiz mevcut değil.</p>"

        # Basit markdown benzeri formatlamayı HTML'e çevir
        # **text** -> <strong>text</strong>
        # - item -> <li>item</li>
        lines = analysis_text.split("\n")
        html_lines = []
        in_list = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                continue

            # Başlık kontrolü
            if line.startswith("**") and line.endswith("**"):
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<h3>{line[2:-2]}</h3>")
            elif line.startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = line[2:].replace("**", "<strong>").replace("**", "</strong>")
                html_lines.append(f"<li>{content}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                # **text** -> <strong>text</strong>
                content = line.replace("**", "<strong>").replace("**", "</strong>")
                html_lines.append(f"<p>{content}</p>")

        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

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
        report = {
            "interview_id": interview_id,
            "duration_seconds": duration_seconds,
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "video_info": video_info,
            "frame_summary": frame_summary,
            "voice_analysis": voice_analysis,
            "ai_analysis": ai_analysis,
            "pyfeat_summary": pyfeat_summary if pyfeat_summary else {},
        }

        json_path = os.path.join(self.reports_dir, f"report_{interview_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return json_path


if __name__ == "__main__":
    print("ReportGenerator modülü hazır.")
