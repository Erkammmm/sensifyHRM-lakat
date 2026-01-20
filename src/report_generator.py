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
        """Plotly grafiklerini oluşturur ve HTML string olarak döndürür."""
        valid_frames = [f for f in frame_analysis if f is not None]
        if not valid_frames:
            return ""

        include_js = True

        def _to_html(fig: go.Figure, div_id: str) -> str:
            nonlocal include_js
            html = fig.to_html(
                include_plotlyjs="cdn" if include_js else False,
                div_id=div_id,
                full_html=False,
            )
            include_js = False
            return html

        charts = []

        def _add_title(title: str):
            charts.append(f"<h3>{title}</h3>")

        def _add_fig(fig: go.Figure, div_id: str):
            charts.append(f"<div class=\"chart-container\">{_to_html(fig, div_id)}</div>")

        # 1.1 Analiz Kapsamı Özeti
        total_frames = int(video_info.get("frame_count", len(frame_analysis) or 0))
        fps = float(video_info.get("fps", 0.0) or 0.0)
        analyzed_frames = len(valid_frames)
        frame_skip = int(round(total_frames / analyzed_frames)) if analyzed_frames else 0

        _add_title("ZAMAN & GENEL YAPI GRAFİKLERİ")
        info_labels = ["Video Süresi (s)", "FPS", "Analiz Edilen Frame", "Frame Skip"]
        info_values = [
            float(duration_seconds),
            float(fps),
            float(analyzed_frames),
            float(frame_skip),
        ]
        fig_info = go.Figure(
            data=[go.Bar(x=info_labels, y=info_values, marker_color="#4c78a8")]
        )
        fig_info.update_layout(
            title="Analiz Kapsamı Özeti",
            template="plotly_white",
            height=350,
        )
        _add_fig(fig_info, "chart_info")

        # 2.1 Zaman İçinde Yüz Aktivitesi
        timestamps = [f.get("timestamp", 0.0) for f in valid_frames]
        mouth_heights = [f.get("raw_features", {}).get("mouth_height_norm", 0.0) for f in valid_frames]
        jaw_open = [f.get("raw_features", {}).get("jaw_open_norm", 0.0) for f in valid_frames]
        eye_openings = [f.get("raw_features", {}).get("eye_opening_norm", 0.0) for f in valid_frames]
        brow_distances = [f.get("raw_features", {}).get("brow_distance_norm", 0.0) for f in valid_frames]

        fig_face_ts = go.Figure()
        fig_face_ts.add_trace(go.Scatter(x=timestamps, y=mouth_heights, mode="lines", name="mouth_height_norm"))
        fig_face_ts.add_trace(go.Scatter(x=timestamps, y=jaw_open, mode="lines", name="jaw_open_norm"))
        fig_face_ts.add_trace(go.Scatter(x=timestamps, y=eye_openings, mode="lines", name="eye_opening_norm"))
        fig_face_ts.add_trace(go.Scatter(x=timestamps, y=brow_distances, mode="lines", name="brow_distance_norm"))
        fig_face_ts.update_layout(
            title="Zaman İçinde Yüz Aktivitesi",
            xaxis_title="Zaman (saniye)",
            yaxis_title="Normalize Değer",
            template="plotly_white",
            height=400,
        )
        _add_title("YÜZ & MİMİK (MEDIAPIPE RAW FEATURES)")
        _add_fig(fig_face_ts, "chart_face_ts")

        # 2.2 Ortalama Yüz Özellikleri
        avg_feat = frame_summary.get("general_statistics", {}).get("average_features", {})
        if avg_feat:
            fig_face_avg = go.Figure(
                data=[
                    go.Bar(
                        x=list(avg_feat.keys()),
                        y=list(avg_feat.values()),
                        marker_color="#72b7b2",
                    )
                ]
            )
            fig_face_avg.update_layout(
                title="Ortalama Yüz Özellikleri",
                template="plotly_white",
                height=350,
            )
            _add_fig(fig_face_avg, "chart_face_avg")

        # 2.3 Emotion Change Points
        change_points = frame_summary.get("emotion_change_points", [])
        if change_points:
            cp_x = [cp.get("timestamp", 0.0) for cp in change_points]
            cp_y = [cp.get("feature", "") for cp in change_points]
            fig_cp = go.Figure(
                data=[
                    go.Scatter(
                        x=cp_x,
                        y=cp_y,
                        mode="markers",
                        marker=dict(size=10, color=cp_x, colorscale="Viridis"),
                        text=[f"%{cp.get('change_percentage', 0.0):.1f}" for cp in change_points],
                    )
                ]
            )
            fig_cp.update_layout(
                title="Emotion Change Points",
                xaxis_title="Zaman (saniye)",
                yaxis_title="Feature",
                template="plotly_white",
                height=350,
            )
            _add_fig(fig_cp, "chart_emotion_cp")

        # 3.1 Gaze dağılımı
        gaze_summary = frame_summary.get("gaze_summary", {})
        dir_perc = gaze_summary.get("direction_percentages", {})
        if dir_perc:
            labels = list(dir_perc.keys())
            values = list(dir_perc.values())
            fig_gaze_dist = go.Figure(
                data=[go.Pie(labels=labels, values=values, hole=0.3)]
            )
            fig_gaze_dist.update_layout(
                title="Göz Bakış Dağılımı",
                template="plotly_white",
                height=350,
            )
            _add_title("BAKIŞ (GAZE ANALİZİ)")
            _add_fig(fig_gaze_dist, "chart_gaze_dist")

        # 3.2 Zaman içinde gaze direction
        gaze_dirs = [f.get("gaze", {}).get("direction", "center") for f in valid_frames]
        dir_map = {"left": 0, "right": 1, "up": 2, "down": 3, "center": 4}
        gaze_vals = [dir_map.get(d, 4) for d in gaze_dirs]
        fig_gaze_ts = go.Figure()
        fig_gaze_ts.add_trace(go.Scatter(x=timestamps, y=gaze_vals, mode="lines", line_shape="hv"))
        fig_gaze_ts.update_layout(
            title="Zaman İçinde Bakış Durumu",
            xaxis_title="Zaman (saniye)",
            yaxis_title="Gaze Direction",
            yaxis=dict(
                tickmode="array",
                tickvals=list(dir_map.values()),
                ticktext=list(dir_map.keys()),
            ),
            template="plotly_white",
            height=350,
        )
        _add_fig(fig_gaze_ts, "chart_gaze_ts")

        # 3.3 Gaze center KPI
        gaze_center = frame_summary.get("general_statistics", {}).get("gaze_center_percentage", 0.0)
        fig_gaze_kpi = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(gaze_center),
                title={"text": "Gaze Center %"},
                gauge={"axis": {"range": [0, 100]}},
            )
        )
        fig_gaze_kpi.update_layout(height=300)
        _add_fig(fig_gaze_kpi, "chart_gaze_kpi")

        # 4.1 Bilişsel Yük
        cog_load = frame_summary.get("cognitive_load_score", {})
        if cog_load:
            fig_cog = go.Figure(
                data=[
                    go.Bar(
                        x=["thinking_reading_percentage", "speaking_percentage"],
                        y=[
                            cog_load.get("thinking_reading_percentage", 0.0),
                            cog_load.get("speaking_percentage", 0.0),
                        ],
                        marker_color="#f58518",
                    )
                ]
            )
            fig_cog.update_layout(
                title="Bilişsel Yük Özeti",
                template="plotly_white",
                height=350,
            )
            _add_title("BİLİŞSEL YÜK / DÜŞÜNME")
            _add_fig(fig_cog, "chart_cog")

        # 4.2 Jaw Open Threshold
        jaw_threshold = cog_load.get("jaw_open_threshold_used", 0.4)
        fig_jaw = go.Figure()
        fig_jaw.add_trace(go.Scatter(x=timestamps, y=jaw_open, mode="lines", name="jaw_open_norm"))
        fig_jaw.add_hline(y=jaw_threshold, line_dash="dash", line_color="red")
        fig_jaw.update_layout(
            title="Jaw Open Threshold Kullanımı",
            xaxis_title="Zaman (saniye)",
            yaxis_title="Normalize Değer",
            template="plotly_white",
            height=350,
        )
        _add_fig(fig_jaw, "chart_jaw_thresh")

        # 5. Ses Analizi
        rv = voice_analysis.get("raw_voice_features", voice_analysis)
        _add_title("SES ANALİZİ (VOICE)")
        rms_series = rv.get("rms_energy_series", {})
        if rms_series.get("values"):
            fig_rms_ts = go.Figure()
            fig_rms_ts.add_trace(go.Scatter(
                x=rms_series.get("times", []),
                y=rms_series.get("values", []),
                mode="lines",
                name="RMS Energy"
            ))
            fig_rms_ts.update_layout(
                title="RMS Energy Zaman Serisi",
                xaxis_title="Zaman (saniye)",
                yaxis_title="Enerji",
                template="plotly_white",
                height=350,
            )
            _add_fig(fig_rms_ts, "chart_rms_ts")

        rms_stats = rv.get("rms_energy", {})
        if rms_stats:
            fig_rms_bar = go.Figure(
                data=[go.Bar(x=list(rms_stats.keys()), y=list(rms_stats.values()))]
            )
            fig_rms_bar.update_layout(
                title="RMS Energy İstatistikleri",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_rms_bar, "chart_rms_bar")

        pitch_series = rv.get("pitch_series", {})
        if pitch_series.get("values"):
            fig_pitch_ts = go.Figure()
            fig_pitch_ts.add_trace(go.Scatter(
                x=pitch_series.get("times", []),
                y=pitch_series.get("values", []),
                mode="lines",
                name="Pitch (F0)"
            ))
            fig_pitch_ts.update_layout(
                title="Pitch (F0) Zaman Serisi",
                xaxis_title="Zaman (saniye)",
                yaxis_title="Hz",
                template="plotly_white",
                height=350,
            )
            _add_fig(fig_pitch_ts, "chart_pitch_ts")

        pitch_hist = rv.get("pitch_histogram", [])
        if pitch_hist:
            fig_pitch_hist = go.Figure(
                data=[go.Histogram(x=pitch_hist, nbinsx=30)]
            )
            fig_pitch_hist.update_layout(
                title="Pitch Dağılımı",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_pitch_hist, "chart_pitch_hist")

        pause_durations = rv.get("pause_durations", [])
        if pause_durations:
            fig_pause = go.Figure(
                data=[go.Histogram(x=pause_durations, nbinsx=20)]
            )
            fig_pause.update_layout(
                title="Pause Durations",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_pause, "chart_pause")

        silence_ratio = float(rv.get("silence_ratio", 0.0))
        fig_silence = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=silence_ratio * 100.0,
                title={"text": "Silence Ratio %"},
                gauge={"axis": {"range": [0, 100]}},
            )
        )
        fig_silence.update_layout(height=300)
        _add_fig(fig_silence, "chart_silence")

        spectral = rv.get("spectral_centroid", {})
        if spectral:
            fig_spec = go.Figure(
                data=[go.Bar(x=list(spectral.keys()), y=list(spectral.values()))]
            )
            fig_spec.update_layout(
                title="Spectral Centroid",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_spec, "chart_spec")

        zcr = rv.get("zero_crossing_rate", {})
        if zcr:
            fig_zcr = go.Figure(
                data=[go.Bar(x=list(zcr.keys()), y=list(zcr.values()))]
            )
            fig_zcr.update_layout(
                title="Zero Crossing Rate",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_zcr, "chart_zcr")

        # 6. Duygu (Py-Feat)
        pyfeat_summary = pyfeat_summary or {}
        emo_dist = pyfeat_summary.get("emotion_distribution", {})
        emo_means = emo_dist.get("mean", {})
        emo_vars = emo_dist.get("variance", {})
        if emo_means:
            _add_title("DUYGU (PYFEAT)")
            fig_emo_pie = go.Figure(
                data=[go.Pie(labels=list(emo_means.keys()), values=list(emo_means.values()), hole=0.3)]
            )
            fig_emo_pie.update_layout(
                title="Ortalama Duygu Dağılımı",
                template="plotly_white",
                height=350,
            )
            _add_fig(fig_emo_pie, "chart_emo_pie")

        if emo_vars:
            fig_emo_var = go.Figure(
                data=[go.Bar(x=list(emo_vars.keys()), y=list(emo_vars.values()))]
            )
            fig_emo_var.update_layout(
                title="Duygu Varyansı",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_emo_var, "chart_emo_var")

        dominant = emo_dist.get("dominant_emotion")
        dominant_score = emo_dist.get("dominant_emotion_score", 0.0)
        if dominant:
            fig_dom = go.Figure(
                go.Indicator(
                    mode="number",
                    value=float(dominant_score),
                    title={"text": f"Dominant Emotion: {dominant}"},
                )
            )
            fig_dom.update_layout(height=200)
            _add_fig(fig_dom, "chart_emo_dom")

        dom_freq = emo_dist.get("dominant_emotion_by_frequency", {})
        if dom_freq.get("emotion") is not None:
            fig_dom_freq = go.Figure(
                data=[
                    go.Bar(
                        x=[dom_freq.get("emotion")],
                        y=[dom_freq.get("count", 0)],
                        marker_color="#54a24b",
                    )
                ]
            )
            fig_dom_freq.update_layout(
                title="Dominant Emotion Frequency",
                template="plotly_white",
                height=250,
            )
            _add_fig(fig_dom_freq, "chart_emo_dom_freq")

        # 7. Head Pose & Beden Dili
        if pyfeat_frame_analysis:
            pose_times = []
            pitch_vals = []
            yaw_vals = []
            roll_vals = []
            for frame in pyfeat_frame_analysis:
                faces = frame.get("faces", [])
                if not faces:
                    continue
                pose = faces[0].get("pose", {})
                if not pose:
                    continue
                pose_times.append(frame.get("timestamp", 0.0))
                pitch_vals.append(pose.get("Pitch", 0.0))
                yaw_vals.append(pose.get("Yaw", 0.0))
                roll_vals.append(pose.get("Roll", 0.0))

            if pose_times:
                _add_title("KAFA POZU & BEDEN DİLİ")
                fig_pose_ts = go.Figure()
                fig_pose_ts.add_trace(go.Scatter(x=pose_times, y=pitch_vals, mode="lines", name="Pitch"))
                fig_pose_ts.add_trace(go.Scatter(x=pose_times, y=yaw_vals, mode="lines", name="Yaw"))
                fig_pose_ts.add_trace(go.Scatter(x=pose_times, y=roll_vals, mode="lines", name="Roll"))
                fig_pose_ts.update_layout(
                    title="Head Pose Zaman Serileri",
                    xaxis_title="Zaman (saniye)",
                    yaxis_title="Derece",
                    template="plotly_white",
                    height=350,
                )
                _add_fig(fig_pose_ts, "chart_pose_ts")

                fig_yaw_hist = go.Figure(
                    data=[go.Histogram(x=yaw_vals, nbinsx=30)]
                )
                fig_yaw_hist.update_layout(
                    title="Yaw Distribution",
                    template="plotly_white",
                    height=300,
                )
                _add_fig(fig_yaw_hist, "chart_yaw_hist")

        pose_stats = pyfeat_summary.get("pose_summary", {})
        if pose_stats:
            keys = []
            means = []
            stds = []
            mins = []
            maxs = []
            for k in ("Pitch", "Yaw", "Roll"):
                if k not in pose_stats:
                    continue
                keys.append(k)
                means.append(pose_stats[k].get("mean", 0.0))
                stds.append(pose_stats[k].get("std", 0.0))
                mins.append(pose_stats[k].get("min", 0.0))
                maxs.append(pose_stats[k].get("max", 0.0))
            if keys:
                fig_pose_bar = go.Figure()
                fig_pose_bar.add_trace(go.Bar(x=keys, y=means, name="mean"))
                fig_pose_bar.add_trace(go.Bar(x=keys, y=stds, name="std"))
                fig_pose_bar.add_trace(go.Bar(x=keys, y=mins, name="min"))
                fig_pose_bar.add_trace(go.Bar(x=keys, y=maxs, name="max"))
                fig_pose_bar.update_layout(
                    title="Head Pose İstatistikleri",
                    barmode="group",
                    template="plotly_white",
                    height=350,
                )
                _add_fig(fig_pose_bar, "chart_pose_bar")

        # 8. Kalite & Güvenilirlik
        face_rate = pyfeat_summary.get("face_detection_rate", None)
        if face_rate is not None:
            fig_face_rate = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(face_rate),
                    title={"text": "Face Detection Rate %"},
                    gauge={"axis": {"range": [0, 100]}},
                )
            )
            fig_face_rate.update_layout(height=300)
            _add_title("KALİTE & GÜVENİLİRLİK")
            _add_fig(fig_face_rate, "chart_face_rate")

        gs = pyfeat_summary.get("general_statistics", {})
        if gs:
            fig_frames = go.Figure(
                data=[
                    go.Bar(
                        x=["frames_analyzed", "total_frames"],
                        y=[gs.get("frames_analyzed", 0), gs.get("total_frames", 0)],
                        marker_color="#b279a2",
                    )
                ]
            )
            fig_frames.update_layout(
                title="Frames Analyzed vs Total Frames",
                template="plotly_white",
                height=300,
            )
            _add_fig(fig_frames, "chart_frames")

        # 9. İK Odaklı Radar Chart'lar
        speech_rate = float(rv.get("speech_rate", {}).get("value", 0.0))
        pause_count = len(pause_durations) if pause_durations else 0
        pitch_var = float(rv.get("pitch_f0", {}).get("std", 0.0))
        comm_values = [
            min(speech_rate / 100.0, 1.0),
            max(0.0, 1.0 - silence_ratio),
            max(0.0, 1.0 - min(pause_count / 20.0, 1.0)),
            min(pitch_var / 50.0, 1.0),
        ]
        fig_comm = go.Figure()
        fig_comm.add_trace(
            go.Scatterpolar(
                r=comm_values,
                theta=["speech_rate", "silence_ratio_inv", "pause_count_inv", "pitch_variance"],
                fill="toself",
                name="İletişim Akıcılığı",
            )
        )
        fig_comm.update_layout(
            title="İletişim Akıcılığı Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white",
            height=350,
        )
        _add_title("İK ODAKLI ÖZET GRAFİKLER (META)")
        _add_fig(fig_comm, "chart_comm_radar")

        gaze_center_norm = float(gaze_center) / 100.0
        pose_std_vals = [pose_stats.get(k, {}).get("std", 0.0) for k in ("Pitch", "Yaw", "Roll") if k in pose_stats]
        pose_std = float(sum(pose_std_vals) / len(pose_std_vals)) if pose_std_vals else 0.0
        emotion_vars = emo_vars.values() if emo_vars else []
        emotion_var_avg = float(sum(emotion_vars) / len(emotion_vars)) if emotion_vars else 0.0
        eye_open_avg = float(avg_feat.get("eye_opening_norm", 0.0)) if avg_feat else 0.0

        trust_values = [
            min(gaze_center_norm, 1.0),
            max(0.0, 1.0 - min(pose_std / 20.0, 1.0)),
            max(0.0, 1.0 - min(emotion_var_avg / 0.1, 1.0)),
            min(eye_open_avg / 0.2, 1.0),
        ]
        fig_trust = go.Figure()
        fig_trust.add_trace(
            go.Scatterpolar(
                r=trust_values,
                theta=["gaze_center", "head_pose_std_inv", "emotion_var_inv", "eye_opening"],
                fill="toself",
                name="Güven & Stabilite",
            )
        )
        fig_trust.update_layout(
            title="Güven & Stabilite Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white",
            height=350,
        )
        _add_fig(fig_trust, "chart_trust_radar")

        return "\n".join(charts)

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
