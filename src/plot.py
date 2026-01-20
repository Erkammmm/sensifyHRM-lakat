"""
Matplotlib/Seaborn tabanlı grafik üretimi.
Rapor için PNG görseller üretir ve HTML'de kullanılmak üzere yol döndürür.
"""

from typing import Dict, List, Optional, Any
import os
import math
import base64

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_fig(fig, output_path: str) -> str:
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _radar_chart(labels: List[str], values: List[float], title: str) -> plt.Figure:
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color="#4c78a8", linewidth=2)
    ax.fill(angles, values, color="#4c78a8", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    return fig


def _kpi_bar(label: str, value: float, max_value: float = 100.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh([label], [value], color="#4c78a8")
    ax.set_xlim(0, max_value)
    ax.set_title(label)
    for i, v in enumerate([value]):
        ax.text(v + max_value * 0.01, i, f"{v:.2f}", va="center")
    sns.despine(ax=ax, left=True, bottom=True)
    ax.get_xaxis().set_visible(False)
    return fig


def generate_report_charts(
    frame_analysis: List[Dict],
    frame_summary: Dict,
    voice_analysis: Dict,
    pyfeat_summary: Optional[Dict],
    pyfeat_frame_analysis: Optional[List[Dict]],
    video_info: Dict,
    duration_seconds: float,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """
    Tüm grafiklerin PNG dosyalarını üretir.
    Dönen liste: [{"title": "...", "path": "..."}]
    """
    _ensure_dir(output_dir)
    charts: List[Dict[str, Any]] = []

    sns.set_theme(style="whitegrid")

    valid_frames = [f for f in frame_analysis if f is not None]
    if not valid_frames:
        return charts

    timestamps = [f.get("timestamp", 0.0) for f in valid_frames]
    mouth_heights = [f.get("raw_features", {}).get("mouth_height_norm", 0.0) for f in valid_frames]
    jaw_open = [f.get("raw_features", {}).get("jaw_open_norm", 0.0) for f in valid_frames]
    eye_openings = [f.get("raw_features", {}).get("eye_opening_norm", 0.0) for f in valid_frames]
    brow_distances = [f.get("raw_features", {}).get("brow_distance_norm", 0.0) for f in valid_frames]

    total_frames = int(video_info.get("frame_count", len(frame_analysis) or 0))
    fps = float(video_info.get("fps", 0.0) or 0.0)
    analyzed_frames = len(valid_frames)
    frame_skip = int(round(total_frames / analyzed_frames)) if analyzed_frames else 0

    # 1.1 Analiz Kapsamı Özeti
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Video Süresi (s)", "FPS", "Analiz Edilen Frame", "Frame Skip"]
    values = [duration_seconds, fps, analyzed_frames, frame_skip]
    sns.barplot(x=labels, y=values, ax=ax, palette="muted")
    ax.set_title("Analiz Kapsamı Özeti")
    ax.set_ylabel("")
    charts.append({"title": "Analiz Kapsamı Özeti", "path": _save_fig(fig, os.path.join(output_dir, "info_summary.png"))})

    # 2.1 Zaman İçinde Yüz Aktivitesi
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timestamps, mouth_heights, label="mouth_height_norm")
    ax.plot(timestamps, jaw_open, label="jaw_open_norm")
    ax.plot(timestamps, eye_openings, label="eye_opening_norm")
    ax.plot(timestamps, brow_distances, label="brow_distance_norm")
    ax.set_title("Zaman İçinde Yüz Aktivitesi")
    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("Normalize Değer")
    ax.legend()
    charts.append({"title": "Zaman İçinde Yüz Aktivitesi", "path": _save_fig(fig, os.path.join(output_dir, "face_activity_ts.png"))})

    # 2.2 Ortalama Yüz Özellikleri
    avg_feat = frame_summary.get("general_statistics", {}).get("average_features", {})
    if avg_feat:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(avg_feat.keys()), y=list(avg_feat.values()), ax=ax, palette="deep")
        ax.set_title("Ortalama Yüz Özellikleri")
        ax.set_ylabel("Ortalama")
        ax.tick_params(axis="x", rotation=25)
        charts.append({"title": "Ortalama Yüz Özellikleri", "path": _save_fig(fig, os.path.join(output_dir, "face_avg.png"))})

    # 2.3 Emotion Change Points
    change_points = frame_summary.get("emotion_change_points", [])
    if change_points:
        fig, ax = plt.subplots(figsize=(6, 3))
        cp_x = [cp.get("timestamp", 0.0) for cp in change_points]
        cp_y = [cp.get("feature", "") for cp in change_points]
        ax.scatter(cp_x, range(len(cp_x)), c=cp_x, cmap="viridis")
        ax.set_yticks(range(len(cp_y)))
        ax.set_yticklabels(cp_y)
        ax.set_title("Emotion Change Points")
        ax.set_xlabel("Zaman (s)")
        charts.append({"title": "Emotion Change Points", "path": _save_fig(fig, os.path.join(output_dir, "emotion_change_points.png"))})

    # 3.1 Gaze Dağılımı
    gaze_summary = frame_summary.get("gaze_summary", {})
    dir_perc = gaze_summary.get("direction_percentages", {})
    if dir_perc:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(dir_perc.values(), labels=dir_perc.keys(), autopct="%1.1f%%")
        ax.set_title("Göz Bakış Dağılımı")
        charts.append({"title": "Göz Bakış Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "gaze_distribution.png"))})

    # 3.2 Zaman İçinde Bakış
    gaze_dirs = [f.get("gaze", {}).get("direction", "center") for f in valid_frames]
    dir_map = {"left": 0, "right": 1, "up": 2, "down": 3, "center": 4}
    gaze_vals = [dir_map.get(d, 4) for d in gaze_dirs]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.step(timestamps, gaze_vals, where="post")
    ax.set_yticks(list(dir_map.values()))
    ax.set_yticklabels(list(dir_map.keys()))
    ax.set_title("Zaman İçinde Bakış Durumu")
    ax.set_xlabel("Zaman (s)")
    charts.append({"title": "Zaman İçinde Bakış Durumu", "path": _save_fig(fig, os.path.join(output_dir, "gaze_timeline.png"))})

    # 3.3 Gaze Center KPI
    gaze_center = frame_summary.get("general_statistics", {}).get("gaze_center_percentage", 0.0)
    fig = _kpi_bar("Gaze Center %", float(gaze_center), 100.0)
    charts.append({"title": "Gaze Center %", "path": _save_fig(fig, os.path.join(output_dir, "gaze_center_kpi.png"))})

    # 4.1 Bilişsel Yük
    cog_load = frame_summary.get("cognitive_load_score", {})
    if cog_load:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["thinking_reading_percentage", "speaking_percentage"]
        values = [cog_load.get("thinking_reading_percentage", 0.0), cog_load.get("speaking_percentage", 0.0)]
        sns.barplot(x=labels, y=values, ax=ax, palette="pastel")
        ax.set_title("Bilişsel Yük Özeti")
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=15)
        charts.append({"title": "Bilişsel Yük Özeti", "path": _save_fig(fig, os.path.join(output_dir, "cognitive_load.png"))})

    # 4.2 Jaw Open Threshold
    jaw_threshold = cog_load.get("jaw_open_threshold_used", 0.4)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(timestamps, jaw_open, label="jaw_open_norm")
    ax.axhline(jaw_threshold, color="red", linestyle="--", label="threshold")
    ax.set_title("Jaw Open Threshold Kullanımı")
    ax.set_xlabel("Zaman (s)")
    ax.legend()
    charts.append({"title": "Jaw Open Threshold", "path": _save_fig(fig, os.path.join(output_dir, "jaw_threshold.png"))})

    # 5. Ses Analizi
    rv = voice_analysis.get("raw_voice_features", voice_analysis)
    rms_series = rv.get("rms_energy_series", {})
    if rms_series.get("values"):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(rms_series.get("times", []), rms_series.get("values", []))
        ax.set_title("RMS Energy Zaman Serisi")
        ax.set_xlabel("Zaman (s)")
        charts.append({"title": "RMS Energy Zaman Serisi", "path": _save_fig(fig, os.path.join(output_dir, "rms_series.png"))})

    rms_stats = rv.get("rms_energy", {})
    if rms_stats:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=list(rms_stats.keys()), y=list(rms_stats.values()), ax=ax)
        ax.set_title("RMS Energy İstatistikleri")
        ax.tick_params(axis="x", rotation=15)
        charts.append({"title": "RMS Energy İstatistikleri", "path": _save_fig(fig, os.path.join(output_dir, "rms_stats.png"))})

    pitch_series = rv.get("pitch_series", {})
    if pitch_series.get("values"):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(pitch_series.get("times", []), pitch_series.get("values", []))
        ax.set_title("Pitch (F0) Zaman Serisi")
        ax.set_xlabel("Zaman (s)")
        charts.append({"title": "Pitch (F0) Zaman Serisi", "path": _save_fig(fig, os.path.join(output_dir, "pitch_series.png"))})

    pitch_hist = rv.get("pitch_histogram", [])
    if pitch_hist:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(pitch_hist, bins=30, ax=ax)
        ax.set_title("Pitch Dağılımı")
        charts.append({"title": "Pitch Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "pitch_hist.png"))})

    pause_durations = rv.get("pause_durations", [])
    if pause_durations:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(pause_durations, bins=20, ax=ax)
        ax.set_title("Pause Durations")
        charts.append({"title": "Pause Durations", "path": _save_fig(fig, os.path.join(output_dir, "pause_durations.png"))})

    silence_ratio = float(rv.get("silence_ratio", 0.0)) * 100.0
    fig = _kpi_bar("Silence Ratio %", silence_ratio, 100.0)
    charts.append({"title": "Silence Ratio %", "path": _save_fig(fig, os.path.join(output_dir, "silence_ratio.png"))})

    spectral = rv.get("spectral_centroid", {})
    if spectral:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=list(spectral.keys()), y=list(spectral.values()), ax=ax)
        ax.set_title("Spectral Centroid")
        charts.append({"title": "Spectral Centroid", "path": _save_fig(fig, os.path.join(output_dir, "spectral_centroid.png"))})

    zcr = rv.get("zero_crossing_rate", {})
    if zcr:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=list(zcr.keys()), y=list(zcr.values()), ax=ax)
        ax.set_title("Zero Crossing Rate")
        charts.append({"title": "Zero Crossing Rate", "path": _save_fig(fig, os.path.join(output_dir, "zcr.png"))})

    # 6. Duygu (Py-Feat)
    pyfeat_summary = pyfeat_summary or {}
    emo_dist = pyfeat_summary.get("emotion_distribution", {})
    emo_means = emo_dist.get("mean", {})
    emo_vars = emo_dist.get("variance", {})
    if emo_means:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(emo_means.values(), labels=emo_means.keys(), autopct="%1.1f%%")
        ax.set_title("Ortalama Duygu Dağılımı")
        charts.append({"title": "Ortalama Duygu Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "emotion_mean.png"))})
    if emo_vars:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=list(emo_vars.keys()), y=list(emo_vars.values()), ax=ax)
        ax.set_title("Duygu Varyansı")
        ax.tick_params(axis="x", rotation=15)
        charts.append({"title": "Duygu Varyansı", "path": _save_fig(fig, os.path.join(output_dir, "emotion_var.png"))})

    dominant = emo_dist.get("dominant_emotion")
    dominant_score = emo_dist.get("dominant_emotion_score", 0.0)
    if dominant:
        fig = _kpi_bar(f"Dominant Emotion: {dominant}", float(dominant_score), 1.0)
        charts.append({"title": "Dominant Emotion KPI", "path": _save_fig(fig, os.path.join(output_dir, "emotion_dominant.png"))})

    dom_freq = emo_dist.get("dominant_emotion_by_frequency", {})
    if dom_freq.get("emotion") is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x=[dom_freq.get("emotion")], y=[dom_freq.get("count", 0)], ax=ax)
        ax.set_title("Dominant Emotion Frequency")
        charts.append({"title": "Dominant Emotion Frequency", "path": _save_fig(fig, os.path.join(output_dir, "emotion_dom_freq.png"))})

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
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(pose_times, pitch_vals, label="Pitch")
            ax.plot(pose_times, yaw_vals, label="Yaw")
            ax.plot(pose_times, roll_vals, label="Roll")
            ax.set_title("Head Pose Zaman Serileri")
            ax.set_xlabel("Zaman (s)")
            ax.legend()
            charts.append({"title": "Head Pose Zaman Serileri", "path": _save_fig(fig, os.path.join(output_dir, "pose_timeseries.png"))})

            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(yaw_vals, bins=30, ax=ax)
            ax.set_title("Yaw Distribution")
            charts.append({"title": "Yaw Distribution", "path": _save_fig(fig, os.path.join(output_dir, "yaw_hist.png"))})

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
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(keys))
            width = 0.2
            ax.bar(x - width * 1.5, means, width, label="mean")
            ax.bar(x - width * 0.5, stds, width, label="std")
            ax.bar(x + width * 0.5, mins, width, label="min")
            ax.bar(x + width * 1.5, maxs, width, label="max")
            ax.set_xticks(x)
            ax.set_xticklabels(keys)
            ax.set_title("Head Pose İstatistikleri")
            ax.legend()
            charts.append({"title": "Head Pose İstatistikleri", "path": _save_fig(fig, os.path.join(output_dir, "pose_stats.png"))})

    # 8. Kalite & Güvenilirlik
    face_rate = pyfeat_summary.get("face_detection_rate", None)
    if face_rate is not None:
        fig = _kpi_bar("Face Detection Rate %", float(face_rate), 100.0)
        charts.append({"title": "Face Detection Rate %", "path": _save_fig(fig, os.path.join(output_dir, "face_detection_kpi.png"))})

    gs = pyfeat_summary.get("general_statistics", {})
    if gs:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["frames_analyzed", "total_frames"]
        values = [gs.get("frames_analyzed", 0), gs.get("total_frames", 0)]
        sns.barplot(x=labels, y=values, ax=ax)
        ax.set_title("Frames Analyzed vs Total Frames")
        charts.append({"title": "Frames Analyzed vs Total Frames", "path": _save_fig(fig, os.path.join(output_dir, "frames_analyzed.png"))})

    # 9. İK Odaklı Radar Chart
    speech_rate = float(rv.get("speech_rate", {}).get("value", 0.0))
    pause_count = len(pause_durations) if pause_durations else 0
    pitch_var = float(rv.get("pitch_f0", {}).get("std", 0.0))
    comm_values = [
        min(speech_rate / 100.0, 1.0),
        max(0.0, 1.0 - silence_ratio / 100.0),
        max(0.0, 1.0 - min(pause_count / 20.0, 1.0)),
        min(pitch_var / 50.0, 1.0),
    ]
    fig = _radar_chart(
        ["speech_rate", "silence_ratio_inv", "pause_count_inv", "pitch_variance"],
        comm_values,
        "İletişim Akıcılığı Radar"
    )
    charts.append({"title": "İletişim Akıcılığı Radar", "path": _save_fig(fig, os.path.join(output_dir, "radar_communication.png"))})

    gaze_center_norm = float(gaze_center) / 100.0
    pose_std_vals = [pose_stats.get(k, {}).get("std", 0.0) for k in ("Pitch", "Yaw", "Roll") if k in pose_stats]
    pose_std = float(sum(pose_std_vals) / len(pose_std_vals)) if pose_std_vals else 0.0
    emotion_vars = list(emo_vars.values()) if emo_vars else []
    emotion_var_avg = float(sum(emotion_vars) / len(emotion_vars)) if emotion_vars else 0.0
    eye_open_avg = float(avg_feat.get("eye_opening_norm", 0.0)) if avg_feat else 0.0
    trust_values = [
        min(gaze_center_norm, 1.0),
        max(0.0, 1.0 - min(pose_std / 20.0, 1.0)),
        max(0.0, 1.0 - min(emotion_var_avg / 0.1, 1.0)),
        min(eye_open_avg / 0.2, 1.0),
    ]
    fig = _radar_chart(
        ["gaze_center", "head_pose_std_inv", "emotion_var_inv", "eye_opening"],
        trust_values,
        "Güven & Stabilite Radar"
    )
    charts.append({"title": "Güven & Stabilite Radar", "path": _save_fig(fig, os.path.join(output_dir, "radar_trust.png"))})

    return charts
