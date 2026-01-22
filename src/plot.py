"""
Matplotlib/Seaborn tabanlı grafik üretimi.
Rapor için PNG görseller üretir ve HTML'de kullanılmak üzere yol döndürür.
"""

from typing import Dict, List, Optional, Any, Tuple
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

    def _clean_list(values: List[Any]) -> List[float]:
        cleaned = []
        for v in values:
            try:
                fv = float(v)
                cleaned.append(fv if math.isfinite(fv) else 0.0)
            except Exception:
                cleaned.append(0.0)
        return cleaned

    def _downsample_indices(length: int, max_points: int) -> Optional[List[int]]:
        if length <= max_points or max_points <= 0:
            return None
        return [int(i) for i in np.linspace(0, length - 1, max_points)]

    def _downsample_series(times: List[float], series_list: List[List[float]], max_points: int) -> Tuple[List[float], List[List[float]], bool]:
        if not times:
            return times, series_list, False
        idx = _downsample_indices(len(times), max_points)
        if not idx:
            return times, series_list, False
        ds_times = [times[i] for i in idx]
        ds_series = [[s[i] for i in idx] for s in series_list]
        return ds_times, ds_series, True

    def _limit_events(events: List[Any], max_events: int) -> List[Any]:
        if len(events) <= max_events or max_events <= 0:
            return events
        idx = [int(i) for i in np.linspace(0, len(events) - 1, max_events)]
        return [events[i] for i in idx]

    timestamps = _clean_list([f.get("timestamp", 0.0) for f in valid_frames])
    mouth_heights = _clean_list([f.get("raw_features", {}).get("mouth_height_norm", 0.0) for f in valid_frames])
    jaw_open = _clean_list([f.get("raw_features", {}).get("jaw_open_norm", 0.0) for f in valid_frames])
    eye_openings = _clean_list([f.get("raw_features", {}).get("eye_opening_norm", 0.0) for f in valid_frames])
    brow_distances = _clean_list([f.get("raw_features", {}).get("brow_distance_norm", 0.0) for f in valid_frames])

    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
            return val if math.isfinite(val) else default
        except Exception:
            return default

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            val = float(value)
            if not math.isfinite(val):
                return default
            return int(val)
        except Exception:
            return default

    total_frames = _safe_int(video_info.get("frame_count", len(frame_analysis) or 0), default=0)
    fps = _safe_float(video_info.get("fps", 0.0), default=0.0)
    analyzed_frames = len(valid_frames)
    if analyzed_frames and analyzed_frames > 0 and math.isfinite(analyzed_frames):
        ratio = total_frames / analyzed_frames if analyzed_frames else 0
        frame_skip = _safe_int(ratio, default=0)
    else:
        frame_skip = 0

    # --- NEW: İlk 5 kritik grafik en başta ---
    max_points = 400
    if duration_seconds >= 1200:
        max_points = 600
    elif duration_seconds >= 600:
        max_points = 800
    # 1) Zaman–Duygu Değişimi (Py-Feat, mediapipe timeline ile hizalı)
    if pyfeat_frame_analysis:
        emo_times = []
        emo_happiness = []
        emo_surprise = []
        emo_neutral = []
        for frame in pyfeat_frame_analysis:
            faces = frame.get("faces", [])
            if not faces:
                continue
            emotions = faces[0].get("emotions", {})
            if not emotions:
                continue
            emo_times.append(frame.get("timestamp", 0.0))
            emo_happiness.append(float(emotions.get("happiness", 0.0)))
            emo_surprise.append(float(emotions.get("surprise", 0.0)))
            emo_neutral.append(float(emotions.get("neutral", 0.0)))
        if emo_times:
            # Mediapipe timeline'a interpolasyon
            target_times = np.array(timestamps, dtype=float)
            src_times = np.array(emo_times, dtype=float)
            if len(src_times) >= 2 and len(target_times) >= 2:
                emo_happiness_interp = np.interp(target_times, src_times, _clean_list(emo_happiness))
                emo_surprise_interp = np.interp(target_times, src_times, _clean_list(emo_surprise))
                emo_neutral_interp = np.interp(target_times, src_times, _clean_list(emo_neutral))
                plot_times = target_times
            else:
                emo_happiness_interp = emo_happiness
                emo_surprise_interp = emo_surprise
                emo_neutral_interp = emo_neutral
                plot_times = emo_times
            plot_times_list = _clean_list(list(plot_times))
            emo_happiness_list = _clean_list(list(emo_happiness_interp))
            emo_surprise_list = _clean_list(list(emo_surprise_interp))
            emo_neutral_list = _clean_list(list(emo_neutral_interp))
            plot_times_list, series_list, was_ds = _downsample_series(
                plot_times_list,
                [emo_happiness_list, emo_surprise_list, emo_neutral_list],
                max_points,
            )
            emo_happiness_list, emo_surprise_list, emo_neutral_list = series_list
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(plot_times_list, emo_happiness_list, label="mutluluk", linewidth=1.0, alpha=0.85)
            ax.plot(plot_times_list, emo_surprise_list, label="şaşkınlık", linewidth=1.0, alpha=0.85)
            ax.plot(plot_times_list, emo_neutral_list, label="nötr", linewidth=1.0, alpha=0.85)
            ax.set_title("Zaman – Duygu Değişimi")
            ax.set_xlabel("Zaman (s)")
            ax.set_ylabel("Skor")
            if was_ds:
                ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
            ax.legend()
        charts.append({"title": "Zaman – Duygu Değişimi", "path": _save_fig(fig, os.path.join(output_dir, "emotion_timeseries.png"))})

    # 2) Gaze Direction Dağılımı (Bar / Pie)
    gaze_summary = frame_summary.get("gaze_summary", {})
    dir_perc = gaze_summary.get("direction_percentages", {})
    gaze_label_map = {
        "left": "sol",
        "right": "sağ",
        "up": "yukarı",
        "down": "aşağı",
        "center": "merkez",
    }
    added_gaze_distribution = False
    if dir_perc:
        fig, ax = plt.subplots(figsize=(5, 3))
        dir_labels = [gaze_label_map.get(k, k) for k in dir_perc.keys()]
        sns.barplot(x=dir_labels, y=list(dir_perc.values()), ax=ax, palette="muted")
        ax.set_title("Bakış Yönü Dağılımı")
        ax.set_ylabel("%")
        charts.append({"title": "Bakış Yönü Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "gaze_distribution_bar.png"))})
        added_gaze_distribution = True

    # 3) Pitch & Energy Zaman Serisi
    rv = voice_analysis.get("raw_voice_features", voice_analysis)
    rms_series = rv.get("rms_energy_series", {})
    pitch_series = rv.get("pitch_series", {})
    added_pitch_energy = False
    if rms_series.get("values") or pitch_series.get("values"):
        fig, ax1 = plt.subplots(figsize=(8, 3))
        rms_times = _clean_list(rms_series.get("times", [])) if rms_series.get("values") else []
        rms_vals = _clean_list(rms_series.get("values", [])) if rms_series.get("values") else []
        pitch_times = _clean_list(pitch_series.get("times", [])) if pitch_series.get("values") else []
        pitch_vals = _clean_list(pitch_series.get("values", [])) if pitch_series.get("values") else []
        if rms_times and rms_vals:
            rms_times, rms_list, rms_ds = _downsample_series(rms_times, [rms_vals], max_points)
            rms_vals = rms_list[0]
        else:
            rms_ds = False
        if pitch_times and pitch_vals:
            pitch_times, pitch_list, pitch_ds = _downsample_series(pitch_times, [pitch_vals], max_points)
            pitch_vals = pitch_list[0]
        else:
            pitch_ds = False
        if rms_series.get("values"):
            ax1.plot(rms_times, rms_vals, color="#1f77b4", label="Enerji (RMS)", linewidth=1.0, alpha=0.85)
            ax1.set_ylabel("Enerji (RMS)")
        ax2 = ax1.twinx()
        if pitch_series.get("values"):
            ax2.plot(pitch_times, pitch_vals, color="#ff7f0e", label="Ses tonu (Pitch)", linewidth=1.0, alpha=0.85)
            ax2.set_ylabel("Hz")
        ax1.set_title("Ses Tonu ve Enerji Zaman Serisi")
        ax1.set_xlabel("Zaman (s)")
        if rms_ds or pitch_ds:
            ax1.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax1.transAxes, fontsize=8)
        charts.append({"title": "Ses Tonu ve Enerji Zaman Serisi", "path": _save_fig(fig, os.path.join(output_dir, "pitch_energy_ts.png"))})
        added_pitch_energy = True

    # 4) Pause Timeline (Stem / Event Plot)
    pause_durations = _clean_list(rv.get("pause_durations", []))
    added_pause_timeline = False
    if pause_durations:
        fig, ax = plt.subplots(figsize=(7, 2.5))
        x = list(range(len(pause_durations)))
        y = pause_durations
        ax.stem(x, y, basefmt=" ")
        long_idx = [i for i, v in enumerate(y) if v >= 0.7]
        if long_idx:
            ax.scatter(long_idx, [y[i] for i in long_idx], color="red", label="Uzun duraksama")
            ax.legend()
        ax.set_title("Duraksama Zaman Çizelgesi")
        ax.set_xlabel("Duraksama sırası")
        ax.set_ylabel("Süre (s)")
        charts.append({"title": "Duraksama Zaman Çizelgesi", "path": _save_fig(fig, os.path.join(output_dir, "pause_timeline.png"))})
        added_pause_timeline = True

    # 5) Pose (Yaw–Pitch) Yoğunluk grafiği kaldırıldı

    # 1.1 Analiz Kapsamı Özeti
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Video Süresi (s)", "FPS", "Analiz Edilen Frame", "Frame Skip"]
    values = [duration_seconds, fps, analyzed_frames, frame_skip]
    sns.barplot(x=labels, y=values, ax=ax, palette="muted")
    ax.set_title("Analiz Kapsamı Özeti")
    ax.set_ylabel("")
    charts.append({"title": "Analiz Kapsamı Özeti", "path": _save_fig(fig, os.path.join(output_dir, "info_summary.png"))})

    # 2.1 Zaman İçinde Yüz Aktivitesi
    ts_ds, series_list, was_ds = _downsample_series(
        timestamps,
        [mouth_heights, jaw_open, eye_openings, brow_distances],
        max_points,
    )
    mouth_heights_ds, jaw_open_ds, eye_openings_ds, brow_distances_ds = series_list
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts_ds, mouth_heights_ds, label="Ağız yüksekliği", linewidth=1.0, alpha=0.85)
    ax.plot(ts_ds, jaw_open_ds, label="Çene açıklığı", linewidth=1.0, alpha=0.85)
    ax.plot(ts_ds, eye_openings_ds, label="Göz açıklığı", linewidth=1.0, alpha=0.85)
    ax.plot(ts_ds, brow_distances_ds, label="Kaş mesafesi", linewidth=1.0, alpha=0.85)
    ax.set_title("Zaman İçinde Yüz Aktivitesi")
    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("Normalize Değer")
    if was_ds:
        ax.text(0.01, -0.2, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
    ax.legend()
    charts.append({"title": "Zaman İçinde Yüz Aktivitesi", "path": _save_fig(fig, os.path.join(output_dir, "face_activity_ts.png"))})

    # 2.2 Ortalama Yüz Özellikleri
    avg_feat = frame_summary.get("general_statistics", {}).get("average_features", {})
    face_label_map = {
        "mouth_width_norm": "Ağız genişliği",
        "mouth_height_norm": "Ağız yüksekliği",
        "eye_opening_norm": "Göz açıklığı",
        "brow_distance_norm": "Kaş mesafesi",
        "jaw_open_norm": "Çene açıklığı",
    }
    if avg_feat:
        labels = [face_label_map.get(k, k) for k in avg_feat.keys()]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=labels, y=list(avg_feat.values()), ax=ax, palette="deep")
        ax.set_title("Ortalama Yüz Özellikleri")
        ax.set_ylabel("Ortalama")
        ax.tick_params(axis="x", rotation=25)
        charts.append({"title": "Ortalama Yüz Özellikleri", "path": _save_fig(fig, os.path.join(output_dir, "face_avg.png"))})

    # 2.3 Konuşma sırasında ani mimik değişimleri
    change_points = frame_summary.get("emotion_change_points", [])
    if change_points:
        change_points = _limit_events(change_points, 30)
        ts_ds, series_list, was_ds = _downsample_series(timestamps, [mouth_heights], max_points)
        mouth_heights_ds = series_list[0]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ts_ds, mouth_heights_ds, color="#1f77b4", label="Ağız yüksekliği", linewidth=1.0, alpha=0.85)
        for cp in change_points:
            ax.axvline(cp.get("timestamp", 0.0), color="red", alpha=0.5, linewidth=0.8)
        ax.set_title("Konuşma sırasında ani mimik değişimleri")
        ax.set_xlabel("Zaman (s)")
        ax.set_ylabel("Normalize Değer")
        if was_ds:
            ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
        charts.append({"title": "Konuşma sırasında ani mimik değişimleri", "path": _save_fig(fig, os.path.join(output_dir, "emotion_change_points.png"))})

    # 3.1 Gaze Dağılımı (önceden eklendiyse tekrar ekleme)
    if dir_perc and not added_gaze_distribution:
        fig, ax = plt.subplots(figsize=(5, 4))
        dir_labels = [gaze_label_map.get(k, k) for k in dir_perc.keys()]
        ax.pie(dir_perc.values(), labels=dir_labels, autopct="%1.1f%%")
        ax.set_title("Göz Bakış Dağılımı")
        charts.append({"title": "Göz Bakış Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "gaze_distribution.png"))})

    # 3.2 Zaman İçinde Bakış
    gaze_dirs = [f.get("gaze", {}).get("direction", "center") for f in valid_frames]
    dir_map = {"left": 0, "right": 1, "up": 2, "down": 3, "center": 4}
    gaze_vals = [dir_map.get(d, 4) for d in gaze_dirs]
    ts_ds, series_list, was_ds = _downsample_series(timestamps, [gaze_vals], max_points)
    gaze_vals_ds = series_list[0]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.step(ts_ds, gaze_vals_ds, where="post")
    ax.set_yticks(list(dir_map.values()))
    ax.set_yticklabels([gaze_label_map.get(k, k) for k in dir_map.keys()])
    ax.set_title("Zaman İçinde Bakış Durumu")
    ax.set_xlabel("Zaman (s)")
    if was_ds:
        ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
    charts.append({"title": "Zaman İçinde Bakış Durumu", "path": _save_fig(fig, os.path.join(output_dir, "gaze_timeline.png"))})

    # 3.3 Gaze Center KPI
    gaze_center = frame_summary.get("general_statistics", {}).get("gaze_center_percentage", 0.0)
    fig = _kpi_bar("Merkez Bakış %", float(gaze_center), 100.0)
    charts.append({"title": "Merkez Bakış %", "path": _save_fig(fig, os.path.join(output_dir, "gaze_center_kpi.png"))})

    # 4.1 Bilişsel Yük
    cog_load = frame_summary.get("cognitive_load_score", {})
    if cog_load:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["Düşünme/Okuma %", "Konuşma %"]
        values = [cog_load.get("thinking_reading_percentage", 0.0), cog_load.get("speaking_percentage", 0.0)]
        sns.barplot(x=labels, y=values, ax=ax, palette="pastel")
        ax.set_title("Bilişsel Yük Özeti")
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=15)
        charts.append({"title": "Bilişsel Yük Özeti", "path": _save_fig(fig, os.path.join(output_dir, "cognitive_load.png"))})

    # 4.2 Konuşma Aktivitesi Zaman Analizi
    jaw_threshold = cog_load.get("jaw_open_threshold_used", 0.4)
    fig, ax = plt.subplots(figsize=(8, 3))
    ts_ds, series_list, was_ds = _downsample_series(timestamps, [jaw_open], max_points)
    jaw_open_ds = series_list[0]
    ax.plot(ts_ds, jaw_open_ds, color="#888888", label="Çene açıklığı", linewidth=1.0, alpha=0.85)
    ax.axhline(jaw_threshold, color="red", linestyle="--", label="threshold")
    ax.fill_between(ts_ds, jaw_threshold, jaw_open_ds, where=np.array(jaw_open_ds) >= jaw_threshold, color="#cfe8ff", alpha=0.6, label="Konuşma")
    ax.fill_between(ts_ds, 0, jaw_threshold, color="#f0f0f0", alpha=0.4, label="Dinleme/Düşünme")
    ax.set_title("Konuşma Aktivitesi Zaman Analizi")
    ax.set_xlabel("Zaman (s)")
    ax.set_ylabel("Normalize Değer")
    if was_ds:
        ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper right")
    charts.append({"title": "Konuşma Aktivitesi Zaman Analizi", "path": _save_fig(fig, os.path.join(output_dir, "jaw_threshold.png"))})

    # 5. Ses Analizi
    rv = voice_analysis.get("raw_voice_features", voice_analysis)
    rms_series = rv.get("rms_energy_series", {})
    if rms_series.get("values") and not added_pitch_energy:
        fig, ax = plt.subplots(figsize=(8, 3))
        rms_times = _clean_list(rms_series.get("times", []))
        rms_vals = _clean_list(rms_series.get("values", []))
        rms_times, rms_list, was_ds = _downsample_series(rms_times, [rms_vals], max_points)
        rms_vals = rms_list[0]
        ax.plot(rms_times, rms_vals, linewidth=1.0, alpha=0.85)
        ax.set_title("Enerji (RMS) Zaman Serisi")
        ax.set_xlabel("Zaman (s)")
        if was_ds:
            ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
        charts.append({"title": "Enerji (RMS) Zaman Serisi", "path": _save_fig(fig, os.path.join(output_dir, "rms_series.png"))})

    rms_stats = rv.get("rms_energy", {})
    if rms_stats:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(x=list(rms_stats.keys()), y=list(rms_stats.values()), ax=ax)
        ax.set_title("Enerji (RMS) İstatistikleri")
        ax.tick_params(axis="x", rotation=15)
        charts.append({"title": "RMS Energy İstatistikleri", "path": _save_fig(fig, os.path.join(output_dir, "rms_stats.png"))})

    pitch_series = rv.get("pitch_series", {})
    if pitch_series.get("values") and not added_pitch_energy:
        fig, ax = plt.subplots(figsize=(8, 3))
        pitch_times = _clean_list(pitch_series.get("times", []))
        pitch_vals = _clean_list(pitch_series.get("values", []))
        pitch_times, pitch_list, was_ds = _downsample_series(pitch_times, [pitch_vals], max_points)
        pitch_vals = pitch_list[0]
        ax.plot(pitch_times, pitch_vals, linewidth=1.0, alpha=0.85)
        ax.set_title("Ses Tonu (Pitch) Zaman Serisi")
        ax.set_xlabel("Zaman (s)")
        if was_ds:
            ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
        charts.append({"title": "Ses Tonu (Pitch) Zaman Serisi", "path": _save_fig(fig, os.path.join(output_dir, "pitch_series.png"))})

    pitch_hist = rv.get("pitch_histogram", [])
    if pitch_hist:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(pitch_hist, bins=30, ax=ax)
        ax.set_title("Ses Tonu (Pitch) Dağılımı")
        charts.append({"title": "Pitch Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "pitch_hist.png"))})

    pause_durations = rv.get("pause_durations", [])
    if pause_durations and not added_pause_timeline:
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(pause_durations, bins=20, ax=ax)
        ax.set_title("Duraksama Süreleri")
        charts.append({"title": "Duraksama Süreleri", "path": _save_fig(fig, os.path.join(output_dir, "pause_durations.png"))})

    silence_ratio = float(rv.get("silence_ratio", 0.0)) * 100.0
    fig = _kpi_bar("Sessizlik Oranı %", silence_ratio, 100.0)
    charts.append({"title": "Sessizlik Oranı %", "path": _save_fig(fig, os.path.join(output_dir, "silence_ratio.png"))})

    spectral = rv.get("spectral_centroid", {})
    spectral_series = rv.get("spectral_centroid_series", {})
    if spectral_series.get("values"):
        fig, ax = plt.subplots(figsize=(8, 3))
        values = _clean_list(spectral_series.get("values", []))
        times = _clean_list(spectral_series.get("times", []))
        times, series_list, was_ds = _downsample_series(times, [values], max_points)
        values = np.array(series_list[0], dtype=float)
        if len(values) >= 5:
            smooth = np.convolve(values, np.ones(5) / 5, mode="same")
        else:
            smooth = values
        ax.plot(times, smooth, color="#1f77b4", label="Spektral Merkez (yumuşatılmış)")
        mean_val = float(spectral.get("mean", np.mean(values))) if spectral else float(np.mean(values))
        ax.axhline(mean_val, color="red", linestyle="--", label="Ortalama")
        ax.set_title("Ses Tonu Parlaklığı ve Vurgu Değişkenliği")
        ax.set_xlabel("Zaman (s)")
        ax.set_ylabel("Spektral Merkez")
        if was_ds:
            ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
        ax.text(0.01, 0.95, "Ham veri: STFT tabanlı spektral merkez (librosa)", transform=ax.transAxes, fontsize=8, va="top")
        ax.text(0.01, 0.88, "Yüksek değer: daha parlak/ince ton izlenimi.", transform=ax.transAxes, fontsize=8, va="top")
        ax.legend()
        charts.append({"title": "Ses Tonu Parlaklığı ve Vurgu Değişkenliği", "path": _save_fig(fig, os.path.join(output_dir, "spectral_centroid.png"))})

    zcr = rv.get("zero_crossing_rate", {})
    zcr_series = rv.get("zero_crossing_rate_series", {})
    if zcr_series.get("values"):
        fig, ax = plt.subplots(figsize=(8, 3))
        values = _clean_list(zcr_series.get("values", []))
        times = _clean_list(zcr_series.get("times", []))
        times, series_list, was_ds = _downsample_series(times, [values], max_points)
        values = np.array(series_list[0], dtype=float)
        ax.plot(times, values, color="#ff7f0e", label="ZCR")
        mean_val = float(zcr.get("mean", np.mean(values))) if zcr else float(np.mean(values))
        ax.axhline(mean_val, color="red", linestyle="--", label="Ortalama")
        ax.set_title("Konuşma Artikülasyon Stabilitesi (ZCR)")
        ax.set_xlabel("Zaman (s)")
        ax.set_ylabel("ZCR")
        if was_ds:
            ax.text(0.01, -0.25, "Uzun video için örneklenmiş görünüm.", transform=ax.transAxes, fontsize=8)
        ax.text(0.01, 0.95, "Ham veri: Zaman alanı sinyalinden hesaplanan ZCR (librosa)", transform=ax.transAxes, fontsize=8, va="top")
        ax.text(0.01, 0.88, "Dalgalanma düşükse artikülasyon daha stabildir.", transform=ax.transAxes, fontsize=8, va="top")
        ax.legend()
        charts.append({"title": "Konuşma Artikülasyon Stabilitesi (ZCR)", "path": _save_fig(fig, os.path.join(output_dir, "zcr.png"))})

    # 6. Duygu (Py-Feat)
    pyfeat_summary = pyfeat_summary or {}
    emo_dist = pyfeat_summary.get("emotion_distribution", {})
    emo_means = emo_dist.get("mean", {})
    emo_vars = emo_dist.get("variance", {})
    emotion_order = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    emotion_label_map = {
        "anger": "öfke",
        "disgust": "iğrenme",
        "fear": "korku",
        "happiness": "mutluluk",
        "sadness": "üzüntü",
        "surprise": "şaşkınlık",
        "neutral": "nötr",
    }

    def _normalize_emotion_stats(stats: Any) -> Dict[str, float]:
        if isinstance(stats, dict):
            return {k: float(stats.get(k, 0.0)) for k in emotion_order}
        if isinstance(stats, list):
            return {emotion_order[i]: float(stats[i]) if i < len(stats) else 0.0 for i in range(len(emotion_order))}
        return {k: 0.0 for k in emotion_order}

    emo_means_norm = _normalize_emotion_stats(emo_means)
    emo_vars_norm = _normalize_emotion_stats(emo_vars)

    if any(emo_means_norm.values()):
        fig, ax = plt.subplots(figsize=(5, 4))
        emo_labels = [emotion_label_map.get(k, k) for k in emo_means_norm.keys()]
        ax.pie(emo_means_norm.values(), labels=emo_labels, autopct="%1.1f%%")
        ax.set_title("Ortalama Duygu Dağılımı")
        charts.append({"title": "Ortalama Duygu Dağılımı", "path": _save_fig(fig, os.path.join(output_dir, "emotion_mean.png"))})
    if any(emo_vars_norm.values()):
        fig, ax = plt.subplots(figsize=(6, 3))
        emo_labels = [emotion_label_map.get(k, k) for k in emo_vars_norm.keys()]
        sns.barplot(x=emo_labels, y=list(emo_vars_norm.values()), ax=ax)
        ax.set_title("Duygu Varyansı")
        ax.tick_params(axis="x", rotation=15)
        charts.append({"title": "Duygu Varyansı", "path": _save_fig(fig, os.path.join(output_dir, "emotion_var.png"))})

    dominant = emo_dist.get("dominant_emotion")
    dominant_score = emo_dist.get("dominant_emotion_score", 0.0)
    if dominant:
        dominant_label = emotion_label_map.get(dominant, dominant)
        fig = _kpi_bar(f"Baskın Duygu: {dominant_label}", float(dominant_score), 1.0)
        charts.append({"title": "Baskın Duygu KPI", "path": _save_fig(fig, os.path.join(output_dir, "emotion_dominant.png"))})

    dom_freq = emo_dist.get("dominant_emotion_by_frequency", {})
    if dom_freq.get("emotion") is not None:
        fig, ax = plt.subplots(figsize=(4, 3))
        dom_label = emotion_label_map.get(dom_freq.get("emotion"), dom_freq.get("emotion"))
        sns.barplot(x=[dom_label], y=[dom_freq.get("count", 0)], ax=ax)
        ax.set_title("Baskın Duygu Sıklığı")
        charts.append({"title": "Baskın Duygu Sıklığı", "path": _save_fig(fig, os.path.join(output_dir, "emotion_dom_freq.png"))})

    # 7. Head Pose grafikleri kaldırıldı (yorumlanabilirlik düşük)

    pose_stats = pyfeat_summary.get("pose_summary", {})
    def _normalize_pose_stats(stats: Any) -> Dict[str, Dict[str, float]]:
        if not isinstance(stats, dict):
            return {}
        normalized = {}
        for key, value in stats.items():
            if not isinstance(value, dict):
                continue
            key_norm = str(key).strip().lower()
            if key_norm in ("pitch", "yaw", "roll"):
                normalized[key_norm] = {
                    "mean": float(value.get("mean", 0.0)),
                    "std": float(value.get("std", 0.0)),
                    "min": float(value.get("min", 0.0)),
                    "max": float(value.get("max", 0.0)),
                }
        return normalized

    pose_stats_norm = _normalize_pose_stats(pose_stats)

    # 8. Kalite & Güvenilirlik
    face_rate = pyfeat_summary.get("face_detection_rate", None)
    if face_rate is not None:
        fig = _kpi_bar("Yüz Tespit Oranı %", float(face_rate), 100.0)
        charts.append({"title": "Yüz Tespit Oranı %", "path": _save_fig(fig, os.path.join(output_dir, "face_detection_kpi.png"))})

    gs = pyfeat_summary.get("general_statistics", {})
    if gs:
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = ["Analiz Edilen Frame", "Toplam Frame"]
        values = [gs.get("frames_analyzed", 0), gs.get("total_frames", 0)]
        sns.barplot(x=labels, y=values, ax=ax)
        ax.set_title("Analiz Edilen vs Toplam Frame")
        charts.append({"title": "Analiz Edilen vs Toplam Frame", "path": _save_fig(fig, os.path.join(output_dir, "frames_analyzed.png"))})

    # Cross-modal ve akademik grafikler kaldırıldı (ürün odaklı sadeleştirme)

    return charts
