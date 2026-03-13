"""
Grafik Oluşturma Modülü
Tüm analiz verileri için matplotlib grafikleri üretir.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalış
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Any, Optional
from collections import Counter


# --- Türkçe renk paleti ---
EMOTION_COLORS = {
    "Mutlu": "#2ecc71",
    "Notr": "#95a5a6",
    "Korku": "#9b59b6",
    "Tiksinti": "#e67e22",
    "Ofkeli": "#e74c3c",
    "Saskin": "#f1c40f",
    "Stresli": "#c0392b",
    "Uzgun": "#3498db",
    # Ses duygu (Türkçe büyük harf)
    "MUTLU": "#2ecc71",
    "NÖTR": "#95a5a6",
    "SAKİN": "#1abc9c",
    "KIZGIN": "#e74c3c",
    "KORKU": "#9b59b6",
    "TİKSİNME": "#e67e22",
    "ÜZGÜN": "#3498db",
    "ŞAŞIRMA": "#f1c40f",
    "SESSİZLİK": "#bdc3c7",
}

GAZE_COLORS = {
    "Ekrana Bakiyor": "#2ecc71",
    "Saga bakiyor": "#e74c3c",
    "Sola bakiyor": "#3498db",
    "Asagi bakiyor": "#9b59b6",
    "Yukari bakiyor": "#f39c12",
}

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
}

# FAZ-3 (signal palette)
SIGNAL_COLORS = {
    "NEGATIVE": "#fb7185",
    "NEUTRAL": "#94a3b8",
    "POSITIVE": "#34d399",
    "LOW": "#60a5fa",
    "MEDIUM": "#fbbf24",
    "HIGH": "#f97316",
    "FOCUSED": "#34d399",
    "AVERTED": "#fb7185",
    "NEUTRAL_STATE": "#94a3b8",
    "POSITIVE_STATE": "#34d399",
    "TENSE": "#fb7185",
    "ELEVATED": "#fbbf24",
}

# Genel stil
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _save_fig(fig, path):
    """Figürü kaydedip kapatır."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _smooth(values, window=5):
    """Basit hareketli ortalama."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


# =====================================================================
# 1) SES DUYGU TIMELINE (HuBERT SER)
# =====================================================================
def plot_audio_emotion_timeline(audio_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Ses duygu analizi zaman serisi grafiği."""
    if not audio_timeline:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    # Duygu etiketlerini sayısala çevir
    unique_emotions = list(set(t["emotion"] for t in audio_timeline))
    emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}

    times = [t["start"] for t in audio_timeline]
    indices = [emotion_to_idx[t["emotion"]] for t in audio_timeline]
    colors = [EMOTION_COLORS.get(t["emotion"], "#95a5a6") for t in audio_timeline]

    ax.scatter(times, indices, c=colors, s=30, alpha=0.7, zorder=5)

    # Yumuşatılmış çizgi
    if len(indices) > 5:
        smoothed = _smooth(np.array(indices, dtype=float), window=5)
        ax.plot(times, smoothed, color="#2c3e50", alpha=0.5, linewidth=1.5)

    ax.set_yticks(range(len(unique_emotions)))
    ax.set_yticklabels(unique_emotions)
    ax.set_xlabel("Zaman (saniye)")
    ax.set_title("🎤 Ses Duygu Analizi (HuBERT SER)", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "audio_emotion_timeline.png")
    _save_fig(fig, path)
    return {"title": "Ses Duygu Analizi - Zaman Serisi", "path": path}


# =====================================================================
# 2) SES DUYGU DAĞILIMI (pasta)
# =====================================================================
def plot_audio_emotion_distribution(audio_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Ses duygu dağılımı pasta grafiği."""
    if not audio_timeline:
        return None

    emotions = [t["emotion"] for t in audio_timeline]
    counter = Counter(emotions)
    labels = list(counter.keys())
    sizes = list(counter.values())
    colors = [EMOTION_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )
    ax.set_title("🎤 Ses Duygu Dağılımı", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "audio_emotion_dist.png")
    _save_fig(fig, path)
    return {"title": "Ses Duygu Dağılımı", "path": path}


# =====================================================================
# 3) METİN DUYGU DAĞILIMI
# =====================================================================
def plot_text_sentiment(text_segments: List[Dict], output_dir: str) -> Optional[Dict]:
    """Metin duygu analizi çubuk grafiği."""
    if not text_segments:
        return None

    # Phase-3: sentiment yok (metin = sadece içerik)
    if not isinstance(text_segments[0], dict) or "sentiment" not in text_segments[0]:
        return None

    sentiments = [s["sentiment"] for s in text_segments]
    counter = Counter(sentiments)
    labels = list(counter.keys())
    values = list(counter.values())
    colors = [SENTIMENT_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Cümle Sayısı")
    ax.set_title("📝 Metin Duygu Analizi (Türkçe BERT)", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "text_sentiment.png")
    _save_fig(fig, path)
    return {"title": "Metin Duygu Analizi", "path": path}


# =====================================================================
# 4) METİN DUYGU TIMELINE
# =====================================================================
def plot_text_sentiment_timeline(text_segments: List[Dict], output_dir: str) -> Optional[Dict]:
    """Metin duygu timeline (pozitif=1, negatif=-1)."""
    if not text_segments:
        return None

    # Phase-3: sentiment yok (metin = sadece içerik)
    if not isinstance(text_segments[0], dict) or "sentiment" not in text_segments[0]:
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    for seg in text_segments:
        mid = (seg["start"] + seg["end"]) / 2
        val = 1 if seg["sentiment"] == "positive" else -1
        color = SENTIMENT_COLORS.get(seg["sentiment"], "#95a5a6")
        ax.bar(mid, val, width=(seg["end"] - seg["start"]), color=color, alpha=0.7)

    ax.axhline(y=0, color="#2c3e50", linewidth=0.8)
    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Duygu")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Negatif", "", "Pozitif"])
    ax.set_title("📝 Metin Duygu Zaman Serisi", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "text_sentiment_timeline.png")
    _save_fig(fig, path)
    return {"title": "Metin Duygu Zaman Serisi", "path": path}


# =====================================================================
# 5) YÜZ DUYGU TIMELINE
# =====================================================================
def plot_face_emotion_timeline(face_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Yüz duygu analizi zaman serisi."""
    if not face_timeline:
        return None

    # FAZ-4 uses emotion_label; FAZ-3 uses emotion. Skip if neither present.
    emo_key = "emotion_label" if "emotion_label" in face_timeline[0] else ("emotion" if "emotion" in face_timeline[0] else None)
    if not emo_key:
        return None
    ts_key = "timestamp_sec" if "timestamp_sec" in face_timeline[0] else "timestamp"

    fig, ax = plt.subplots(figsize=(14, 5))

    unique_emotions = list(set(f[emo_key] for f in face_timeline if f.get(emo_key)))
    emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}

    times = [f.get(ts_key, 0) for f in face_timeline]
    indices = [emotion_to_idx.get(f.get(emo_key, ""), 0) for f in face_timeline]
    colors = [EMOTION_COLORS.get(f.get(emo_key, ""), "#95a5a6") for f in face_timeline]

    ax.scatter(times, indices, c=colors, s=25, alpha=0.7, zorder=5)

    if len(indices) > 5:
        smoothed = _smooth(np.array(indices, dtype=float), window=5)
        ax.plot(times, smoothed, color="#2c3e50", alpha=0.5, linewidth=1.5)

    ax.set_yticks(range(len(unique_emotions)))
    ax.set_yticklabels(unique_emotions)
    ax.set_xlabel("Zaman (saniye)")
    ax.set_title("😊 Yüz Duygu Analizi (MediaPipe)", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "face_emotion_timeline.png")
    _save_fig(fig, path)
    return {"title": "Yüz Duygu Analizi - Zaman Serisi", "path": path}


# =====================================================================
# 6) YÜZ DUYGU DAĞILIMI
# =====================================================================
def plot_face_emotion_distribution(face_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Yüz duygu dağılımı pasta grafiği."""
    if not face_timeline:
        return None

    # FAZ-4 uses emotion_label; FAZ-3 uses emotion. Skip if neither present.
    emo_key = "emotion_label" if "emotion_label" in face_timeline[0] else ("emotion" if "emotion" in face_timeline[0] else None)
    if not emo_key:
        return None

    emotions = [f.get(emo_key, "") for f in face_timeline if f.get(emo_key)]
    counter = Counter(emotions)
    labels = list(counter.keys())
    sizes = list(counter.values())
    colors = [EMOTION_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("😊 Yüz Duygu Dağılımı", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "face_emotion_dist.png")
    _save_fig(fig, path)
    return {"title": "Yüz Duygu Dağılımı", "path": path}


# =====================================================================
# 7) BAKIM YÖNÜ DAĞILIMI
# =====================================================================
def plot_gaze_distribution(face_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Bakış yönü dağılımı bar grafiği."""
    if not face_timeline:
        return None

    # FAZ-4: gaze_direction (str label); FAZ-3 fallback: gaze
    gaze_key = "gaze_direction" if "gaze_direction" in face_timeline[0] else "gaze"
    gazes = [f.get(gaze_key, "") for f in face_timeline if f.get(gaze_key)]
    if not gazes:
        return None
    counter = Counter(gazes)
    labels = list(counter.keys())
    values = list(counter.values())
    total = sum(values)
    colors = [GAZE_COLORS.get(l, "#95a5a6") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        pct = (val / total) * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val} (%{pct:.1f})", va="center", fontweight="bold")

    ax.set_xlabel("Kayıt Sayısı")
    ax.set_title("👁️ Bakış Yönü Dağılımı", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "gaze_distribution.png")
    _save_fig(fig, path)
    return {"title": "Bakış Yönü Dağılımı", "path": path}


# =====================================================================
# 8) GÖZ KIRPMA TIMELINE
# =====================================================================
def plot_blink_timeline(face_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Göz kırpma birikim grafiği."""
    if not face_timeline:
        return None

    # FAZ-4 has no blink_total; skip gracefully
    if "blink_total" not in face_timeline[0]:
        return None

    # FAZ-4: timestamp_sec; FAZ-3 fallback: timestamp
    ts_key = "timestamp_sec" if "timestamp_sec" in face_timeline[0] else "timestamp"
    times = [f.get(ts_key, 0) for f in face_timeline]
    blinks = [f["blink_total"] for f in face_timeline]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, blinks, color="#3498db", linewidth=2, alpha=0.8)
    ax.fill_between(times, blinks, alpha=0.15, color="#3498db")
    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Toplam Göz Kırpma")
    ax.set_title("👁️ Göz Kırpma Birikimi", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "blink_timeline.png")
    _save_fig(fig, path)
    return {"title": "Göz Kırpma Birikimi", "path": path}


# =====================================================================
# 9) RMS ENERJİ
# =====================================================================
def plot_rms_energy(voice_features: Dict, output_dir: str) -> Optional[Dict]:
    """RMS enerji zaman serisi."""
    rms = voice_features.get("energy_rms", {})
    series = rms.get("series", {})
    times = series.get("times", [])
    values = series.get("values", [])
    if not times or not values:
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, values, color="#e74c3c", linewidth=1, alpha=0.7)
    ax.fill_between(times, values, alpha=0.15, color="#e74c3c")

    mean_val = rms.get("mean", 0)
    ax.axhline(y=mean_val, color="#2c3e50", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Ortalama: {mean_val:.4f}")
    ax.legend()

    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("RMS Enerji")
    ax.set_title("🔊 Ses Enerjisi (RMS)", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "rms_energy.png")
    _save_fig(fig, path)
    return {"title": "Ses Enerjisi (RMS)", "path": path}


# =====================================================================
# 10) PİTCH (F0)
# =====================================================================
def plot_pitch(voice_features: Dict, output_dir: str) -> Optional[Dict]:
    """Pitch (F0) zaman serisi."""
    pitch = voice_features.get("pitch_f0", {})
    series = pitch.get("series", {})
    times = series.get("times", [])
    values = series.get("values", [])
    if not times or not values:
        return None

    values_arr = np.array(values)
    times_arr = np.array(times)
    # Sıfır olmayanları filtrele (voiced)
    mask = values_arr > 0
    if not mask.any():
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(times_arr[mask], values_arr[mask], s=3, color="#3498db", alpha=0.6)

    mean_val = pitch.get("mean", 0)
    if mean_val > 0:
        ax.axhline(y=mean_val, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"Ortalama: {mean_val:.1f} Hz")
        ax.legend()

    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Frekans (Hz)")
    ax.set_title("🎵 Ses Perdesi (Pitch / F0)", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "pitch_f0.png")
    _save_fig(fig, path)
    return {"title": "Ses Perdesi (Pitch / F0)", "path": path}


# =====================================================================
# 11) WAVEFORM + VAD
# =====================================================================
def plot_waveform_vad(voice_features: Dict, output_dir: str) -> Optional[Dict]:
    """Dalga formu + konuşma/sessizlik segmentleri."""
    waveform = voice_features.get("waveform", {})
    w_times = waveform.get("times", [])
    w_values = waveform.get("values", [])
    if not w_times or not w_values:
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(w_times, w_values, color="#2c3e50", linewidth=0.4, alpha=0.6)

    # VAD segmentlerini yeşil olarak göster
    segments = voice_features.get("speech_silence", {}).get("speech_segments", [])
    for seg in segments:
        ax.axvspan(seg["start"], seg["end"], alpha=0.15, color="#2ecc71")

    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Genlik")
    ax.set_title("📊 Dalga Formu + Konuşma Segmentleri", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "waveform_vad.png")
    _save_fig(fig, path)
    return {"title": "Dalga Formu + Konuşma Segmentleri", "path": path}


# =====================================================================
# 12) KONUŞMA / SESSİZLİK ORANLARI
# =====================================================================
def plot_speech_silence_ratio(voice_features: Dict, output_dir: str) -> Optional[Dict]:
    """Konuşma / sessizlik süresi bar grafiği."""
    ss = voice_features.get("speech_silence", {})
    speech = ss.get("total_speech_seconds", 0)
    silence = ss.get("total_silence_seconds", 0)
    if speech == 0 and silence == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Konuşma", "Sessizlik"]
    values = [speech, silence]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Süre (saniye)")
    ax.set_title("🗣️ Konuşma / Sessizlik Süresi", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "speech_silence.png")
    _save_fig(fig, path)
    return {"title": "Konuşma / Sessizlik Süresi", "path": path}


# =====================================================================
# 13) MEL SPECTROGRAM
# =====================================================================
def plot_mel_spectrogram(voice_features: Dict, output_dir: str) -> Optional[Dict]:
    """Mel spectrogram ısı haritası."""
    mel = voice_features.get("mel_spectrogram", {})
    mel_values = mel.get("values", [])
    mel_times = mel.get("times", [])
    mel_freqs = mel.get("frequencies", [])
    if not mel_values or not mel_times:
        return None

    fig, ax = plt.subplots(figsize=(14, 5))
    mel_arr = np.array(mel_values)
    extent = [mel_times[0], mel_times[-1], mel_freqs[0] if mel_freqs else 0,
              mel_freqs[-1] if mel_freqs else 8000]

    im = ax.imshow(mel_arr, aspect="auto", origin="lower", extent=extent,
                   cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="dB")
    ax.set_xlabel("Zaman (saniye)")
    ax.set_ylabel("Frekans (Hz)")
    ax.set_title("🎼 Mel Spectrogram", fontweight="bold", fontsize=13)

    path = os.path.join(output_dir, "mel_spectrogram.png")
    _save_fig(fig, path)
    return {"title": "Mel Spectrogram", "path": path}


# =====================================================================
# 14) DURAKLAMA SÜRELERİ
# =====================================================================
def plot_pause_durations(voice_features: Dict, output_dir: str) -> Optional[Dict]:
    """Her konuşma segmenti öncesi duraklama süreleri."""
    ss = voice_features.get("speech_silence", {})
    pauses = ss.get("response_pre_silence_seconds", [])
    if not pauses or len(pauses) < 2:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(1, len(pauses) + 1)
    ax.bar(x, pauses, color="#9b59b6", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Segment No")
    ax.set_ylabel("Duraklama (saniye)")
    ax.set_title("⏸️ Konuşma Öncesi Duraklama Süreleri", fontweight="bold", fontsize=13)

    if len(pauses) > 0:
        avg = np.mean(pauses)
        ax.axhline(y=avg, color="#e74c3c", linestyle="--", linewidth=1,
                   label=f"Ort: {avg:.2f}s")
        ax.legend()

    path = os.path.join(output_dir, "pause_durations.png")
    _save_fig(fig, path)
    return {"title": "Duraklama Süreleri", "path": path}


# =====================================================================
# 15) TUTARSIZLIK TIMELINE
# =====================================================================
def plot_anomalies(anomalies: List[Dict], output_dir: str) -> Optional[Dict]:
    """Tutarsızlık noktalarını gösteren grafik."""
    if not anomalies:
        return None

    fig, ax = plt.subplots(figsize=(14, 4))

    for i, a in enumerate(anomalies):
        # time_range parse: "3.5s - 7.2s"
        parts = a["time_range"].replace("s", "").split(" - ")
        if len(parts) == 2:
            try:
                start = float(parts[0])
                end = float(parts[1])
                mid = (start + end) / 2
                ax.axvspan(start, end, alpha=0.25, color="#e74c3c")
                ax.annotate(
                    f"#{i+1}", (mid, 0.5),
                    fontsize=8, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e74c3c", alpha=0.3),
                )
            except ValueError:
                pass

    ax.set_ylim(0, 1)
    ax.set_xlabel("Zaman (saniye)")
    ax.set_title("⚠️ Tutarsızlık Noktaları (Metin ↔ Yüz)", fontweight="bold", fontsize=13)
    ax.yaxis.set_visible(False)

    path = os.path.join(output_dir, "anomalies.png")
    _save_fig(fig, path)
    return {"title": "Tutarsızlık Noktaları", "path": path}


# =====================================================================
# FAZ-3: AUDIO SIGNAL GRAFİKLERİ
# =====================================================================
def plot_audio_signal_distributions(audio_signal_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Valence ve Arousal state dağılımını tek figürde gösterir."""
    if not audio_signal_timeline:
        return None

    valences = [t.get("valence_state", "") for t in audio_signal_timeline]
    arousals = [t.get("arousal_state", "") for t in audio_signal_timeline]
    v_counter = Counter(valences)
    a_counter = Counter(arousals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("🎛️ Audio Signal Dağılımı (Valence/Arousal)", fontsize=13, fontweight="bold")

    # Valence
    ax = axes[0]
    v_labels = list(v_counter.keys())
    v_vals = list(v_counter.values())
    v_colors = [SIGNAL_COLORS.get(l, "#95a5a6") for l in v_labels]
    ax.bar(v_labels, v_vals, color=v_colors, edgecolor="white", linewidth=1.0)
    ax.set_title("Valence State")
    ax.set_ylabel("Parça Sayısı")

    # Arousal
    ax = axes[1]
    a_labels = list(a_counter.keys())
    a_vals = list(a_counter.values())
    a_colors = [SIGNAL_COLORS.get(l, "#95a5a6") for l in a_labels]
    ax.bar(a_labels, a_vals, color=a_colors, edgecolor="white", linewidth=1.0)
    ax.set_title("Arousal State")

    fig.tight_layout()
    path = os.path.join(output_dir, "audio_signal_dist.png")
    _save_fig(fig, path)
    return {"title": "Audio Signal Dağılımı (Valence/Arousal)", "path": path}


def plot_audio_signal_timeline(audio_signal_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Valence/Arousal state'leri zaman serisi olarak gösterir."""
    if not audio_signal_timeline:
        return None

    # state -> index (ordered for readability)
    v_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    a_order = ["LOW", "MEDIUM", "HIGH"]
    v_to_idx = {v: i for i, v in enumerate(v_order)}
    a_to_idx = {a: i for i, a in enumerate(a_order)}

    times = [t.get("start", 0) for t in audio_signal_timeline]
    v_idx = [v_to_idx.get(t.get("valence_state", "NEUTRAL"), 1) for t in audio_signal_timeline]
    a_idx = [a_to_idx.get(t.get("arousal_state", "MEDIUM"), 1) for t in audio_signal_timeline]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].scatter(times, v_idx, c=[SIGNAL_COLORS.get(t.get("valence_state", ""), "#94a3b8") for t in audio_signal_timeline],
                    s=25, alpha=0.8)
    axes[0].set_yticks(range(len(v_order)))
    axes[0].set_yticklabels(v_order)
    axes[0].set_title("Valence Timeline", fontweight="bold")

    axes[1].scatter(times, a_idx, c=[SIGNAL_COLORS.get(t.get("arousal_state", ""), "#fbbf24") for t in audio_signal_timeline],
                    s=25, alpha=0.8)
    axes[1].set_yticks(range(len(a_order)))
    axes[1].set_yticklabels(a_order)
    axes[1].set_title("Arousal Timeline", fontweight="bold")
    axes[1].set_xlabel("Zaman (saniye)")

    fig.tight_layout()
    path = os.path.join(output_dir, "audio_signal_timeline.png")
    _save_fig(fig, path)
    return {"title": "Audio Signal Timeline (Valence/Arousal)", "path": path}


# =====================================================================
# FAZ-3: VISUAL SIGNAL GRAFİKLERİ
# =====================================================================
def plot_visual_signal_distributions(visual_timeline: List[Dict], output_dir: str) -> Optional[Dict]:
    """Facial/Attention/Stress state dağılımları."""
    if not visual_timeline:
        return None
    if not isinstance(visual_timeline[0], dict) or "facial_state" not in visual_timeline[0]:
        return None

    facial = [t.get("facial_state", "") for t in visual_timeline]
    attn = [t.get("attention_state", "") for t in visual_timeline]
    stress = [t.get("stress_indicator", "") for t in visual_timeline]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("👁️ Visual Signal Dağılımı", fontsize=13, fontweight="bold")

    for ax, title, values in [
        (axes[0], "Facial State", facial),
        (axes[1], "Attention State", attn),
        (axes[2], "Stress Indicator", stress),
    ]:
        counter = Counter(values)
        labels = list(counter.keys())
        vals = list(counter.values())
        colors = [SIGNAL_COLORS.get(l, "#95a5a6") for l in labels]
        ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.0)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    path = os.path.join(output_dir, "visual_signal_dist.png")
    _save_fig(fig, path)
    return {"title": "Visual Signal Dağılımı (Facial/Attention/Stress)", "path": path}

# =====================================================================
# 16) KPI ÖZETİ (Dashboard)
# =====================================================================
def plot_kpi_dashboard(face_summary: Dict, text_summary: Dict,
                       audio_summary: Dict, voice_features: Dict,
                       output_dir: str) -> Optional[Dict]:
    """Tüm modüllerin KPI göstergeleri tek bir dashboard'da."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("📊 Mülakat KPI Özeti", fontsize=16, fontweight="bold", y=1.02)

    # 1) Odak Skoru
    ax = axes[0, 0]
    focus = face_summary.get("focus_score", 0)
    ax.barh(["Odak"], [focus], color="#2ecc71" if focus > 60 else "#e74c3c")
    ax.set_xlim(0, 100)
    ax.set_title("👁️ Odak Skoru")
    ax.text(focus + 1, 0, f"%{focus:.0f}", va="center", fontweight="bold")

    # 2) Göz Kırpma
    ax = axes[0, 1]
    bpm = face_summary.get("blink_rate_per_min", 0)
    color = "#2ecc71" if 10 <= bpm <= 25 else "#e74c3c"
    ax.barh(["Kırpma/dk"], [bpm], color=color)
    ax.set_title("👁️ Göz Kırpma Hızı")
    ax.text(bpm + 0.3, 0, f"{bpm:.1f}", va="center", fontweight="bold")

    # 3) Baskın Yüz Duygusu
    ax = axes[0, 2]
    dom_face = face_summary.get("dominant_emotion", "?")
    ax.text(0.5, 0.5, dom_face, fontsize=24, ha="center", va="center",
            fontweight="bold", color=EMOTION_COLORS.get(dom_face, "#2c3e50"),
            transform=ax.transAxes)
    ax.set_title("😊 Baskın Yüz Duygusu")
    ax.axis("off")

    # 4) Metin Duygu Oranı
    ax = axes[1, 0]
    pos = text_summary.get("sentiment_percentages", {}).get("positive", 0)
    neg = text_summary.get("sentiment_percentages", {}).get("negative", 0)
    ax.barh(["Pozitif", "Negatif"], [pos, neg],
            color=[SENTIMENT_COLORS["positive"], SENTIMENT_COLORS["negative"]])
    ax.set_xlim(0, 100)
    ax.set_title("📝 Metin Duygu Oranı")

    # 5) Baskın Ses Duygusu
    ax = axes[1, 1]
    dom_audio = audio_summary.get("dominant_emotion", "?")
    ax.text(0.5, 0.5, dom_audio, fontsize=20, ha="center", va="center",
            fontweight="bold", color=EMOTION_COLORS.get(dom_audio, "#2c3e50"),
            transform=ax.transAxes)
    ax.set_title("🎤 Baskın Ses Duygusu")
    ax.axis("off")

    # 6) Konuşma/Sessizlik oranı
    ax = axes[1, 2]
    ss = voice_features.get("speech_silence", {})
    speech = ss.get("total_speech_seconds", 0)
    silence = ss.get("total_silence_seconds", 0)
    total = speech + silence
    if total > 0:
        ax.pie([speech, silence], labels=["Konuşma", "Sessizlik"],
               colors=["#2ecc71", "#e74c3c"], autopct="%1.0f%%",
               textprops={"fontsize": 9})
    ax.set_title("🗣️ Konuşma/Sessizlik")

    fig.tight_layout()

    path = os.path.join(output_dir, "kpi_dashboard.png")
    _save_fig(fig, path)
    return {"title": "Mülakat KPI Özeti", "path": path}


# =====================================================================
# ANA FONKSİYON: Tüm grafikleri üret
# =====================================================================
def generate_report_charts(report: Dict, output_dir: str) -> List[Dict[str, Any]]:
    """
    Tüm rapor grafiklerini üretir.

    Args:
        report: pipeline.process_interview() çıktısı
        output_dir: Grafiklerin kaydedileceği klasör

    Returns:
        list[dict]: Her grafik için {"title": str, "path": str}
    """
    os.makedirs(output_dir, exist_ok=True)

    # Verileri çıkar
    phase = (report.get("phase", "v2") or "v2").lower()
    text_segments = report.get("text_analysis", {}).get("segments", [])
    text_summary = report.get("text_analysis", {}).get("summary", {})
    audio_timeline = report.get("audio_emotion_analysis", {}).get("timeline", [])
    audio_summary = report.get("audio_emotion_analysis", {}).get("summary", {})
    face_timeline = report.get("face_analysis", {}).get("timeline", [])
    face_summary = report.get("face_analysis", {}).get("summary", {})
    audio_signal_timeline = report.get("audio_signal_analysis", {}).get("timeline", [])
    visual_signal_timeline = report.get("visual_signal_analysis", {}).get("timeline", [])
    voice_raw = report.get("voice_analysis", {}).get("raw_voice_features", {})
    anomalies = report.get("anomalies", [])

    charts = []

    # Sırayla grafikleri üret
    if phase == "v3":
        chart_funcs = [
            lambda: plot_visual_signal_distributions(visual_signal_timeline, output_dir),
            lambda: plot_gaze_distribution(face_timeline, output_dir),
            lambda: plot_blink_timeline(face_timeline, output_dir),
            lambda: plot_audio_signal_distributions(audio_signal_timeline, output_dir),
            lambda: plot_audio_signal_timeline(audio_signal_timeline, output_dir),
            # Teknik ses grafikleri (Detaylar)
            lambda: plot_rms_energy(voice_raw, output_dir),
            lambda: plot_pitch(voice_raw, output_dir),
            lambda: plot_waveform_vad(voice_raw, output_dir),
            lambda: plot_speech_silence_ratio(voice_raw, output_dir),
            lambda: plot_mel_spectrogram(voice_raw, output_dir),
            lambda: plot_pause_durations(voice_raw, output_dir),
        ]
    else:
        chart_funcs = [
            lambda: plot_kpi_dashboard(face_summary, text_summary, audio_summary, voice_raw, output_dir),
            lambda: plot_face_emotion_timeline(face_timeline, output_dir),
            lambda: plot_face_emotion_distribution(face_timeline, output_dir),
            lambda: plot_gaze_distribution(face_timeline, output_dir),
            lambda: plot_blink_timeline(face_timeline, output_dir),
            lambda: plot_audio_emotion_timeline(audio_timeline, output_dir),
            lambda: plot_audio_emotion_distribution(audio_timeline, output_dir),
            lambda: plot_text_sentiment(text_segments, output_dir),
            lambda: plot_text_sentiment_timeline(text_segments, output_dir),
            lambda: plot_rms_energy(voice_raw, output_dir),
            lambda: plot_pitch(voice_raw, output_dir),
            lambda: plot_waveform_vad(voice_raw, output_dir),
            lambda: plot_speech_silence_ratio(voice_raw, output_dir),
            lambda: plot_mel_spectrogram(voice_raw, output_dir),
            lambda: plot_pause_durations(voice_raw, output_dir),
            lambda: plot_anomalies(anomalies, output_dir),
        ]

    for func in chart_funcs:
        try:
            result = func()
            if result:
                charts.append(result)
        except Exception as e:
            print(f"[Plot] Grafik üretim hatası: {e}")

    print(f"[Plot] {len(charts)} grafik oluşturuldu.")
    return charts


if __name__ == "__main__":
    print("Plot modülü hazır.")
