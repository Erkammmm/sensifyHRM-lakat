"""
Yüz Analizi Modülü (FAZ-4)

UniFace tabanlı:
  - RetinaFace       : yüz tespiti
  - DDAMFN AffectNet7: 7-sınıf duygu sınıflandırması + güven skoru
  - MobileGaze       : bakış yönü tahmini (pitch_deg, yaw_deg)
"""

import os
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np

from uniface.attribute import Emotion
from uniface.constants import DDAMFNWeights, GazeWeights, RetinaFaceWeights
from uniface.detection import RetinaFace
from uniface.gaze import MobileGaze

# Her 30. kare işlenir — CPU optimizasyonu.
# 30fps video → ~1fps efektif; interview analizi için yeterli, GPU'suz sunucuda 2x hızlanma.
PROCESS_EVERY_N = 30


class FaceAnalyzer:
    """
    UniFace tabanlı yüz analizi.

    Çıktı (timeline kayıt başına):
        {
            "timestamp_sec"      : float,  # karedeki video zamanı
            "emotion_label"      : str,    # Happy|Sad|Angry|Fear|Disgust|Surprise|Neutral
            "emotion_confidence" : float,  # 0.0 – 1.0
            "gaze_pitch_deg"     : float,  # + yukarı, - aşağı
            "gaze_yaw_deg"       : float,  # + sağa,   - sola
            "face_detected"      : bool
        }
    """

    def __init__(self):
        print(f"[{self.__class__.__name__}] Başlatılıyor... UniFace modelleri yükleniyor...")
        self.detector = RetinaFace(model_name=RetinaFaceWeights.MNET_025)
        self.gaze_estimator = MobileGaze(model_name=GazeWeights.RESNET18)
        self.emotion_predictor = Emotion(model_name=DDAMFNWeights.AFFECNET7)
        print(f"[{self.__class__.__name__}] Hazır!")

    def process_video(
        self,
        video_path: str,
        show_video: bool = False,
        phase3_enabled: bool = False,
    ) -> Tuple[List[Dict], Dict]:
        """
        Video dosyasını analiz eder.

        Args:
            video_path      : Video dosya yolu.
            show_video      : Geriye dönük uyumluluk için tutuldu; headless pipeline'da kullanılmaz.
            phase3_enabled  : Geriye dönük uyumluluk için tutuldu; FAZ-4'te her zaman yeni
                              format (emotion_label + gaze_deg) üretilir.

        Returns:
            (timeline, summary)
            timeline : her işlenen kare için bir kayıt içeren liste
            summary  : {total_blinks, blink_rate_per_min, duration_sec,
                        focus_score, dominant_emotion, data_count}
        """
        if not os.path.exists(video_path):
            print(f"[{self.__class__.__name__}] Video bulunamadı: {video_path}")
            return [], {}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[{self.__class__.__name__}] Video açılamadı: {video_path}")
            return [], {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(
            f"[{self.__class__.__name__}] Video Analizi Başladı "
            f"(Kaynak FPS: {fps:.0f} -> Her {PROCESS_EVERY_N}. kare isleniyor, "
            f"~{fps / PROCESS_EVERY_N:.1f} kare/sn efektif)..."
        )

        timeline: List[Dict] = []
        frame_count = 0  # videodaki gerçek kare numarası (grab() ile atlananlar dahil)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # cap.get() burada mevcut kare pozisyonunu saniye cinsinden verir
            timestamp_sec = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)

            # --- Yüz tespiti ---
            faces = self.detector.detect(frame)

            if not faces:
                # Yüz yoksa boş kayıt; pipeline çökmez
                timeline.append(
                    {
                        "timestamp_sec": timestamp_sec,
                        "emotion_label": "Neutral",
                        "emotion_confidence": 0.0,
                        "gaze_pitch_deg": 0.0,
                        "gaze_yaw_deg": 0.0,
                        "face_detected": False,
                    }
                )
            else:
                # Birden fazla yüz varsa en yüksek tespit güveni olan yüzü seç
                face = max(faces, key=lambda f: float(f.confidence))
                x1, y1, x2, y2 = map(int, face.bbox[:4])

                # --- Duygu tahmini ---
                emo_result = self.emotion_predictor.predict(frame, face.landmarks)
                emotion_label = emo_result.emotion.capitalize()
                emotion_confidence = round(float(emo_result.confidence), 3)

                # Surprise bias patch: Türk yüz mimiklerinde DDAMFN Surprise'a
                # aşırı yükleniyor. Düşük güvenle gelen Surprise → Neutral say.
                if emotion_label == "Surprise" and emotion_confidence < 0.75:
                    emotion_label = "Neutral"

                # --- Bakış yönü tahmini ---
                crop_y1 = max(0, y1)
                crop_y2 = min(frame_height, y2)
                crop_x1 = max(0, x1)
                crop_x2 = min(frame_width, x2)
                face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                gaze_pitch_deg = 0.0
                gaze_yaw_deg = 0.0
                if face_crop.size > 0:
                    gaze_result = self.gaze_estimator.estimate(face_crop)
                    gaze_pitch_deg = round(float(np.degrees(gaze_result.pitch)), 2)
                    gaze_yaw_deg = round(float(np.degrees(gaze_result.yaw)), 2)

                timeline.append(
                    {
                        "timestamp_sec": timestamp_sec,
                        "emotion_label": emotion_label,
                        "emotion_confidence": emotion_confidence,
                        "gaze_pitch_deg": gaze_pitch_deg,
                        "gaze_yaw_deg": gaze_yaw_deg,
                        "face_detected": True,
                    }
                )

            # Sonraki PROCESS_EVERY_N - 1 kareyi kör tarama ile atla
            for _ in range(PROCESS_EVERY_N - 1):
                if cap.grab():
                    frame_count += 1
                else:
                    break

        cap.release()

        # --- Özet istatistikler ---
        duration_sec = frame_count / fps if fps > 0 else 0.0
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 1.0

        detected = [r for r in timeline if r["face_detected"]]

        # Odak skoru: median-offset ile normalize edip eşikle
        # Eşikler contextual_aggregator ile tutarlı: |pitch| ≤ 20°, |yaw| ≤ 22°
        if detected:
            raw_pitches = sorted([r["gaze_pitch_deg"] for r in detected])
            raw_yaws = sorted([r["gaze_yaw_deg"] for r in detected])
            p_median = raw_pitches[len(raw_pitches) // 2]
            y_median = raw_yaws[len(raw_yaws) // 2]
            focus_count = sum(
                1
                for r in detected
                if abs(r["gaze_pitch_deg"] - p_median) <= 20.0
                and abs(r["gaze_yaw_deg"] - y_median) <= 22.0
            )
        else:
            focus_count = 0
        focus_score = round((focus_count / len(timeline)) * 100.0, 1) if timeline else 0.0

        emotion_labels = [r["emotion_label"] for r in detected]
        dominant_emotion = (
            Counter(emotion_labels).most_common(1)[0][0] if emotion_labels else "Veri Yok"
        )

        summary = {
            # UniFace ile göz kırpma tespiti yapılmıyor; geriye dönük uyumluluk için 0
            "total_blinks": 0,
            "blink_rate_per_min": 0.0,
            "duration_sec": round(duration_sec, 2),
            "focus_score": focus_score,
            "dominant_emotion": dominant_emotion,
            "data_count": len(timeline),
        }

        print(
            f"[{self.__class__.__name__}] Yüz analizi tamamlandı: "
            f"{len(timeline)} kayıt ({len(detected)} karede yüz tespit edildi)."
        )
        return timeline, summary


# ── FAZ-5: Delta tabanlı gaze analizi ──────────────────────────────────

_GAZE_BASELINE_WINDOW   = 30.0   # saniye — ilk N saniyeden baseline hesapla
_GAZE_AWAY_YAW_DELTA    = 15.0   # derece  — yaw sapma eşiği
_GAZE_AWAY_PITCH_DELTA  = 12.0   # derece  — pitch sapma eşiği
_GAZE_AWAY_MIN_DURATION =  6.0   # saniye  — kesintisiz bu kadar sürmeli


def _mmss(sec: float) -> str:
    m = int(sec // 60)
    s = int(round(sec % 60))
    if s >= 60:
        m += 1; s = 0
    return f"{m:02d}:{s:02d}"


def _build_gaze_event(
    start: float, end: float, duration: float, frames: List[tuple]
) -> Dict:
    """Tek bir sürekli bakış kayması olayı için dict oluşturur."""
    avg_yaw   = sum(f[1] for f in frames) / len(frames)
    avg_pitch = sum(f[2] for f in frames) / len(frames)

    if abs(avg_pitch) > abs(avg_yaw):
        direction = "yukarı" if avg_pitch > 0 else "aşağı"
    else:
        direction = "sağa" if avg_yaw > 0 else "sola"

    if direction == "aşağı":
        interpretation = "Not veya başka kaynağa bakış"
    elif direction in ("sola", "sağa"):
        interpretation = "Ekran dışına dikkat kayması"
    else:
        interpretation = "Düşünme ya da hatırlama refleksi"

    return {
        "start":            round(start, 1),
        "end":              round(end, 1),
        "start_label":      _mmss(start),
        "end_label":        _mmss(end),
        "duration_seconds": round(duration, 1),
        "direction":        direction,
        "interpretation":   interpretation,
    }


def compute_gaze_delta_analysis(
    timeline: List[Dict],
    video_info: Dict,
    baseline_window: float = _GAZE_BASELINE_WINDOW,
    yaw_delta: float       = _GAZE_AWAY_YAW_DELTA,
    pitch_delta: float     = _GAZE_AWAY_PITCH_DELTA,
    min_duration: float    = _GAZE_AWAY_MIN_DURATION,
) -> Dict:
    """
    Delta tabanlı göz analizi.

    Mutlak değer (kamera açısına bağlı, güvenilmez) yerine baseline'dan
    sapma ölçer. İlk BASELINE_WINDOW saniyenin medyanını referans alır;
    YAW_DELTA veya PITCH_DELTA'yı aşan ve MIN_DURATION saniye kesintisiz
    süren periyotları "gaze_away_event" olarak kaydeder.

    Telefon videosu (dikey, width < height) otomatik tespit edilir ve
    eşikler buna göre genişletilir.

    Returns dict:
        baseline_yaw, baseline_pitch,
        gaze_away_events: [{start, end, start_label, end_label,
                            duration_seconds, direction, interpretation}],
        total_gaze_away_seconds, gaze_away_percentage,
        data_quality: "yuksek"|"orta"|"telefon_videosu"|"dusuk"
    """
    # ── Telefon videosu tespiti ─────────────────────────────────────────
    w = int(video_info.get("width", 0) or 0)
    h = int(video_info.get("height", 0) or 0)
    is_portrait = (h > w > 0)
    if is_portrait:
        # Dikey videoda yüz büyük göründüğünden açılar daha sert kayar;
        # eşiği gevşet (x1.3)
        yaw_delta   *= 1.3
        pitch_delta *= 1.3
        print(f"[GazeAnalysis] Dikey video tespit edildi ({w}x{h}). "
              f"Eşikler genişletildi: yaw±{yaw_delta:.1f}° pitch±{pitch_delta:.1f}°")

    # ── Yüz tespiti istatistikleri & veri kalitesi ──────────────────────
    detected = [r for r in timeline if r.get("face_detected", True)]
    total_frames = max(1, len(timeline))
    det_rate = len(detected) / total_frames

    if det_rate < 0.10:
        data_quality = "dusuk"
    elif is_portrait:
        data_quality = "telefon_videosu"
    elif det_rate < 0.50:
        data_quality = "orta"
    else:
        data_quality = "yuksek"

    print(f"[GazeAnalysis] Yüz tespit oranı: {det_rate*100:.0f}% → kalite: {data_quality}")

    if len(detected) < 5:
        print("[GazeAnalysis] Yetersiz veri, analiz atlandı.")
        return {
            "baseline_yaw": 0.0, "baseline_pitch": 0.0,
            "gaze_away_events": [],
            "total_gaze_away_seconds": 0.0,
            "gaze_away_percentage": 0.0,
            "data_quality": data_quality,
        }

    # ── Baseline hesapla (ilk BASELINE_WINDOW sn medyanı) ──────────────
    baseline_frames = [r for r in detected if r["timestamp_sec"] <= baseline_window]
    if len(baseline_frames) < 3:
        # Çok kısa video: ilk %25 kare
        baseline_frames = detected[: max(5, len(detected) // 4)]

    b_yaws   = sorted(r["gaze_yaw_deg"]   for r in baseline_frames)
    b_pitches = sorted(r["gaze_pitch_deg"] for r in baseline_frames)
    baseline_yaw   = b_yaws[len(b_yaws) // 2]
    baseline_pitch = b_pitches[len(b_pitches) // 2]
    print(f"[GazeAnalysis] Baseline — yaw: {baseline_yaw:.1f}°  pitch: {baseline_pitch:.1f}°  "
          f"({len(baseline_frames)} kareden)")

    # ── Sürekli sapma olaylarını tespit et ─────────────────────────────
    events: List[Dict] = []
    event_start: float | None = None
    event_frames: List[tuple] = []   # (timestamp, yaw_dev, pitch_dev)

    for rec in detected:
        t         = float(rec["timestamp_sec"])
        yaw_dev   = rec["gaze_yaw_deg"]   - baseline_yaw
        pitch_dev = rec["gaze_pitch_deg"] - baseline_pitch
        is_away   = abs(yaw_dev) > yaw_delta or abs(pitch_dev) > pitch_delta

        if is_away:
            if event_start is None:
                event_start = t
            event_frames.append((t, yaw_dev, pitch_dev))
        else:
            if event_start is not None:
                dur = t - event_start
                if dur >= min_duration:
                    events.append(_build_gaze_event(event_start, t, dur, event_frames))
                event_start = None
                event_frames = []

    # Video sonu kapanmayan event
    if event_start is not None and event_frames:
        end_t = detected[-1]["timestamp_sec"]
        dur   = end_t - event_start
        if dur >= min_duration:
            events.append(_build_gaze_event(event_start, end_t, dur, event_frames))

    # ── Özet istatistikler ──────────────────────────────────────────────
    video_dur      = float(video_info.get("duration_seconds") or 0.0)
    total_away_sec = sum(e["duration_seconds"] for e in events)
    away_pct       = (total_away_sec / video_dur * 100.0) if video_dur > 0 else 0.0

    print(f"[GazeAnalysis] {len(events)} gaze-away olayı — "
          f"toplam {total_away_sec:.1f}s (%{away_pct:.1f})")

    return {
        "baseline_yaw":            round(baseline_yaw, 2),
        "baseline_pitch":          round(baseline_pitch, 2),
        "gaze_away_events":        events,
        "total_gaze_away_seconds": round(total_away_sec, 1),
        "gaze_away_percentage":    round(away_pct, 1),
        "data_quality":            data_quality,
    }


if __name__ == "__main__":
    analyzer = FaceAnalyzer()
    print("FaceAnalyzer modülü hazır.")
