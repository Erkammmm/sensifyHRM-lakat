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


if __name__ == "__main__":
    analyzer = FaceAnalyzer()
    print("FaceAnalyzer modülü hazır.")
