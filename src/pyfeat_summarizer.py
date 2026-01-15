"""
Py-Feat Summary Builder
Frame bazlı Py-Feat çıktılarından sadeleştirilmiş özet üretir.
"""

from typing import Dict, List, Any
from collections import Counter
import numpy as np


class PyFeatSummarizer:
    """
    Py-Feat frame bazlı çıktılarını özetler.
    Özet: duygu dağılımları, yüz tespit oranı ve genel istatistikler.
    """

    @staticmethod
    def summarize(frame_analysis: List[Dict[str, Any]], total_frames: int, fps: float, frame_skip: int) -> Dict[str, Any]:
        frames_analyzed = len(frame_analysis)
        frames_with_faces = 0
        faces_detected = 0

        emotion_values: Dict[str, List[float]] = {}
        dominant_counter = Counter()
        pose_values: Dict[str, List[float]] = {}

        for frame in frame_analysis:
            faces = frame.get("faces", [])
            if faces:
                frames_with_faces += 1
            for face in faces:
                faces_detected += 1
                emotions = face.get("emotions", {})
                if not emotions:
                    continue
                # dominant emotion (bu yüz için)
                dominant_emotion = max(emotions, key=emotions.get)
                dominant_counter[dominant_emotion] += 1
                for emo, val in emotions.items():
                    emotion_values.setdefault(emo, []).append(float(val))

                pose = face.get("pose", {})
                if pose:
                    for key in ("Pitch", "Yaw", "Roll"):
                        if key in pose:
                            pose_values.setdefault(key, []).append(float(pose.get(key, 0.0)))

        emotion_means = {
            emo: float(np.mean(vals)) if vals else 0.0 for emo, vals in emotion_values.items()
        }
        emotion_variances = {
            emo: float(np.var(vals)) if vals else 0.0 for emo, vals in emotion_values.items()
        }

        dominant_emotion = None
        dominant_emotion_score = 0.0
        if emotion_means:
            dominant_emotion = max(emotion_means, key=emotion_means.get)
            dominant_emotion_score = float(emotion_means.get(dominant_emotion, 0.0))

        dominant_freq = None
        dominant_freq_count = 0
        if dominant_counter:
            dominant_freq = dominant_counter.most_common(1)[0][0]
            dominant_freq_count = dominant_counter.most_common(1)[0][1]

        face_detection_rate = (
            (frames_with_faces / frames_analyzed * 100.0) if frames_analyzed > 0 else 0.0
        )
        avg_faces_per_frame = (
            (faces_detected / frames_analyzed) if frames_analyzed > 0 else 0.0
        )

        pose_stats = {}
        if pose_values:
            for key, vals in pose_values.items():
                if vals:
                    pose_stats[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }

        return {
            "emotion_distribution": {
                "mean": emotion_means,
                "variance": emotion_variances,
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_score": float(dominant_emotion_score),
                "dominant_emotion_by_frequency": {
                    "emotion": dominant_freq,
                    "count": int(dominant_freq_count),
                },
            },
            "pose_summary": pose_stats,
            "face_detection_rate": float(face_detection_rate),
            "general_statistics": {
                "total_frames": int(total_frames),
                "frames_analyzed": int(frames_analyzed),
                "frames_with_faces": int(frames_with_faces),
                "faces_detected": int(faces_detected),
                "avg_faces_per_frame": float(avg_faces_per_frame),
                "frame_skip": int(frame_skip),
                "fps": float(fps),
            },
        }


if __name__ == "__main__":
    print("PyFeatSummarizer modülü hazır.")
