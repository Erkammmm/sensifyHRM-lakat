"""
Py-Feat Summary Builder
Frame bazlı Py-Feat çıktılarından sadeleştirilmiş özet üretir.
"""

from typing import Dict, List, Any, Optional
from collections import Counter
import numpy as np


class PyFeatSummarizer:
    """
    Py-Feat frame bazlı çıktılarını özetler.
    Özet: duygu dağılımları, yüz tespit oranı ve genel istatistikler.
    """

    @staticmethod
    def summarize(frame_analysis: List[Dict[str, Any]], total_frames: int, fps: float, frame_skip: int) -> Dict[str, Any]:
        valid_frames = [f for f in frame_analysis if f is not None]
        frames_analyzed = len(valid_frames)
        frames_with_faces = 0
        faces_detected = 0

        emotion_values: Dict[str, List[float]] = {}
        dominant_counter = Counter()
        pose_values: Dict[str, List[float]] = {}

        for frame in valid_frames:
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
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                if arr.size == 0:
                    continue
                mean_val = float(np.nanmean(arr))
                std_val = float(np.nanstd(arr))
                min_val = float(np.nanmin(arr))
                max_val = float(np.nanmax(arr))
                if not np.isfinite(mean_val):
                    mean_val = 0.0
                if not np.isfinite(std_val):
                    std_val = 0.0
                if not np.isfinite(min_val):
                    min_val = 0.0
                if not np.isfinite(max_val):
                    max_val = 0.0
                pose_stats[key] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                }

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                v = float(value)
                return int(v) if np.isfinite(v) else default
            except Exception:
                return default

        def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
            try:
                v = float(value)
                return v if np.isfinite(v) else default
            except Exception:
                return default

        def _safe_mean(values: List[float]) -> Optional[float]:
            if not values:
                return None
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                return None
            val = float(np.nanmean(arr))
            return val if np.isfinite(val) else None

        def _safe_std(values: List[float]) -> Optional[float]:
            if not values:
                return None
            arr = np.array(values, dtype=float)
            if arr.size == 0:
                return None
            val = float(np.nanstd(arr))
            return val if np.isfinite(val) else None

        def _safe_div(numerator: float, denominator: float) -> Optional[float]:
            if denominator == 0:
                return None
            try:
                val = numerator / denominator
                return val if np.isfinite(val) else None
            except Exception:
                return None

        face_detection_rate = _safe_div(frames_with_faces, frames_analyzed)
        face_detection_rate = (face_detection_rate * 100.0) if face_detection_rate is not None else None

        avg_faces_per_frame = _safe_div(faces_detected, frames_analyzed)

        return {
            "emotion_distribution": {
                "mean": {k: _safe_float(v) for k, v in emotion_means.items()},
                "variance": {k: _safe_float(v) for k, v in emotion_variances.items()},
                "dominant_emotion": dominant_emotion,
                "dominant_emotion_score": _safe_float(dominant_emotion_score),
                "dominant_emotion_by_frequency": {
                    "emotion": dominant_freq,
                    "count": _safe_int(dominant_freq_count),
                },
            },
            "pose_summary": pose_stats,
            "face_detection_rate": _safe_float(face_detection_rate),
            "general_statistics": {
                "total_frames": _safe_int(total_frames),
                "frames_analyzed": _safe_int(frames_analyzed),
                "frames_with_faces": _safe_int(frames_with_faces),
                "faces_detected": _safe_int(faces_detected),
                "avg_faces_per_frame": _safe_float(avg_faces_per_frame),
                "frame_skip": _safe_int(frame_skip),
                "fps": _safe_float(fps),
            },
        }


if __name__ == "__main__":
    print("PyFeatSummarizer modülü hazır.")
