"""
Frame Analysis Summarizer
Ham frame_analysis verisini sadeleştirir ve mülakat karakteristiklerini çıkarır.
"""

from typing import Dict, List, Optional
import numpy as np


class FrameAnalysisSummarizer:
    """
    Ham frame_analysis verisini özetler ve Gemini API'ye göndermek için
    sadeleştirilmiş bir rapor üretir.
    """

    @staticmethod
    def summarize(frame_analysis: List[Optional[Dict]]) -> Dict:
        """
        Frame analysis listesini özetler.

        Args:
            frame_analysis: Her frame için {
                "timestamp": float,
                "raw_features": {
                    "mouth_width_norm": float,
                    "mouth_height_norm": float,
                    "eye_opening_norm": float,
                    "brow_distance_norm": float,
                    "jaw_open_norm": float,
                },
                "gaze": {
                    "x": float,
                    "y": float,
                    "direction": str,
                }
            }

        Returns:
            Özetlenmiş rapor:
            {
                "general_statistics": {...},
                "emotion_change_points": [...],
                "gaze_summary": {...},
                "cognitive_load_score": {...}
            }
        """
        # Geçerli frame'leri filtrele
        valid_frames = [f for f in frame_analysis if f is not None]
        if not valid_frames:
            return {
                "general_statistics": {},
                "emotion_change_points": [],
                "gaze_summary": {},
                "cognitive_load_score": {},
            }

        # 1. Genel İstatistikler
        general_stats = FrameAnalysisSummarizer._compute_general_statistics(valid_frames)

        # 2. Duygu Değişim Noktaları
        change_points = FrameAnalysisSummarizer._detect_emotion_changes(valid_frames)

        # 3. Göz Analizi Özeti
        gaze_summary = FrameAnalysisSummarizer._summarize_gaze(valid_frames)

        # 4. Bilişsel Yük Skoru
        cognitive_load = FrameAnalysisSummarizer._compute_cognitive_load(valid_frames)

        return {
            "general_statistics": general_stats,
            "emotion_change_points": change_points,
            "gaze_summary": gaze_summary,
            "cognitive_load_score": cognitive_load,
        }

    @staticmethod
    def _compute_general_statistics(frames: List[Dict]) -> Dict:
        """Tüm mülakat boyunca ortalama ham özellikleri hesaplar."""
        if not frames:
            return {}

        # Ham özellikleri topla
        mouth_widths = []
        mouth_heights = []
        eye_openings = []
        brow_distances = []
        jaw_opens = []
        gaze_center_count = 0

        for frame in frames:
            rf = frame.get("raw_features", {})
            if rf:
                mouth_widths.append(rf.get("mouth_width_norm", 0.0))
                mouth_heights.append(rf.get("mouth_height_norm", 0.0))
                eye_openings.append(rf.get("eye_opening_norm", 0.0))
                brow_distances.append(rf.get("brow_distance_norm", 0.0))
                jaw_opens.append(rf.get("jaw_open_norm", 0.0))

            gaze = frame.get("gaze", {})
            if gaze.get("direction") == "center":
                gaze_center_count += 1

        total_frames = len(frames)
        gaze_center_percentage = (
            (gaze_center_count / total_frames * 100.0) if total_frames > 0 else 0.0
        )

        return {
            "average_features": {
                "mouth_width_norm": float(np.mean(mouth_widths)) if mouth_widths else 0.0,
                "mouth_height_norm": float(np.mean(mouth_heights)) if mouth_heights else 0.0,
                "eye_opening_norm": float(np.mean(eye_openings)) if eye_openings else 0.0,
                "brow_distance_norm": float(np.mean(brow_distances)) if brow_distances else 0.0,
                "jaw_open_norm": float(np.mean(jaw_opens)) if jaw_opens else 0.0,
            },
            "gaze_center_percentage": float(gaze_center_percentage),
            "total_frames_analyzed": total_frames,
        }

    @staticmethod
    def _detect_emotion_changes(frames: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        Ham özelliklerde radikal değişimleri tespit eder.
        Bir özellik %threshold (örn. %30) oranında değiştiğinde kaydeder.
        """
        if len(frames) < 2:
            return []

        change_points = []
        feature_names = [
            "mouth_width_norm",
            "mouth_height_norm",
            "eye_opening_norm",
            "brow_distance_norm",
            "jaw_open_norm",
        ]

        # Önceki frame'in değerlerini sakla
        prev_features = {}
        for frame in frames:
            rf = frame.get("raw_features", {})
            if not rf:
                continue

            timestamp = frame.get("timestamp", 0.0)

            # İlk frame ise sadece sakla
            if not prev_features:
                prev_features = {name: rf.get(name, 0.0) for name in feature_names}
                continue

            # Her özellik için değişimi kontrol et
            for name in feature_names:
                prev_val = prev_features.get(name, 0.0)
                curr_val = rf.get(name, 0.0)

                if prev_val == 0.0:
                    continue

                # Yüzde değişim hesapla
                change_ratio = abs((curr_val - prev_val) / prev_val) if prev_val != 0 else 0.0

                if change_ratio >= threshold:
                    change_points.append(
                        {
                            "timestamp": float(timestamp),
                            "feature": name,
                            "previous_value": float(prev_val),
                            "current_value": float(curr_val),
                            "change_percentage": float(change_ratio * 100.0),
                        }
                    )

            # Güncel değerleri sakla
            prev_features = {name: rf.get(name, 0.0) for name in feature_names}

        return change_points

    @staticmethod
    def _summarize_gaze(frames: List[Dict]) -> Dict:
        """Gözlerin hangi yöne ne kadar süre baktığını özetler."""
        if not frames:
            return {}

        direction_counts = {}
        total = len(frames)

        for frame in frames:
            gaze = frame.get("gaze", {})
            direction = gaze.get("direction", "center")
            direction_counts[direction] = direction_counts.get(direction, 0) + 1

        # Yüzdeleri hesapla
        direction_percentages = {}
        for direction, count in direction_counts.items():
            direction_percentages[direction] = float((count / total * 100.0) if total > 0 else 0.0)

        return {
            "direction_counts": direction_counts,
            "direction_percentages": direction_percentages,
            "total_frames": total,
        }

    @staticmethod
    def _compute_cognitive_load(frames: List[Dict]) -> Dict:
        """
        Bilişsel Yük Skoru: jaw_open_norm yüksekken (konuşurken)
        gaze_direction merkez dışındaysa bunu 'Thinking/Reading' olarak işaretle.
        """
        if not frames:
            return {}

        thinking_reading_count = 0
        total_speaking_frames = 0

        # jaw_open_norm için eşik (konuşma tespiti)
        jaw_open_threshold = 0.4  # Bu değer ayarlanabilir

        for frame in frames:
            rf = frame.get("raw_features", {})
            gaze = frame.get("gaze", {})

            jaw_open = rf.get("jaw_open_norm", 0.0)
            gaze_dir = gaze.get("direction", "center")

            # Konuşma tespiti
            if jaw_open >= jaw_open_threshold:
                total_speaking_frames += 1

                # Konuşurken merkez dışına bakıyorsa
                if gaze_dir != "center":
                    thinking_reading_count += 1

        total_frames = len(frames)
        thinking_reading_percentage = (
            (thinking_reading_count / total_frames * 100.0) if total_frames > 0 else 0.0
        )
        speaking_percentage = (
            (total_speaking_frames / total_frames * 100.0) if total_frames > 0 else 0.0
        )

        return {
            "thinking_reading_count": thinking_reading_count,
            "thinking_reading_percentage": float(thinking_reading_percentage),
            "total_speaking_frames": total_speaking_frames,
            "speaking_percentage": float(speaking_percentage),
            "jaw_open_threshold_used": float(jaw_open_threshold),
        }


if __name__ == "__main__":
    print("FrameAnalysisSummarizer modülü hazır.")
