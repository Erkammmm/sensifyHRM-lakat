"""
Py-Feat Analyzer
CPU üzerinde, hız öncelikli Py-Feat analizi.
Frame bazlı duygu, kafa pozisyonu, landmark ve ham çıktıları üretir.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import cv2
import tempfile
import os
import math


class PyFeatAnalyzer:
    """
    Py-Feat tabanlı yüz analizi.
    Duygu, pose, landmark ve AU benzeri ham çıktıları frame bazlı döndürür.
    """

    def __init__(
        self,
        frame_skip: int = 5,
        batch_size: int = 8,
        device: str = "cpu",
    ):
        self.frame_skip = max(1, int(frame_skip))
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self._detector = None

    def _init_detector(self):
        if self._detector is not None:
            return

        # CPU-only ve hız öncelikli
        import os
        import warnings
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("TQDM_DISABLE", "1")
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            from feat import Detector
        except Exception as exc:
            raise ImportError("Py-Feat (feat) kütüphanesi bulunamadı.") from exc

        # Py-Feat farklı sürümlerde farklı parametre kabul edebilir
        try:
            self._detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                emotion_model="resmasknet",
                facepose_model="img2pose",
                au_model="xgb",
                device=self.device,
                verbose=False,
            )
        except TypeError:
            self._detector = Detector()

    @staticmethod
    def _df_to_records(df) -> List[Dict[str, Any]]:
        if df is None:
            return []
        try:
            return df.reset_index().to_dict(orient="records")
        except Exception:
            try:
                return df.to_dict(orient="records")
            except Exception:
                return []

    @staticmethod
    def _get_frame_face(record: Dict[str, Any], default_frame: int, default_face: int = 0) -> Tuple[int, int]:
        frame = record.get("frame", record.get("Frame", record.get("frame_idx", default_frame)))
        face = record.get("face", record.get("face_id", record.get("Face", default_face)))
        try:
            frame_val = float(frame)
            frame = int(frame_val) if math.isfinite(frame_val) else int(default_frame)
        except Exception:
            frame = int(default_frame)
        try:
            face_val = float(face)
            face = int(face_val) if math.isfinite(face_val) else int(default_face)
        except Exception:
            face = int(default_face)
        return frame, face

    @staticmethod
    def _extract_landmarks_from_record(record: Dict[str, Any]) -> Optional[List[List[float]]]:
        # x_0,y_0 formatı varsa landmark listesi üret
        x_cols = [k for k in record.keys() if str(k).startswith("x_")]
        y_cols = [k for k in record.keys() if str(k).startswith("y_")]
        if len(x_cols) == len(y_cols) and x_cols:
            landmarks = []
            for i in range(len(x_cols)):
                x_val = record.get(f"x_{i}")
                y_val = record.get(f"y_{i}")
                if x_val is None or y_val is None:
                    continue
                landmarks.append([float(x_val), float(y_val)])
            return landmarks if landmarks else None
        return None

    def analyze_frames(self, frames: List[np.ndarray], fps: float = 30.0) -> Dict[str, Any]:
        """
        Frame listesi üzerinde Py-Feat analizi yapar.

        Returns:
            {
                "frame_analysis": [...],
                "metadata": {...}
            }
        """
        if not frames:
            return {
                "frame_analysis": [],
                "metadata": {
                    "status": "empty",
                    "message": "Frame listesi boş",
                    "frame_skip": self.frame_skip,
                    "fps": float(fps),
                },
            }

        self._init_detector()

        frame_indices = list(range(0, len(frames), self.frame_skip))
        sampled_frames = [frames[i] for i in frame_indices]
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sampled_frames]

        # Önce direkt numpy array ile dene, olmazsa dosya path fallback kullan
        try:
            fex = self._detector.detect_image(rgb_frames, batch_size=self.batch_size)
        except Exception:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_paths = []
                for idx, frame in enumerate(sampled_frames):
                    path = os.path.join(tmpdir, f"frame_{idx}.png")
                    cv2.imwrite(path, frame)
                    image_paths.append(path)
                try:
                    fex = self._detector.detect_image(image_paths, batch_size=self.batch_size)
                except TypeError:
                    fex = self._detector.detect_image(image_paths)

        # Bileşenleri ayıkla
        emotions_df = getattr(fex, "emotions", None)
        pose_df = getattr(fex, "pose", getattr(fex, "poses", None))
        au_df = getattr(fex, "aus", None)
        facebox_df = getattr(fex, "facebox", None)
        landmarks_df = getattr(fex, "landmarks", None)

        emotion_records = self._df_to_records(emotions_df)
        pose_records = self._df_to_records(pose_df)
        au_records = self._df_to_records(au_df)
        facebox_records = self._df_to_records(facebox_df)
        landmark_records = self._df_to_records(landmarks_df)

        # Frame/face bazlı birleştirme
        faces_map: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(dict)

        for idx, rec in enumerate(emotion_records):
            frame, face = self._get_frame_face(rec, idx)
            emotions = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("emotions", emotions)

        for idx, rec in enumerate(pose_records):
            frame, face = self._get_frame_face(rec, idx)
            pose = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("pose", pose)

        for idx, rec in enumerate(au_records):
            frame, face = self._get_frame_face(rec, idx)
            aus = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("aus", aus)

        for idx, rec in enumerate(facebox_records):
            frame, face = self._get_frame_face(rec, idx)
            facebox = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("face_box", facebox)

        # Landmarklar (farklı formatları desteklemek için ekstra kontrol)
        if landmark_records:
            for idx, rec in enumerate(landmark_records):
                frame, face = self._get_frame_face(rec, idx)
                landmarks = self._extract_landmarks_from_record(rec)
                if landmarks is not None:
                    faces_map[(frame, face)].setdefault("landmarks", landmarks)
                else:
                    faces_map[(frame, face)].setdefault("landmarks_raw", rec)
        elif hasattr(landmarks_df, "values"):
            try:
                lm_values = np.array(landmarks_df.values)
                for idx, row in enumerate(lm_values):
                    if row.ndim == 2:
                        landmarks = row.tolist()
                    else:
                        if len(row) % 2 == 0:
                            landmarks = row.reshape(-1, 2).tolist()
                        else:
                            landmarks = row.tolist()
                    faces_map[(idx, 0)].setdefault("landmarks", landmarks)
            except Exception:
                pass

        # Frame bazlı çıktı oluştur (detect_image lokal indeks döner)
        frame_analysis: List[Dict[str, Any]] = []
        for local_idx, original_idx in enumerate(frame_indices):
            timestamp = float(original_idx / fps) if fps else 0.0
            faces = []
            for (frame_idx, face_idx), payload in faces_map.items():
                if frame_idx != local_idx:
                    continue
                face_entry = {"face_id": int(face_idx)}
                face_entry.update(payload)
                faces.append(face_entry)
            if not faces:
                # Yüz yoksa bu frame'i boş bırak
                frame_analysis.append(None)
                continue
            frame_analysis.append(
                {
                    "frame_index": int(original_idx),
                    "timestamp": float(timestamp),
                    "faces": faces,
                }
            )

        return {
            "frame_analysis": [f for f in frame_analysis if f is not None],
            "metadata": {
                "status": "ok",
                "frame_skip": self.frame_skip,
                "fps": float(fps),
                "frames_analyzed": len([f for f in frame_analysis if f is not None]),
                "total_frames": len(frames),
            },
        }

    def analyze_video(
        self,
        video_path: str,
        fps: float = 30.0,
        total_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Video dosyası üzerinden Py-Feat analizi yapar (doğrudan video girdi).
        """
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self._init_detector()

        try:
            fex = self._detector.detect_video(video_path)
        except AttributeError as exc:
            raise RuntimeError("Py-Feat detect_video desteklemiyor.") from exc

        # Bileşenleri ayıkla
        emotions_df = getattr(fex, "emotions", None)
        pose_df = getattr(fex, "pose", getattr(fex, "poses", None))
        au_df = getattr(fex, "aus", None)
        facebox_df = getattr(fex, "facebox", None)
        landmarks_df = getattr(fex, "landmarks", None)

        emotion_records = self._df_to_records(emotions_df)
        pose_records = self._df_to_records(pose_df)
        au_records = self._df_to_records(au_df)
        facebox_records = self._df_to_records(facebox_df)
        landmark_records = self._df_to_records(landmarks_df)

        faces_map: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(dict)

        for idx, rec in enumerate(emotion_records):
            frame, face = self._get_frame_face(rec, idx)
            emotions = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("emotions", emotions)

        for idx, rec in enumerate(pose_records):
            frame, face = self._get_frame_face(rec, idx)
            pose = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("pose", pose)

        for idx, rec in enumerate(au_records):
            frame, face = self._get_frame_face(rec, idx)
            aus = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("aus", aus)

        for idx, rec in enumerate(facebox_records):
            frame, face = self._get_frame_face(rec, idx)
            facebox = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("face_box", facebox)

        if landmark_records:
            for idx, rec in enumerate(landmark_records):
                frame, face = self._get_frame_face(rec, idx)
                landmarks = self._extract_landmarks_from_record(rec)
                if landmarks is not None:
                    faces_map[(frame, face)].setdefault("landmarks", landmarks)
                else:
                    faces_map[(frame, face)].setdefault("landmarks_raw", rec)
        elif hasattr(landmarks_df, "values"):
            try:
                lm_values = np.array(landmarks_df.values)
                for idx, row in enumerate(lm_values):
                    if row.ndim == 2:
                        landmarks = row.tolist()
                    else:
                        if len(row) % 2 == 0:
                            landmarks = row.reshape(-1, 2).tolist()
                        else:
                            landmarks = row.tolist()
                    faces_map[(idx, 0)].setdefault("landmarks", landmarks)
            except Exception:
                pass

        if total_frames is None or total_frames <= 0:
            if faces_map:
                max_frame = max([k[0] for k in faces_map.keys()])
                total_frames = max_frame + 1
            else:
                total_frames = 0

        frame_indices = list(range(0, int(total_frames), self.frame_skip)) if total_frames else []

        frame_analysis: List[Dict[str, Any]] = []
        for frame_idx in frame_indices:
            timestamp = float(frame_idx / fps) if fps else 0.0
            faces = []
            for (f_idx, face_idx), payload in faces_map.items():
                if f_idx != frame_idx:
                    continue
                face_entry = {"face_id": int(face_idx)}
                face_entry.update(payload)
                faces.append(face_entry)
            if not faces:
                frame_analysis.append(None)
                continue
            frame_analysis.append(
                {
                    "frame_index": int(frame_idx),
                    "timestamp": float(timestamp),
                    "faces": faces,
                }
            )

        return {
            "frame_analysis": [f for f in frame_analysis if f is not None],
            "metadata": {
                "status": "ok",
                "frame_skip": self.frame_skip,
                "fps": float(fps),
                "frames_analyzed": len([f for f in frame_analysis if f is not None]),
                "total_frames": int(total_frames),
            },
        }


if __name__ == "__main__":
    print("PyFeatAnalyzer modülü hazır.")
