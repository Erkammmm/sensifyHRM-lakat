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
        window_size_seconds: float = 5.0,
        silence_sampling_seconds: float = 12.0,
        speech_sampling_seconds: float = 5.0,
        face_confidence_threshold: float = 0.0,
    ):
        self.frame_skip = max(1, int(frame_skip))
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.window_size_seconds = float(window_size_seconds)
        self.silence_sampling_seconds = float(silence_sampling_seconds)
        self.speech_sampling_seconds = float(speech_sampling_seconds)
        self.face_confidence_threshold = float(face_confidence_threshold)
        self._detector = None
        self._window_cache: Dict[str, Optional[List[Dict[str, Any]]]] = {}
        self._last_cache_key: Optional[str] = None
        self._no_face_sentinel = object()

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

    @staticmethod
    def _extract_face_confidence(frame_result: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(frame_result, dict):
            return None
        # Olası alan adları (mediapipe veya başka kaynaklar)
        for key in ("face_confidence", "detection_confidence", "confidence", "score"):
            val = frame_result.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            return v if math.isfinite(v) else default
        except Exception:
            return default

    @staticmethod
    def _extract_facebox_confidence(face_entry: Dict[str, Any]) -> Optional[float]:
        facebox = face_entry.get("face_box") if isinstance(face_entry, dict) else None
        if not isinstance(facebox, dict):
            return None
        for key in ("confidence", "score", "probability"):
            val = facebox.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def _reset_cache_if_needed(self, cache_key: str) -> None:
        if self._last_cache_key != cache_key:
            self._window_cache = {}
            self._last_cache_key = cache_key

    def _select_window_indices(
        self,
        total_frames: int,
        fps: float,
        voice_summary: Optional[Dict[str, Any]],
        window_size_seconds: float,
        silence_sampling_seconds: float,
        speech_sampling_seconds: float,
    ) -> List[int]:
        if total_frames <= 0 or fps <= 0:
            return []
        duration_seconds = total_frames / float(fps)
        window_size_seconds = max(1.0, window_size_seconds)
        total_windows = int(math.ceil(duration_seconds / window_size_seconds))
        rms_series = {}
        if isinstance(voice_summary, dict):
            rms_series = voice_summary.get("rms_energy_series", {})
            if "raw_voice_features" in voice_summary:
                rms_series = voice_summary.get("raw_voice_features", {}).get("rms_energy_series", rms_series)

        rms_times = rms_series.get("times") or []
        rms_values = rms_series.get("values") or []
        rms_threshold = None
        if rms_values:
            vals = np.array(rms_values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                rms_threshold = float(np.percentile(vals, 60))

        speech_stride = max(1, int(round(speech_sampling_seconds / window_size_seconds)))
        silence_stride = max(1, int(round(silence_sampling_seconds / window_size_seconds)))

        window_indices = []
        for w in range(total_windows):
            window_start = w * window_size_seconds
            window_end = min((w + 1) * window_size_seconds, duration_seconds)
            is_speech = True
            if rms_times and rms_values and rms_threshold is not None:
                # Bu pencere içindeki RMS ortalamasını hesapla
                window_vals = [
                    v for t, v in zip(rms_times, rms_values) if window_start <= t < window_end
                ]
                if window_vals:
                    mean_val = float(np.mean(window_vals))
                    is_speech = mean_val >= rms_threshold
            # Sessiz bölgelerde daha seyrek örnekle
            stride = speech_stride if is_speech else silence_stride
            if w % stride == 0:
                window_indices.append(w)
        return window_indices

    def _detect_images_with_fallback(self, rgb_frames: List[np.ndarray]) -> Any:
        # Py-Feat batch destekliyorsa küçük batch ile çalıştır
        try:
            return self._detector.detect_image(rgb_frames, batch_size=self.batch_size)
        except TypeError:
            return self._detector.detect_image(rgb_frames)
        except Exception:
            # Numpy array desteklenmiyorsa dosya path fallback
            with tempfile.TemporaryDirectory() as tmpdir:
                image_paths = []
                for idx, rgb in enumerate(rgb_frames):
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    path = os.path.join(tmpdir, f"frame_{idx}.png")
                    cv2.imwrite(path, bgr)
                    image_paths.append(path)
                try:
                    return self._detector.detect_image(image_paths, batch_size=self.batch_size)
                except TypeError:
                    return self._detector.detect_image(image_paths)
                except Exception:
                    # Tek tek dene (kırılma olmasın)
                    results = []
                    for path in image_paths:
                        try:
                            results.append(self._detector.detect_image([path], batch_size=1))
                        except Exception:
                            results.append(None)
                    return results

    def _build_faces_map(self, fex, frame_index_map: List[int]) -> Dict[Tuple[int, int], Dict[str, Any]]:
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

        def _map_frame(local_frame: int) -> Optional[int]:
            if local_frame < 0 or local_frame >= len(frame_index_map):
                return None
            return frame_index_map[local_frame]

        for idx, rec in enumerate(emotion_records):
            local_frame, face = self._get_frame_face(rec, idx)
            frame = _map_frame(local_frame)
            if frame is None:
                continue
            emotions = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("emotions", emotions)

        for idx, rec in enumerate(pose_records):
            local_frame, face = self._get_frame_face(rec, idx)
            frame = _map_frame(local_frame)
            if frame is None:
                continue
            pose = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("pose", pose)

        for idx, rec in enumerate(au_records):
            local_frame, face = self._get_frame_face(rec, idx)
            frame = _map_frame(local_frame)
            if frame is None:
                continue
            aus = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("aus", aus)

        for idx, rec in enumerate(facebox_records):
            local_frame, face = self._get_frame_face(rec, idx)
            frame = _map_frame(local_frame)
            if frame is None:
                continue
            facebox = {k: float(v) for k, v in rec.items() if k not in ("frame", "Frame", "frame_idx", "face", "face_id", "Face")}
            faces_map[(frame, face)].setdefault("face_box", facebox)

        if landmark_records:
            for idx, rec in enumerate(landmark_records):
                local_frame, face = self._get_frame_face(rec, idx)
                frame = _map_frame(local_frame)
                if frame is None:
                    continue
                landmarks = self._extract_landmarks_from_record(rec)
                if landmarks is not None:
                    faces_map[(frame, face)].setdefault("landmarks", landmarks)
                else:
                    faces_map[(frame, face)].setdefault("landmarks_raw", rec)
        elif hasattr(landmarks_df, "values"):
            try:
                lm_values = np.array(landmarks_df.values)
                for idx, row in enumerate(lm_values):
                    frame = _map_frame(idx)
                    if frame is None:
                        continue
                    if row.ndim == 2:
                        landmarks = row.tolist()
                    else:
                        if len(row) % 2 == 0:
                            landmarks = row.reshape(-1, 2).tolist()
                        else:
                            landmarks = row.tolist()
                    faces_map[(frame, 0)].setdefault("landmarks", landmarks)
            except Exception:
                pass

        return faces_map

    def analyze_frames_windowed(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        voice_summary: Optional[Dict[str, Any]] = None,
        face_presence_frames: Optional[List[Optional[Dict[str, Any]]]] = None,
        window_size_seconds: Optional[float] = None,
        silence_sampling_seconds: Optional[float] = None,
        speech_sampling_seconds: Optional[float] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Zaman pencereli Py-Feat analizi.
        Her pencere için yalnızca 1 frame seçilir (hız optimizasyonu).
        """
        if not frames:
            return {
                "frame_analysis": [],
                "metadata": {
                    "status": "empty",
                    "message": "Frame listesi boş",
                    "fps": float(fps),
                },
            }

        self._init_detector()

        if fps <= 0:
            fps = 30.0

        window_size_seconds = float(window_size_seconds or self.window_size_seconds)
        silence_sampling_seconds = float(silence_sampling_seconds or self.silence_sampling_seconds)
        speech_sampling_seconds = float(speech_sampling_seconds or self.speech_sampling_seconds)

        total_frames = len(frames)
        cache_key = cache_key or f"{total_frames}:{fps:.3f}"
        self._reset_cache_if_needed(cache_key)

        window_indices = self._select_window_indices(
            total_frames=total_frames,
            fps=fps,
            voice_summary=voice_summary,
            window_size_seconds=window_size_seconds,
            silence_sampling_seconds=silence_sampling_seconds,
            speech_sampling_seconds=speech_sampling_seconds,
        )

        selected_frame_indices: List[int] = []
        window_id_map: Dict[int, int] = {}

        for w in window_indices:
            window_start = int(round(w * window_size_seconds * fps))
            window_end = min(int(round((w + 1) * window_size_seconds * fps)), total_frames)
            if window_end <= window_start:
                continue

            # Duygu trendi segment-level olduğundan, pencere başına 1 temsilci frame seçmek güvenlidir.
            candidate_indices = list(range(window_start, window_end))
            if face_presence_frames is not None:
                face_candidates = []
                for idx in candidate_indices:
                    if idx >= len(face_presence_frames):
                        continue
                    if face_presence_frames[idx] is None:
                        continue
                    conf = self._extract_face_confidence(face_presence_frames[idx])
                    if conf is not None and conf < self.face_confidence_threshold:
                        continue
                    face_candidates.append((idx, conf))
                if not face_candidates:
                    # Yüz yoksa py-feat çalıştırma
                    self._window_cache.setdefault(f"{cache_key}:{w}", self._no_face_sentinel)
                    continue
                # Tercih: yüksek güven, yoksa orta frame
                face_candidates.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
                selected_idx = face_candidates[0][0]
            else:
                selected_idx = candidate_indices[len(candidate_indices) // 2]

            cache_entry = self._window_cache.get(f"{cache_key}:{w}")
            if cache_entry is self._no_face_sentinel:
                continue
            if cache_entry is not None:
                # Cache'den gelen sonuçları ekle
                frame_analysis.extend(cache_entry)
                continue
            if cache_entry is None:
                selected_frame_indices.append(selected_idx)
                window_id_map[selected_idx] = w

        if not selected_frame_indices:
            return {
                "frame_analysis": [],
                "metadata": {
                    "status": "ok",
                    "fps": float(fps),
                    "window_size_seconds": window_size_seconds,
                    "silence_sampling_seconds": silence_sampling_seconds,
                    "speech_sampling_seconds": speech_sampling_seconds,
                    "frames_analyzed": 0,
                    "total_frames": total_frames,
                },
            }

        # Seçilen frame'leri batch olarak çalıştır
        rgb_frames = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in selected_frame_indices]
        try:
            fex = self._detect_images_with_fallback(rgb_frames)
        except Exception:
            fex = None

        frame_analysis: List[Dict[str, Any]] = []

        # detect_image list yerine tek tek döndüyse birleştir
        if isinstance(fex, list):
            for local_idx, sub_fex in enumerate(fex):
                if sub_fex is None:
                    continue
                actual_idx = selected_frame_indices[local_idx]
                faces_map = self._build_faces_map(sub_fex, [actual_idx])
                for (frame_idx, face_idx), payload in faces_map.items():
                    if frame_idx != actual_idx:
                        continue
                    face_entry = {"face_id": int(face_idx)}
                    face_entry.update(payload)
                    conf = self._extract_facebox_confidence(face_entry)
                    if conf is not None and conf < self.face_confidence_threshold:
                        continue
                    frame_analysis.append(
                        {
                            "frame_index": int(frame_idx),
                            "timestamp": float(frame_idx / fps),
                            "faces": [face_entry],
                        }
                    )
        elif fex is not None:
            faces_map = self._build_faces_map(fex, selected_frame_indices)
            for frame_idx in selected_frame_indices:
                faces = []
                for (f_idx, face_idx), payload in faces_map.items():
                    if f_idx != frame_idx:
                        continue
                    face_entry = {"face_id": int(face_idx)}
                    face_entry.update(payload)
                    conf = self._extract_facebox_confidence(face_entry)
                    if conf is not None and conf < self.face_confidence_threshold:
                        continue
                    faces.append(face_entry)
                if not faces:
                    continue
                frame_analysis.append(
                    {
                        "frame_index": int(frame_idx),
                        "timestamp": float(frame_idx / fps),
                        "faces": faces,
                    }
                )

        # Cache'e yaz
        for entry in frame_analysis:
            frame_idx = entry.get("frame_index")
            window_id = window_id_map.get(frame_idx)
            if window_id is None:
                continue
            self._window_cache[f"{cache_key}:{window_id}"] = [entry]

        return {
            "frame_analysis": frame_analysis,
            "metadata": {
                "status": "ok",
                "fps": float(fps),
                "sampling_mode": "windowed",
                "window_size_seconds": window_size_seconds,
                "silence_sampling_seconds": silence_sampling_seconds,
                "speech_sampling_seconds": speech_sampling_seconds,
                "frames_analyzed": len(frame_analysis),
                "total_frames": total_frames,
                "batch_size": self.batch_size,
            },
        }

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
