"""
Mediapipe tabanlı yüz + iris + ham özellikler & bakış yönü analizi.

- DeepFace ve MobileGaze yerine sadece Mediapipe FaceMesh (refine_landmarks=True) kullanılır.
- Her işlenen frame için şu formatta çıktı üretir:
  {
      "timestamp": float,
      "raw_features": {
          "mouth_width_norm": float,   # normalize edilmiş ağız genişliği
          "mouth_height_norm": float,  # normalize edilmiş ağız yüksekliği
          "eye_opening_norm": float,   # normalize edilmiş göz açıklığı
          "brow_distance_norm": float, # normalize edilmiş kaş-göz mesafesi
          "jaw_open_norm": float,      # normalize edilmiş çene açıklığı
      },
      "gaze": {
          "x": float,
          "y": float,
          "direction": str,  # "left", "right", "up", "down", "center"
      },
  }
"""

from typing import List, Dict, Optional

import os
import cv2
import numpy as np


class KalmanFilter1D:
    """
    Basit 1D Kalman filtresi.
    Hem duygular (skorlar) hem de bakış vektörü (x,y) için gürültü azaltmakta kullanılır.
    """

    def __init__(self, process_variance: float = 1e-3, measurement_variance: float = 5e-2):
        self.Q = float(process_variance)
        self.R = float(measurement_variance)
        self.x = 0.0
        self.P = 1.0
        self.initialized = False

    def update(self, z: float) -> float:
        z = float(z)
        if not self.initialized:
            self.x = z
            self.initialized = True
            return z
        # Prediction
        self.P = self.P + self.Q
        # Update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return float(self.x)


class MediapipeFaceGazeAnalyzer:
    """
    Tek bir Mediapipe FaceMesh çözümü ile hem basit duygu skorları
    (blendshape benzeri ölçümler) hem de iris tabanlı bakış yönü hesaplar.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        frame_skip: int = 3,
        use_kalman_gaze: bool = True,
        use_kalman_features: bool = True,
    ):
        """
        Args:
            min_detection_confidence: Yüz tespiti için minimum güven skoru.
            min_tracking_confidence: Takip için minimum güven skoru.
            frame_skip: Her kaç karede bir analiz yapılacağı (1 = her kare).
            use_kalman_gaze: Gaze için Kalman filtresi kullan (gürültü azaltma).
        """
        self.frame_skip = max(1, int(frame_skip))
        self.use_kalman_gaze = use_kalman_gaze
        self.use_kalman_features = use_kalman_features
        
        # Mediapipe parametreleri (lazy loading için sakla)
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        
        # Mediapipe'ı lazy loading ile başlat (ilk kullanımda)
        self._mp_face_mesh = None
        self._face_mesh = None

        # Gaze ve ham özellikler için Kalman filtreleri (gürültü azaltma)
        self._gaze_kf_x = KalmanFilter1D(process_variance=5e-2, measurement_variance=1e-1)
        self._gaze_kf_y = KalmanFilter1D(process_variance=5e-2, measurement_variance=1e-1)
        self._last_gaze = {"x": 0.0, "y": 0.0, "direction": "center"}
        self._feature_kf = {}

    # ---- Yardımcı metotlar ----

    @staticmethod
    def _get_landmark_xy(
        landmark, image_width: int, image_height: int
    ) -> np.ndarray:
        return np.array(
            [landmark.x * image_width, landmark.y * image_height], dtype=np.float32
        )

    def _compute_raw_features(
        self, landmarks, image_width: int, image_height: int
    ) -> Dict[str, float]:
        """
        Mediapipe FaceMesh landmark'larından HAM ölçümleri döndürür.
        Yorum yok, sadece geometrik ölçümler (normalize edilmiş).
        """
        # Göz noktaları
        left_eye_top = self._get_landmark_xy(landmarks[159], image_width, image_height)
        left_eye_bottom = self._get_landmark_xy(landmarks[145], image_width, image_height)
        right_eye_top = self._get_landmark_xy(landmarks[386], image_width, image_height)
        right_eye_bottom = self._get_landmark_xy(landmarks[374], image_width, image_height)
        
        # Kaş noktaları
        left_eyebrow = self._get_landmark_xy(landmarks[70], image_width, image_height)
        right_eyebrow = self._get_landmark_xy(landmarks[300], image_width, image_height)
        
        # Ağız noktaları
        left_mouth_corner = self._get_landmark_xy(landmarks[61], image_width, image_height)
        right_mouth_corner = self._get_landmark_xy(landmarks[291], image_width, image_height)
        upper_lip = self._get_landmark_xy(landmarks[13], image_width, image_height)
        lower_lip = self._get_landmark_xy(landmarks[14], image_width, image_height)
        chin = self._get_landmark_xy(landmarks[152], image_width, image_height)
        
        # Ham ölçümler
        brow_dist_left = np.linalg.norm(left_eye_top - left_eyebrow)
        brow_dist_right = np.linalg.norm(right_eye_top - right_eyebrow)
        brow_distance = float((brow_dist_left + brow_dist_right) / 2.0)
        
        mouth_width = np.linalg.norm(right_mouth_corner - left_mouth_corner)
        mouth_height = np.linalg.norm(upper_lip - lower_lip)
        
        left_eye_opening = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_opening = np.linalg.norm(right_eye_top - right_eye_bottom)
        eye_opening = float((left_eye_opening + right_eye_opening) / 2.0)
        
        jaw_open_distance = np.linalg.norm(chin - upper_lip)
        
        # Normalizasyon: yüz ölçeği (iki göz arası mesafe)
        left_eye_center = (left_eye_top + left_eye_bottom) / 2.0
        right_eye_center = (right_eye_top + right_eye_bottom) / 2.0
        face_scale = np.linalg.norm(right_eye_center - left_eye_center)
        if face_scale < 1e-3:
            face_scale = 1.0
        
        # Normalize edilmiş ham özellikler
        features = {
            "mouth_width_norm": float(mouth_width / face_scale),
            "mouth_height_norm": float(mouth_height / face_scale),
            "eye_opening_norm": float(eye_opening / face_scale),
            "brow_distance_norm": float(brow_distance / face_scale),
            "jaw_open_norm": float(jaw_open_distance / face_scale),
        }

        if not self.use_kalman_features:
            return features

        # Zaman içinde ham özelliklere hafif Kalman filtresi uygula (jitter azaltma)
        if not self._feature_kf:
            for name in features.keys():
                # Çok küçük process_variance → sabit dururken değerler sakin, hareket edince yavaşça takip eder
                self._feature_kf[name] = KalmanFilter1D(
                    process_variance=1e-4,
                    measurement_variance=5e-3,
                )

        smoothed = {}
        for name, value in features.items():
            kf = self._feature_kf.get(name)
            if kf is None:
                kf = KalmanFilter1D(process_variance=1e-4, measurement_variance=5e-3)
                self._feature_kf[name] = kf
            smoothed[name] = float(kf.update(value))

        return smoothed

    def _init_gaze_baseline(self):
        """Gaze kalibrasyonu için baseline değişkenlerini oluşturur."""
        if not hasattr(self, "_gaze_baseline_initialized"):
            self._gaze_baseline_initialized = True
            self._gaze_baseline_samples = []  # avg_pos listesi
            self._gaze_baseline_mean = np.array([0.5, 0.5], dtype=np.float32)
            self._gaze_baseline_std = np.array([0.1, 0.1], dtype=np.float32)
            self._gaze_baseline_ready = False

    def _update_gaze_baseline(self, avg_pos: np.ndarray):
        """İlk birkaç saniyedeki iris pozisyonlarından baseline istatistiklerini çıkarır."""
        self._gaze_baseline_samples.append(avg_pos)
        if len(self._gaze_baseline_samples) >= 15:
            arr = np.stack(self._gaze_baseline_samples, axis=0)
            self._gaze_baseline_mean = arr.mean(axis=0)
            self._gaze_baseline_std = arr.std(axis=0)
            # Çok küçük std değerleri aşırı büyük sapma üretmesin diye alt/üst sınır koy.
            self._gaze_baseline_std = np.clip(self._gaze_baseline_std, 0.1, 0.5)
            self._gaze_baseline_ready = True

    def _compute_gaze_from_iris(
        self, landmarks, image_width: int, image_height: int, timestamp: float
    ) -> Dict[str, object]:
        """
        Iris landmarklarından 2D bakış vektörü ve yön etiketi üretir.

        - İlk ~2 saniyede toplanan iris pozisyonları ile bir "merkez" (baseline)
          ve standart sapma estimasyonu yapılır.
        - Sonraki frame'lerde bakış, bu baseline'a göre normalize edilir.
        """
        self._init_gaze_baseline()

        # Sol göz için ana noktalar
        left_eye_outer = self._get_landmark_xy(
            landmarks[33], image_width, image_height
        )
        left_eye_inner = self._get_landmark_xy(
            landmarks[133], image_width, image_height
        )
        left_eye_top = self._get_landmark_xy(
            landmarks[159], image_width, image_height
        )
        left_eye_bottom = self._get_landmark_xy(
            landmarks[145], image_width, image_height
        )

        # Sağ göz için ana noktalar
        right_eye_outer = self._get_landmark_xy(
            landmarks[263], image_width, image_height
        )
        right_eye_inner = self._get_landmark_xy(
            landmarks[362], image_width, image_height
        )
        right_eye_top = self._get_landmark_xy(
            landmarks[386], image_width, image_height
        )
        right_eye_bottom = self._get_landmark_xy(
            landmarks[374], image_width, image_height
        )

        # Göz açıklığı (blink tespiti için)
        left_eye_opening = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_opening = np.linalg.norm(right_eye_top - right_eye_bottom)
        eye_opening = float((left_eye_opening + right_eye_opening) / 2.0)

        # Iris merkezleri (refine_landmarks=True ile gelir)
        # Sol iris: 468–471 arası noktalar
        left_iris_points = [
            self._get_landmark_xy(landmarks[i], image_width, image_height)
            for i in range(468, 472)
        ]
        right_iris_points = [
            self._get_landmark_xy(landmarks[i], image_width, image_height)
            for i in range(473, 477)
        ]
        left_iris_center = np.mean(left_iris_points, axis=0)
        right_iris_center = np.mean(right_iris_points, axis=0)

        # Her göz için iris'in göz kutusu içindeki normalize konumu (0-1)
        def _norm_pos(center, horiz_a, horiz_b, vert_a, vert_b) -> np.ndarray:
            """
            İris merkezini göz kutusu içinde 0-1 aralığına normalize eder.
            Değerler clamp edilerek 0-1 aralığında tutulur (taşmaları engellemek için).
            """
            x_min = min(horiz_a[0], horiz_b[0])
            x_max = max(horiz_a[0], horiz_b[0])
            y_min = min(vert_a[1], vert_b[1])
            y_max = max(vert_a[1], vert_b[1])
            width = max(x_max - x_min, 1e-3)
            height = max(y_max - y_min, 1e-3)
            x_norm = (center[0] - x_min) / width
            y_norm = (center[1] - y_min) / height
            x_norm = float(np.clip(x_norm, 0.0, 1.0))
            y_norm = float(np.clip(y_norm, 0.0, 1.0))
            return np.array([x_norm, y_norm], dtype=np.float32)

        left_pos = _norm_pos(
            left_iris_center,
            left_eye_outer,
            left_eye_inner,
            left_eye_top,
            left_eye_bottom,
        )
        right_pos = _norm_pos(
            right_iris_center,
            right_eye_outer,
            right_eye_inner,
            right_eye_top,
            right_eye_bottom,
        )

        # İki gözün ortalamasını al ve [0,1] aralığına sıkıştır
        avg_pos = (left_pos + right_pos) / 2.0  # [0,1] aralığında
        avg_pos = np.clip(avg_pos, 0.0, 1.0)

        # Göz açıklığı normalizasyonu (blink için threshold)
        # Göz genişliği ile normalize edelim
        eye_width = np.linalg.norm(left_eye_inner - left_eye_outer)
        if eye_width < 1e-3:
            eye_width = 1.0
        eye_opening_norm = eye_opening / eye_width

        # İlk 2 saniyede baseline kalibrasyonu
        if timestamp <= 2.0:
            self._update_gaze_baseline(avg_pos)
            # Kalibrasyon aşamasında yönü zorla "center" yap
            gaze_vec = (avg_pos - self._gaze_baseline_mean) / self._gaze_baseline_std
            gaze_vec = np.clip(gaze_vec, -3.0, 3.0)
            return {
                "x": float(gaze_vec[0]),
                "y": float(gaze_vec[1]),
                "direction": "center",
            }

        # Baseline hazır ise ona göre normalize et, değilse kaba merkez 0.5'e göre
        if self._gaze_baseline_ready:
            gaze_vec = (avg_pos - self._gaze_baseline_mean) / self._gaze_baseline_std
        else:
            gaze_vec = (avg_pos - 0.5) * 2.0  # eski davranış (fallback)

        gaze_vec = np.clip(gaze_vec, -2.0, 2.0)

        # Blink anında gaze'i güncelleme (göz neredeyse kapalıysa)
        if eye_opening_norm < 0.18:
            # Önceki değeri koru
            prev = self._last_gaze
            return {
                "x": float(prev.get("x", 0.0)),
                "y": float(prev.get("y", 0.0)),
                "direction": prev.get("direction", "center"),
            }

        # Kalman filtresi ile x ve y bileşenlerini zaman içinde yumuşat
        if self.use_kalman_gaze:
            gaze_x = float(self._gaze_kf_x.update(gaze_vec[0]))  # negatif = sol, pozitif = sağ
            gaze_y = float(self._gaze_kf_y.update(gaze_vec[1]))  # negatif = yukarı, pozitif = aşağı
        else:
            gaze_x = float(gaze_vec[0])
            gaze_y = float(gaze_vec[1])

        # Basit ama biraz daha hassas eşikler (~0.7 std civarı)
        horiz_th = 0.7
        vert_th = 0.7
        direction = "center"
        if abs(gaze_x) > abs(gaze_y):
            if gaze_x < -horiz_th:
                direction = "left"
            elif gaze_x > horiz_th:
                direction = "right"
        else:
            if gaze_y < -vert_th:
                direction = "up"
            elif gaze_y > vert_th:
                direction = "down"

        result = {
            "x": float(gaze_x),
            "y": float(gaze_y),
            "direction": direction,
        }
        self._last_gaze = result
        return result

    def _ensure_face_mesh(self):
        """Mediapipe FaceMesh'i lazy loading ile başlatır."""
        if self._face_mesh is None:
            # Protobuf uyumluluğu için Python implementation'a zorla
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            # refine_landmarks=True -> iris noktalarını da içerir
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

    # ---- Ana API ----

    def process_frame(
        self, frame_bgr: np.ndarray, timestamp: float
    ) -> Optional[Dict[str, object]]:
        """
        Tek bir BGR frame için analiz yapar (canlı kamera kullanımı için).
        """
        self._ensure_face_mesh()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_result = self._face_mesh.process(rgb)

        if not mp_result.multi_face_landmarks:
            return None

        face_landmarks = mp_result.multi_face_landmarks[0].landmark

        # Yüz landmark'larının 2D koordinatları (çizim için)
        landmarks_xy = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks
        ]

        # Yüz bounding box'ı (tüm landmark'ların min/max'ı)
        xs = [pt[0] for pt in landmarks_xy]
        ys = [pt[1] for pt in landmarks_xy]
        x_min = max(0, min(xs))
        y_min = max(0, min(ys))
        x_max = min(w, max(xs))
        y_max = min(h, max(ys))
        bbox = (x_min, y_min, x_max, y_max)
        raw_features = self._compute_raw_features(face_landmarks, w, h)
        gaze = self._compute_gaze_from_iris(face_landmarks, w, h, timestamp)

        return {
            "timestamp": float(timestamp),
            "raw_features": raw_features,
            "gaze": gaze,
            "bbox": bbox,
            "landmarks": landmarks_xy,
        }

    def analyze_frames(
        self, frames: List[np.ndarray], fps: float
    ) -> List[Optional[Dict]]:
        """
        Video frame listesini analiz eder.

        Args:
            frames: BGR formatında frame listesi.
            fps: Videonun gerçek FPS değeri (timestamp için kullanılır).

        Returns:
            Her işlenen frame için:
            {
                'timestamp': float,
                'emotions': {...},
                'gaze': {...}
            }
            İşlenmeyen (skip edilen) kareler için None döner.
        """
        results: List[Optional[Dict]] = []
        if fps <= 0:
            fps = 30.0

        for idx, frame in enumerate(frames):
            # Frame skip
            if idx % self.frame_skip != 0:
                results.append(None)
                continue

            try:
                self._ensure_face_mesh()
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_result = self._face_mesh.process(rgb)

                if not mp_result.multi_face_landmarks:
                    results.append(None)
                    continue

                timestamp = float(idx / fps)

                face_landmarks = mp_result.multi_face_landmarks[0].landmark

                raw_features = self._compute_raw_features(face_landmarks, w, h)
                gaze = self._compute_gaze_from_iris(face_landmarks, w, h, timestamp)

                results.append(
                    {
                        "timestamp": timestamp,
                        "raw_features": raw_features,
                        "gaze": gaze,
                    }
                )
            except Exception as e:
                # İlk birkaç hata için log yaz, sonrasında sessizce devam et
                if not hasattr(self, "_error_count"):
                    self._error_count = 0
                if self._error_count < 3:
                    print(
                        f"[MediapipeFaceGazeAnalyzer] Hata (frame {idx}): {str(e)}"
                    )
                    self._error_count += 1
                results.append(None)

        return results


if __name__ == "__main__":
    print("MediapipeFaceGazeAnalyzer modülü hazır.")

