"""
Yüz Analizi Modülü
MediaPipe FaceLandmarker (blendshape) ile duygu, bakış yönü ve göz kırpma analizi.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import Counter
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Ayarlar ---
RAPOR_SIKLIGI = 0.5  # Saniyede bir veri kaydet
TARGET_FPS = float(os.getenv("SENSIFYHR_FACE_TARGET_FPS", "5.0"))

# Gaze/attention calibration
CALIB_SEARCH_SEC = float(os.getenv("SENSIFYHR_GAZE_CALIB_SEARCH_SEC", "30"))
CALIB_WINDOW_SEC = float(os.getenv("SENSIFYHR_GAZE_CALIB_WINDOW_SEC", "3.0"))
READING_DOWN_SEC = float(os.getenv("SENSIFYHR_GAZE_READING_SEC", "2.5"))


class FaceAnalyzer:
    """
    MediaPipe FaceLandmarker kullanarak:
    - Blendshape skorlarından duygu çıkarımı (kural tabanlı)
    - Göz bakış yönü tespiti
    - Göz kırpma sayımı
    """

    def __init__(self, model_path=None):
        print(f"[{self.__class__.__name__}] Başlatılıyor... Model yükleniyor...")

        if model_path is None:
            # weights/ klasöründen model dosyasını bul
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "..", "weights", "face_landmarker.task")
            model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        # MediaPipe C++ backend Unicode yolları açamıyor (Türkçe karakter sorunu).
        # Bu yüzden dosyayı Python ile okuyup model_asset_buffer olarak veriyoruz.
        with open(model_path, "rb") as f:
            model_data = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print(f"[{self.__class__.__name__}] Hazır!")

    # ------------------------------------------------------------------
    # FAZ-3: Head pose + gaze calibration helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def _median(xs: List[float]) -> float:
        xs2 = [float(x) for x in xs if isinstance(x, (int, float))]
        if not xs2:
            return 0.0
        xs2.sort()
        mid = len(xs2) // 2
        if len(xs2) % 2 == 1:
            return xs2[mid]
        return 0.5 * (xs2[mid - 1] + xs2[mid])

    @staticmethod
    def _mad(xs: List[float], center: float) -> float:
        dev = [abs(float(x) - float(center)) for x in xs if isinstance(x, (int, float))]
        return FaceAnalyzer._median(dev) if dev else 0.0

    @staticmethod
    def _extract_head_pose_deg(det_result) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        MediaPipe FaceLandmarker video mode dönüş matrisinden (yaklaşık) yaw/pitch/roll çıkarır.
        Not: Mutlak doğruluk yerine, **tutarlı** ve **kalibrasyonla** kullanılabilir delta'lar hedeflenir.
        """
        mats = getattr(det_result, "facial_transformation_matrixes", None)
        if not mats:
            return None, None, None
        m0 = mats[0]
        # m0 genelde 4x4; farklı tiplerde tolist() olmayabilir -> numpy'a zorla
        try:
            arr = np.array(m0, dtype=np.float32)
        except Exception:
            try:
                arr = np.array(getattr(m0, "data", m0), dtype=np.float32)
            except Exception:
                return None, None, None

        if arr.size < 9:
            return None, None, None
        # 4x4 veya 3x3 normalize
        if arr.shape == (4, 4):
            Rm = arr[:3, :3]
        elif arr.shape == (3, 3):
            Rm = arr
        else:
            # bazı sürümlerde flat olabilir
            flat = arr.reshape(-1)
            if flat.size >= 16:
                Rm = flat[:16].reshape(4, 4)[:3, :3]
            elif flat.size >= 9:
                Rm = flat[:9].reshape(3, 3)
            else:
                return None, None, None

        # SciPy varsa Euler çıkarımı daha güvenli
        try:
            from scipy.spatial.transform import Rotation as _R

            rot = _R.from_matrix(Rm)
            # x=pitch, y=yaw, z=roll (kalibrasyon ile kullanılacak)
            pitch, yaw, roll = rot.as_euler("xyz", degrees=True).tolist()
            return float(yaw), float(pitch), float(roll)
        except Exception:
            # Basit fallback (kalibrasyon odaklı)
            r = Rm
            try:
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(r[2, 0])))))
                yaw = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
                roll = math.degrees(math.atan2(float(r[2, 1]), float(r[2, 2])))
                return float(yaw), float(pitch), float(roll)
            except Exception:
                return None, None, None

    @staticmethod
    def _compute_gaze_direction_calibrated(
        yaw: Optional[float],
        pitch: Optional[float],
        eye_down: float,
        eye_left: float,
        eye_right: float,
        eye_up: float,
        calib: Dict[str, float],
    ) -> str:
        """
        Calibrated gaze direction:
          - Default: BELIRSIZ
          - Direction seçimi: baseline'a göre delta + minimum kanıt
          - ON_SCREEN (Ekrana) = baş genel olarak baseline civarında (A yaklaşımı)
        """
        if yaw is None or pitch is None:
            return "Belirsiz"

        yaw0 = float(calib.get("yaw0", 0.0))
        pitch0 = float(calib.get("pitch0", 0.0))
        # eşikler: robust + taban minimum
        yaw_thr = float(calib.get("yaw_thr", 12.0))
        pitch_thr = float(calib.get("pitch_thr", 10.0))
        eye_thr = float(calib.get("eye_thr", 0.18))

        dyaw = float(yaw) - yaw0
        dpitch = float(pitch) - pitch0

        # Eye deltas (baseline)
        ed0 = float(calib.get("eye_down0", 0.0))
        el0 = float(calib.get("eye_left0", 0.0))
        er0 = float(calib.get("eye_right0", 0.0))
        eu0 = float(calib.get("eye_up0", 0.0))
        d_down = float(eye_down) - ed0
        d_left = float(eye_left) - el0
        d_right = float(eye_right) - er0
        d_up = float(eye_up) - eu0

        # Yön için kanıt puanları
        scores = {
            "Asagi bakiyor": max(0.0, (dpitch / pitch_thr), (d_down / eye_thr)),
            "Yukari bakiyor": max(0.0, (-dpitch / pitch_thr), (d_up / eye_thr)),
            "Saga bakiyor": max(0.0, (dyaw / yaw_thr), (d_left / eye_thr)),   # note: eyeLookInLeft -> sağ
            "Sola bakiyor": max(0.0, (-dyaw / yaw_thr), (d_right / eye_thr)), # note: eyeLookInRight -> sol
        }

        # En güçlü yön
        best_dir, best_score = max(scores.items(), key=lambda kv: kv[1])
        # ON_SCREEN koşulu (A): baş baseline yakınında ve güçlü yön kanıtı yoksa
        on_screen = (abs(dyaw) < yaw_thr) and (abs(dpitch) < pitch_thr)

        # minimum kanıt eşiği: 1.0 = eşik kadar sapma
        if best_score >= 1.0:
            return best_dir
        if on_screen:
            return "Ekrana Bakiyor"
        return "Belirsiz"

    @staticmethod
    def _calibrate_baseline(timeline: List[Dict], effective_fps: float) -> Dict[str, float]:
        """
        Otomatik kalibrasyon: ilk CALIB_SEARCH_SEC içinde en stabil (yaw/pitch varyansı düşük) pencereyi bul.
        """
        if not timeline:
            return {"yaw0": 0.0, "pitch0": 0.0, "yaw_thr": 12.0, "pitch_thr": 10.0, "eye_thr": 0.18}

        search_end = float(CALIB_SEARCH_SEC)
        win_sec = max(1.5, float(CALIB_WINDOW_SEC))
        win_n = max(5, int(round(win_sec * max(1.0, float(effective_fps)))))

        samples = [
            t for t in timeline
            if isinstance(t, dict)
            and float(t.get("timestamp", 0.0) or 0.0) <= search_end
            and t.get("yaw_deg") is not None
            and t.get("pitch_deg") is not None
        ]
        if len(samples) < win_n:
            samples = [t for t in timeline if t.get("yaw_deg") is not None and t.get("pitch_deg") is not None]
        if len(samples) < 5:
            return {"yaw0": 0.0, "pitch0": 0.0, "yaw_thr": 12.0, "pitch_thr": 10.0, "eye_thr": 0.18}

        # kayan pencere
        best = None
        best_score = None
        for i in range(0, max(1, len(samples) - win_n + 1)):
            w = samples[i : i + win_n]
            yaws = [float(x.get("yaw_deg")) for x in w if x.get("yaw_deg") is not None]
            pits = [float(x.get("pitch_deg")) for x in w if x.get("pitch_deg") is not None]
            if len(yaws) < 3 or len(pits) < 3:
                continue
            # stabilite skoru: varyans toplamı
            score = float(np.var(yaws) + np.var(pits))
            if best_score is None or score < best_score:
                best_score = score
                best = w

        if not best:
            best = samples[:win_n]

        yaw0 = FaceAnalyzer._median([float(x.get("yaw_deg")) for x in best if x.get("yaw_deg") is not None])
        pitch0 = FaceAnalyzer._median([float(x.get("pitch_deg")) for x in best if x.get("pitch_deg") is not None])
        yaw_mad = FaceAnalyzer._mad([float(x.get("yaw_deg")) for x in best if x.get("yaw_deg") is not None], yaw0)
        pitch_mad = FaceAnalyzer._mad([float(x.get("pitch_deg")) for x in best if x.get("pitch_deg") is not None], pitch0)

        # eye baselines
        eye_down0 = FaceAnalyzer._median([float(x.get("eye_down", 0.0) or 0.0) for x in best])
        eye_left0 = FaceAnalyzer._median([float(x.get("eye_left", 0.0) or 0.0) for x in best])
        eye_right0 = FaceAnalyzer._median([float(x.get("eye_right", 0.0) or 0.0) for x in best])
        eye_up0 = FaceAnalyzer._median([float(x.get("eye_up", 0.0) or 0.0) for x in best])

        # thresholds: robust spread + minimums
        yaw_thr = max(10.0, (3.0 * yaw_mad) + 6.0)
        pitch_thr = max(8.0, (3.0 * pitch_mad) + 5.0)
        eye_thr = 0.18  # conservative; can be env-tuned later

        return {
            "yaw0": float(yaw0),
            "pitch0": float(pitch0),
            "yaw_thr": float(yaw_thr),
            "pitch_thr": float(pitch_thr),
            "eye_thr": float(eye_thr),
            "eye_down0": float(eye_down0),
            "eye_left0": float(eye_left0),
            "eye_right0": float(eye_right0),
            "eye_up0": float(eye_up0),
        }

    @staticmethod
    def _compute_reading_events(timeline: List[Dict]) -> List[Dict]:
        """DOWN kesintisiz episode'larını çıkarır (>= READING_DOWN_SEC)."""
        events: List[Dict] = []
        if not timeline:
            return events

        cur_start = None
        cur_end = None
        for item in timeline:
            ts = float(item.get("timestamp", 0.0) or 0.0)
            gaze = (item.get("gaze") or "").strip().lower()
            is_down = "asagi" in gaze  # "Asagi bakiyor"
            if is_down:
                if cur_start is None:
                    cur_start = ts
                cur_end = ts
            else:
                if cur_start is not None and cur_end is not None:
                    dur = float(cur_end - cur_start)
                    if dur >= float(READING_DOWN_SEC):
                        events.append({"start": round(cur_start, 2), "end": round(cur_end, 2), "duration_sec": round(dur, 2)})
                cur_start = None
                cur_end = None

        if cur_start is not None and cur_end is not None:
            dur = float(cur_end - cur_start)
            if dur >= float(READING_DOWN_SEC):
                events.append({"start": round(cur_start, 2), "end": round(cur_end, 2), "duration_sec": round(dur, 2)})

        return events

    @staticmethod
    def _classify_emotion(scores):
        """Blendshape skorlarından duygu çıkarımı (kural tabanlı)."""
        brow_inner_up = scores.get("browInnerUp", 0)
        brow_down_left = scores.get("browDownLeft", 0)
        brow_down_right = scores.get("browDownRight", 0)
        brow_outer_up = (
            scores.get("browOuterUpLeft", 0) + scores.get("browOuterUpRight", 0)
        ) / 2
        eye_wide = (
            scores.get("eyeWideLeft", 0) + scores.get("eyeWideRight", 0)
        ) / 2
        smile_avg = (
            scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)
        ) / 2
        nose_sneer = (
            scores.get("noseSneerLeft", 0) + scores.get("noseSneerRight", 0)
        ) / 2
        jaw_open = scores.get("jawOpen", 0)

        # Korku
        if brow_inner_up > 0.2 and brow_outer_up > 0.1 and eye_wide > 0.1:
            return "Korku"
        # Tiksinti
        if nose_sneer > 0.15:
            return "Tiksinti"
        # Mutlu
        if smile_avg > 0.35:
            return "Mutlu"
        # Şaşkın
        if brow_inner_up > 0.4 and jaw_open > 0.10:
            return "Saskin"
        # Öfkeli
        if brow_down_left > 0.35 and brow_down_right > 0.35:
            return "Ofkeli"
        # Stresli
        stress_score = (
            scores.get("mouthRollLower", 0) + scores.get("mouthShrugLower", 0)
        ) / 3
        if stress_score > 0.25:
            return "Stresli"

        return "Notr"

    @staticmethod
    def _classify_facial_state(scores) -> str:
        """
        FAZ-3: Duygu etiketi değil, davranışsal facial_state üretir.
        - POSITIVE: belirgin gülümseme
        - TENSE: stres/gerginlik sinyali yüksek
        - NEUTRAL: diğer
        """
        smile_avg = (
            scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)
        ) / 2
        stress_score = (
            scores.get("mouthRollLower", 0) + scores.get("mouthShrugLower", 0)
        ) / 3
        brow_down = (
            scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)
        ) / 2

        if smile_avg > 0.35:
            return "POSITIVE"
        if stress_score > 0.25 or brow_down > 0.35:
            return "TENSE"
        return "NEUTRAL"

    @staticmethod
    def _stress_indicator(scores) -> str:
        """
        FAZ-3: Stress indicator state.
        """
        stress_score = (
            scores.get("mouthRollLower", 0) + scores.get("mouthShrugLower", 0)
        ) / 3
        if stress_score >= 0.35:
            return "HIGH"
        if stress_score >= 0.22:
            return "ELEVATED"
        return "LOW"

    @staticmethod
    def _detect_gaze(scores):
        """Göz bakış yönünü tespit eder."""
        look_in_left = scores.get("eyeLookInLeft", 0)
        look_in_right = scores.get("eyeLookInRight", 0)
        look_out_left = scores.get("eyeLookOutLeft", 0)
        look_out_right = scores.get("eyeLookOutRight", 0)
        look_down = (
            scores.get("eyeLookDownLeft", 0) + scores.get("eyeLookDownRight", 0)
        ) / 2

        if look_in_left > 0.5:
            return "Saga bakiyor"
        if look_in_right > 0.5:
            return "Sola bakiyor"
        if look_down > 0.6:
            return "Asagi bakiyor"
        if look_out_left > 0.5 or look_out_right > 0.5:
            return "Yukari bakiyor"

        return "Ekrana Bakiyor"

    @staticmethod
    def _attention_state_from_gaze(gaze: str) -> str:
        """
        FAZ-3: Attention state (FOCUSED / AVERTED) üretir.
        """
        return "FOCUSED" if isinstance(gaze, str) and "Ekrana" in gaze else "AVERTED"

    def process_video(self, video_path, show_video=False, phase3_enabled: bool = False):
        """
        Video dosyasını analiz eder.

        Args:
            video_path: Video dosya yolu
            show_video: Analiz sırasında pencere göster (debug)

        Returns:
            tuple: (timeline, summary)
                Phase-2 timeline: {"timestamp","emotion","gaze","blink_total"}
                Phase-3 timeline: {"timestamp","facial_state","attention_state","stress_indicator","gaze","blink_total"}
                summary: dict - Özet istatistikler (focus_score/blink_rate vb.)
        """
        if not os.path.exists(video_path):
            print(f"Video bulunamadı: {video_path}")
            return [], {}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0

        # Ürün optimizasyonu: her frame'i işlemek gereksiz pahalı; hedef FPS'e göre örnekle
        target_fps = TARGET_FPS if TARGET_FPS > 0 else 5.0
        frame_interval = max(1, int(round(float(fps) / float(target_fps))))
        effective_fps = float(fps) / float(frame_interval) if frame_interval > 0 else float(fps)
        print(f"Video Analizi Başladı (Kaynak FPS: {fps:.0f} → İşlenen ~{effective_fps:.1f} FPS)...")

        analysis_timeline = []
        blink_count = 0
        eye_closed = False
        last_record_time = -1.0
        frame_count = 0
        timestamp_ms = 0
        last_ts_ms = -1  # MediaPipe detect_for_video monotonik timestamp ister

        while cap.isOpened():
            # grab: decode maliyetini azaltır; sadece seçilen aralıklarda retrieve edilir
            grabbed = cap.grab()
            if not grabbed:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            success, frame = cap.retrieve()
            if not success:
                continue

            # Bazı codec'lerde CAP_PROP_POS_MSEC güvenilir değil (aynı/geri dönebiliyor).
            # MediaPipe VIDEO mode: timestamp'ler monotonik artmalı.
            raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            try:
                raw_ms = float(raw_ms)
            except Exception:
                raw_ms = -1.0

            if raw_ms is None or raw_ms <= 0:
                # fallback: frame index + fps
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                try:
                    frame_idx = float(frame_idx)
                except Exception:
                    frame_idx = float(frame_count)
                raw_ms = (frame_idx / float(fps)) * 1000.0 if fps > 0 else (frame_count * 33.333)

            ts_ms = int(round(raw_ms))
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            timestamp_ms = ts_ms
            timestamp_sec = float(ts_ms) / 1000.0

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.landmarker.detect_for_video(
                mp_image, int(ts_ms)
            )

            if detection_result and detection_result.face_blendshapes:
                shapes = detection_result.face_blendshapes[0]
                scores = {cat.category_name: cat.score for cat in shapes}
                # head pose (deg) + raw eye look signals (for calibrated gaze)
                yaw_deg, pitch_deg, roll_deg = self._extract_head_pose_deg(detection_result)
                eye_left = float(scores.get("eyeLookInLeft", 0.0) or 0.0)
                eye_right = float(scores.get("eyeLookInRight", 0.0) or 0.0)
                eye_up = float((scores.get("eyeLookOutLeft", 0.0) + scores.get("eyeLookOutRight", 0.0)) / 2.0)
                eye_down = float((scores.get("eyeLookDownLeft", 0.0) + scores.get("eyeLookDownRight", 0.0)) / 2.0)

                # Göz kırpma tespiti
                left_blink = scores.get("eyeBlinkLeft", 0)
                right_blink = scores.get("eyeBlinkRight", 0)
                if left_blink > 0.5 and right_blink > 0.5:
                    if not eye_closed:
                        blink_count += 1
                        eye_closed = True
                else:
                    eye_closed = False

                # Periyodik kayıt
                if timestamp_sec - last_record_time >= RAPOR_SIKLIGI:
                    gaze = self._detect_gaze(scores)  # geçici; kalibrasyon sonrası overwrite edilecek

                    if phase3_enabled:
                        facial_state = self._classify_facial_state(scores)
                        attention_state = self._attention_state_from_gaze(gaze)
                        stress_indicator = self._stress_indicator(scores)

                        # FAZ-3: Yetkinlik karnesi için max-pooling'e uygun sayısal sinyaller
                        smile_avg = (
                            scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)
                        ) / 2
                        brow_down = (
                            scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)
                        ) / 2
                        eye_wide = (
                            scores.get("eyeWideLeft", 0) + scores.get("eyeWideRight", 0)
                        ) / 2
                        jaw_open = scores.get("jawOpen", 0)
                        stress_score = (
                            scores.get("mouthRollLower", 0) + scores.get("mouthShrugLower", 0)
                        ) / 3

                        data_packet = {
                            "timestamp": round(timestamp_sec, 2),
                            "facial_state": facial_state,
                            "attention_state": attention_state,
                            "stress_indicator": stress_indicator,
                            # legacy-compatible fields (behavioral, not emotion)
                            "gaze": gaze,
                            "blink_total": blink_count,
                            # numeric cues (0..1 approx)
                            "smile_score": round(float(smile_avg), 4),
                            "brow_down_score": round(float(brow_down), 4),
                            "eye_wide_score": round(float(eye_wide), 4),
                            "jaw_open_score": round(float(jaw_open), 4),
                            "stress_score": round(float(stress_score), 4),
                            # calibrated gaze inputs
                            "yaw_deg": None if yaw_deg is None else round(float(yaw_deg), 2),
                            "pitch_deg": None if pitch_deg is None else round(float(pitch_deg), 2),
                            "roll_deg": None if roll_deg is None else round(float(roll_deg), 2),
                            "eye_left": round(float(eye_left), 4),
                            "eye_right": round(float(eye_right), 4),
                            "eye_up": round(float(eye_up), 4),
                            "eye_down": round(float(eye_down), 4),
                        }
                    else:
                        emotion = self._classify_emotion(scores)
                        data_packet = {
                            "timestamp": round(timestamp_sec, 2),
                            "emotion": emotion,
                            "gaze": gaze,
                            "blink_total": blink_count,
                        }

                    analysis_timeline.append(data_packet)
                    last_record_time = timestamp_sec

                    # Debug: pencerede göster
                    if show_video:
                        if phase3_enabled:
                            label = data_packet.get("facial_state", "NEUTRAL")
                            cv2.putText(
                                frame, f"State: {label}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            )
                        else:
                            cv2.putText(
                                frame, f"Duygu: {emotion}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            )
                        cv2.imshow("Video Analiz", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break

        cap.release()
        if show_video:
            cv2.destroyAllWindows()

        # ---------------------------------------------------------------
        # FAZ-3: Calibrated gaze post-process (default UNKNOWN)
        # ---------------------------------------------------------------
        gaze_distribution: Dict[str, int] = {}
        gaze_percentages: Dict[str, float] = {}
        reading_events: List[Dict] = []
        calib: Dict[str, float] = {}
        if phase3_enabled and analysis_timeline:
            calib = self._calibrate_baseline(analysis_timeline, effective_fps=effective_fps)
            for item in analysis_timeline:
                yaw = item.get("yaw_deg")
                pitch = item.get("pitch_deg")
                gaze2 = self._compute_gaze_direction_calibrated(
                    yaw=yaw,
                    pitch=pitch,
                    eye_down=float(item.get("eye_down", 0.0) or 0.0),
                    eye_left=float(item.get("eye_left", 0.0) or 0.0),
                    eye_right=float(item.get("eye_right", 0.0) or 0.0),
                    eye_up=float(item.get("eye_up", 0.0) or 0.0),
                    calib=calib,
                )
                item["gaze"] = gaze2
                item["attention_state"] = self._attention_state_from_gaze(gaze2)

            # distribution
            def _bucket(g: str) -> str:
                g = (g or "").lower()
                if "ekrana" in g:
                    return "ON_SCREEN"
                if "asagi" in g:
                    return "DOWN"
                if "saga" in g:
                    return "RIGHT"
                if "sola" in g:
                    return "LEFT"
                if "yukari" in g:
                    return "UP"
                return "UNKNOWN"

            for item in analysis_timeline:
                k = _bucket(item.get("gaze", ""))
                gaze_distribution[k] = gaze_distribution.get(k, 0) + 1

            total = sum(gaze_distribution.values()) or 1
            gaze_percentages = {k: round((v / total) * 100.0, 1) for k, v in gaze_distribution.items()}

            # reading events (DOWN >= threshold)
            reading_events = self._compute_reading_events(analysis_timeline)

        # --- Özet İstatistikler ---
        duration_min = (timestamp_ms / 1000) / 60 if timestamp_ms > 0 else 1
        bpm = blink_count / duration_min if duration_min > 0 else 0

        if analysis_timeline:
            focused_count = sum(1 for item in analysis_timeline if "Ekrana" in item.get("gaze", ""))
            # Not: default UNKNOWN olmadığı durumda şişme oluyordu. Artık gaze kalibre ediliyor.
            focus_score = (focused_count / len(analysis_timeline)) * 100

            if phase3_enabled:
                states = [item.get("facial_state", "") for item in analysis_timeline]
                dominant_emotion = Counter(states).most_common(1)[0][0] if states else "Veri Yok"
            else:
                emotions = [item["emotion"] for item in analysis_timeline]
                dominant_emotion = Counter(emotions).most_common(1)[0][0]
        else:
            focus_score = 0.0
            dominant_emotion = "Veri Yok"

        summary = {
            "total_blinks": blink_count,
            "blink_rate_per_min": round(bpm, 1),
            "duration_sec": round(frame_count / fps, 2),
            "focus_score": round(focus_score, 1),
            "dominant_emotion": dominant_emotion,
            "data_count": len(analysis_timeline),
            # gaze detail (v3)
            "gaze_distribution": gaze_distribution,
            "gaze_percentages": gaze_percentages,
            "gaze_calibration": calib,
            "reading_suspicion_events": reading_events,
        }

        print(f"Yüz analizi tamamlandı: {len(analysis_timeline)} kayıt.")
        return analysis_timeline, summary


if __name__ == "__main__":
    analyzer = FaceAnalyzer()
    print("FaceAnalyzer modülü hazır.")
