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

# --- Ayarlar ---
RAPOR_SIKLIGI = 0.5  # Saniyede bir veri kaydet
TARGET_FPS = float(os.getenv("SENSIFYHR_FACE_TARGET_FPS", "5.0"))


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
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print(f"[{self.__class__.__name__}] Hazır!")

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
                    gaze = self._detect_gaze(scores)

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

        # --- Özet İstatistikler ---
        duration_min = (timestamp_ms / 1000) / 60 if timestamp_ms > 0 else 1
        bpm = blink_count / duration_min if duration_min > 0 else 0

        if analysis_timeline:
            focused_count = sum(1 for item in analysis_timeline if "Ekrana" in item.get("gaze", ""))
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
        }

        print(f"Yüz analizi tamamlandı: {len(analysis_timeline)} kayıt.")
        return analysis_timeline, summary


if __name__ == "__main__":
    analyzer = FaceAnalyzer()
    print("FaceAnalyzer modülü hazır.")
