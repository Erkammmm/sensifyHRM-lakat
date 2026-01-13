"""
Mediapipe tabanlı canlı kamera testi.

- Yüz + ham özellikler (mouth_width, mouth_height, eye_opening, brow_distance, jaw_open)
- Gaze (x, y, direction: center/left/right/up/down)

Kullanım:
    (sensifyhr) python test_webcam_mediapipe.py
    (sensifyhr) python test_webcam_mediapipe.py 1   # farklı kamera index'i için
"""

import sys
import os
from typing import Optional

import cv2

# Proje root'unu path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mediapipe_face_gaze_analyzer import MediapipeFaceGazeAnalyzer


def draw_overlay(frame, analysis: Optional[dict]) -> None:
    """Frame üzerine basit overlay ve bounding box çizer."""
    h, w = frame.shape[:2]
    if analysis is None:
        cv2.putText(
            frame,
            "No face",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    raw_features = analysis.get("raw_features", {})
    gaze = analysis.get("gaze", {})
    bbox = analysis.get("bbox", None)
    landmarks = analysis.get("landmarks", [])

    # Yüz bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Landmark noktaları (yüz + gözler)
    for (lx, ly) in landmarks:
        cv2.circle(frame, (lx, ly), 1, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # Gaze yönü
    gaze_dir = gaze.get("direction", "center")
    gx = gaze.get("x", 0.0)
    gy = gaze.get("y", 0.0)
    cv2.putText(
        frame,
        f"Gaze: {gaze_dir} (x={gx:.2f}, y={gy:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Ham özellikler (anlık, ayrı satırlarda)
    lines = [
        f"Mouth Width:  {raw_features.get('mouth_width_norm', 0.0):.3f}",
        f"Mouth Height: {raw_features.get('mouth_height_norm', 0.0):.3f}",
        f"Eye Opening:  {raw_features.get('eye_opening_norm', 0.0):.3f}",
        f"Brow Distance: {raw_features.get('brow_distance_norm', 0.0):.3f}",
        f"Jaw Open:     {raw_features.get('jaw_open_norm', 0.0):.3f}",
    ]
    y0 = 60
    for i, text in enumerate(lines):
        y = y0 + i * 20
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main():
    cam_index = 0
    if len(sys.argv) > 1:
        try:
            cam_index = int(sys.argv[1])
        except ValueError:
            pass

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Kamera acilamadi (index={cam_index})")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    analyzer = MediapipeFaceGazeAnalyzer(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        frame_skip=1,  # canlıda her kareyi işliyoruz
        use_kalman_gaze=True,
    )

    frame_idx = 0
    print("Canli kamera basladi. Cikmak icin 'q' tusuna bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp = frame_idx / fps

        analysis = analyzer.process_frame(frame, timestamp)
        draw_overlay(frame, analysis)

        cv2.imshow("Mediapipe Live (ESC/q: cikis)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

