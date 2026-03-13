from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import cv2
import time
import numpy as np

# UniFace Modülleri
from uniface.detection import RetinaFace
from uniface.constants import RetinaFaceWeights
from uniface.gaze import MobileGaze
from uniface.constants import GazeWeights
from uniface.attribute import Emotion         # Duygu modülü
from uniface.constants import DDAMFNWeights   # Duygu ağırlıkları

# --- Yardımcı Fonksiyon ---
def get_source_type(source: str) -> str:
    ext = Path(source).suffix.lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return 'video'
    return 'unknown'
# ----------------------------------------------------------------------------------

def process_video(detector, gaze_estimator, emotion_predictor, video_path: str, save_dir: str = 'outputs'):
    """Process a video file (JSON Output Only - Gaze & Emotion)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return

    # Video özellikleri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Kayıt klasörü ve JSON dosya yolu
    os.makedirs(save_dir, exist_ok=True)
    output_json_path = os.path.join(save_dir, f'{Path(video_path).stem}_gaze_emotion_fast.json')
    
    # --- MÜLAKAT OPTİMİZASYONU: Saniyede 3 Kare (333 ms) ---
    PROCESS_EVERY_N = 10  
    
    print(f'Processing video: {video_path} ({total_frames} frames)')
    print(f'DIKKAT: Sadece JSON verisi çıkartılacak. (Atlama Değeri: {PROCESS_EVERY_N})')
    
    frame_count = 0       
    processed_count = 0   
    
    analysis_results = []
    
    start_time = time.time()        
    batch_start_time = time.time()  
    batch_frame_count = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sayaçları artır
        frame_count += 1
        processed_count += 1
        batch_frame_count += 1

        # ---------------- 1. AĞIR YAPAY ZEKA BLOĞU ----------------
        faces = detector.detect(frame)
        
        # Bu kare için örnekteki JSON formatına uygun taslak
        frame_data_json = {
            "frame_index": frame_count,
            "faces": []
        }
        
        for face in faces:
            bbox = face.bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # --- Duygu Analizi ---
            emo_result = emotion_predictor.predict(frame, face.landmarks)
            
            # --- Güvenli Gaze Analizi Kırpması ---
            crop_y1, crop_y2 = max(0, y1), min(height, y2)
            crop_x1, crop_x2 = max(0, x1), min(width, x2)
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            pitch_deg, yaw_deg = 0.0, 0.0

            if face_crop.size > 0:
                gaze_result = gaze_estimator.estimate(face_crop)
                # Radyanı dereceye çevirip virgülden sonra 2 basamağa yuvarlıyoruz
                pitch_deg = round(float(np.degrees(gaze_result.pitch)), 2)
                yaw_deg = round(float(np.degrees(gaze_result.yaw)), 2)

            # Yüz verisini (Gaze + Emotion) örnek JSON formatıyla aynı hiyerarşide listeye ekle
            frame_data_json["faces"].append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(face.confidence), 3),
                "emotion": {
                    "label": emo_result.emotion.capitalize(),
                    "confidence": round(float(emo_result.confidence), 3)
                },
                "gaze": {
                    "pitch_deg": pitch_deg,
                    "yaw_deg": yaw_deg
                }
            })
        
        # O karedeki analiz paketini ana listeye ekle
        analysis_results.append(frame_data_json)

        # ---------------- 2. GEREKSİZ KARELERİ FİZİKSEL OLARAK ATLA (KÖR TARAMA) ----------------
        for _ in range(PROCESS_EVERY_N - 1):
            if cap.grab():
                frame_count += 1
                batch_frame_count += 1
            else:
                break

        # --- ANLIK HIZ (TARAMA HIZI) HESAPLAMA ---
        if processed_count % 10 == 0:
            current_time = time.time()
            batch_duration = current_time - batch_start_time
            
            fps_hizi = batch_frame_count / batch_duration if batch_duration > 0 else 0
            print(f'  Video İlerlemesi: {frame_count}/{total_frames} kare... (Gerçek Tarama Hızı: {fps_hizi:.1f} fps)')
            
            batch_start_time = current_time
            batch_frame_count = 0

    cap.release()
    
    # ---------------- 3. JSON DOSYASINI KAYDET ----------------
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4)
        
    toplam_sure = time.time() - start_time
    print(f'\n[BAŞARILI] Analiz Tamamlandı!')
    print(f'JSON Verisi Kaydedildi: {output_json_path}')
    print(f'Toplam Süre: {toplam_sure:.1f} saniye')


def main():
    parser = argparse.ArgumentParser(description='Run video gaze and emotion estimation (JSON Output Only)')
    parser.add_argument('--source', type=str, required=True, help='Video path (.mp4, .avi, vb.)')
    parser.add_argument('--save-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    # Modelleri Başlat
    print("Modeller yükleniyor...")
    detector = RetinaFace(model_name=RetinaFaceWeights.MNET_025)
    gaze_estimator = MobileGaze(model_name=GazeWeights.RESNET18)
    emotion_predictor = Emotion(model_name=DDAMFNWeights.AFFECNET7)

    source_type = get_source_type(args.source)

    if source_type == 'video':
        if not os.path.exists(args.source):
            print(f'Error: Video not found: {args.source}')
            return
        process_video(detector, gaze_estimator, emotion_predictor, args.source, args.save_dir)
    else:
        print(f"Error: Lütfen geçerli bir video dosyası yolu girin. '{args.source}' desteklenmiyor.")
        print('Desteklenen formatlar: videolar (.mp4, .avi, .mkv, .mov)')

if __name__ == '__main__':
    main()