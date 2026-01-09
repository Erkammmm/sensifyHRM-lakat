"""
Göz Teması Analizi Modülü
Gaze Estimation Model kullanarak göz teması ve bakış yönü analizi yapar.
MediaPipe Face Detection kullanarak yüz tespiti yapar.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
from collections import deque

try:
    from src.gaze_estimation_model import GazeEstimationModel
except ImportError:
    try:
        from gaze_estimation_model import GazeEstimationModel
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from gaze_estimation_model import GazeEstimationModel


class EyeContactAnalyzer:
    """
    Göz teması ve bakış yönü analizi yapan sınıf.
    Gaze Estimation Model (pretrained) kullanır.
    MediaPipe Face Detection ile yüz tespiti yapar.
    
    Sadece MobileGaze modelinin ham çıktılarını kullanır.
    Hiçbir ek yorum, açı, landmark, consistency veya eye-contact skoru hesaplamaz.
    """
    
    def __init__(self, 
                 model_name: str = "mobilenetv2",
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 device: str = "cpu"):
        """
        Göz teması analizcisini başlatır.
        
        Args:
            model_name: Gaze estimation model adı (resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0)
            min_detection_confidence: Minimum yüz tespit güveni (MediaPipe Face Detection için)
            min_tracking_confidence: Minimum takip güveni (MediaPipe Face Detection için)
            device: Cihaz (cpu veya cuda)
        """
        # Gaze estimation modelini başlat
        print(f"[EyeContactAnalyzer] Gaze estimation modeli yükleniyor: {model_name}")
        self.gaze_model = GazeEstimationModel(model_name=model_name, device=device)
        print(f"[EyeContactAnalyzer] Gaze estimation modeli yüklendi.")
        
        # MediaPipe Face Detection (yüz crop için)
        try:
            self.mp_face_detection = mp.solutions.face_detection
        except AttributeError:
            import mediapipe.python.solutions.face_detection as face_detection_module
            self.mp_face_detection = face_detection_module
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = kısa mesafe, 1 = uzun mesafe
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_face(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Frame'de yüz tespiti yapar ve yüz bölgesini crop eder.
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            Yüz bölgesi (crop edilmiş) ve bounding box bilgileri veya None
        """
        # MediaPipe RGB format bekliyor
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return None
        
        # İlk yüzü al (çoklu yüz desteklenmiyor şimdilik)
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = frame.shape[:2]
        
        # Bounding box koordinatlarını hesapla
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        x_max = int((bbox.xmin + bbox.width) * w)
        y_max = int((bbox.ymin + bbox.height) * h)
        
        # Sınırları kontrol et
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Yüz bölgesini crop et (biraz padding ekle)
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        face_crop = frame[y_min:y_max, x_min:x_max]
        
        if face_crop.size == 0:
            return None
        
        return {
            'face_crop': face_crop,
            'bbox': (x_min, y_min, x_max, y_max),
            'confidence': detection.score[0] if detection.score else 0.0
        }
    
    def analyze_frame(self, frame: np.ndarray, frame_index: int = 0) -> Optional[Dict]:
        """
        Tek bir frame'de göz teması analizi yapar.
        
        Args:
            frame: BGR formatında görüntü
            frame_index: Frame indeksi (raw_gaze_predictions için)
            
        Returns:
            Analiz sonuçları veya None
        """
        # Yüz tespiti ve crop
        face_data = self.detect_face(frame)
        
        if face_data is None:
            return None
        
        face_crop = face_data['face_crop']
        bbox = face_data['bbox']
        
        # Gaze estimation
        gaze_result = self.gaze_model.estimate_gaze(face_crop)
        
        if gaze_result is None:
            return None
        
        # Modelin ham çıktılarını al
        gaze_class = gaze_result.get('gaze_class', 'unknown')
        confidence = gaze_result.get('confidence', 1.0)
        
        return {
            'frame_index': frame_index,
            'gaze_class': gaze_class,
            'confidence': confidence,
            'bbox': bbox,
            'has_face': True
        }
    
    def analyze_frames(self, frames: List[np.ndarray], sample_rate: int = 5) -> List[Optional[Dict]]:
        """
        Birden fazla frame'i analiz eder.
        
        Args:
            frames: Frame'lerin listesi
            sample_rate: Her N frame'de bir analiz yap (performans için, 1 = tüm frame'ler)
            
        Returns:
            Her frame için analiz sonuçları (None = yüz bulunamadı veya hata)
        """
        results = []
        total_frames = len(frames)
        successful = 0
        
        for i, frame in enumerate(frames):
            # Sample rate kontrolü
            if i % sample_rate != 0:
                results.append(None)
                continue
            
            try:
                result = self.analyze_frame(frame, frame_index=i)
                results.append(result)
                if result is not None:
                    successful += 1
            except Exception as e:
                # Hata durumunda None ekle
                if not hasattr(self, '_error_count'):
                    self._error_count = 0
                if self._error_count < 3:
                    print(f"[EyeContactAnalyzer] Hata (frame {i}): {str(e)}")
                    self._error_count += 1
                results.append(None)
        
        print(f"[EyeContactAnalyzer] {total_frames} frame analiz edildi, {successful} başarılı ({100.0*successful/max(1, len([r for r in results if r is not None])):.1f}%)")
        
        return results
    
    def calculate_eye_contact_metrics(self, eye_contact_results: List[Optional[Dict]]) -> Dict:
        """
        Göz teması metriklerini hesaplar (sadeleştirilmiş versiyon).

        Not:
        - Sadece MobileGaze modelinin verdiği sınıflar kullanılır.
        - Herhangi bir açı (angle), eye contact skoru, consistency veya coverage hesabı yapılmaz.
        - Her frame için raw_gaze_predictions listesi oluşturulur.

        Args:
            eye_contact_results: Frame bazlı göz teması sonuçları

        Returns:
            {
                "raw_gaze_predictions": [...],
                "gaze_counts": { ... },
                "gaze_ratios": { ... }
            }
        """
        # Geçerli sonuçları filtrele
        valid_results = [r for r in eye_contact_results if r is not None and r.get('has_face', False)]

        # Eğer hiç yüz bulunamadıysa
        if len(valid_results) == 0:
            return {
                "raw_gaze_predictions": [],
                "gaze_counts": {},
                "gaze_ratios": {}
            }

        # Raw gaze predictions listesi oluştur
        raw_gaze_predictions = []
        gaze_classes = []
        
        for result in valid_results:
            frame_index = result.get('frame_index', 0)
            gaze_class = result.get('gaze_class', 'unknown')
            confidence = result.get('confidence', 1.0)
            
            # Raw prediction kaydı
            raw_gaze_predictions.append({
                'frame_index': int(frame_index),
                'gaze_class': gaze_class,
                'confidence': float(confidence)
            })
            
            gaze_classes.append(gaze_class)

        if not gaze_classes:
            return {
                "raw_gaze_predictions": [],
                "gaze_counts": {},
                "gaze_ratios": {}
            }

        # Tüm sınıfları say
        gaze_counts: Dict[str, int] = {}
        for gc in gaze_classes:
            gaze_counts[gc] = gaze_counts.get(gc, 0) + 1

        total = len(gaze_classes)

        # Oranları hesapla
        gaze_ratios: Dict[str, float] = {}
        for gc, count in gaze_counts.items():
            gaze_ratios[gc] = float(count) / float(total) if total > 0 else 0.0

        return {
            "raw_gaze_predictions": raw_gaze_predictions,
            "gaze_counts": gaze_counts,
            "gaze_ratios": gaze_ratios
        }
