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
        
        # Eye contact threshold (derece cinsinden)
        # Yaw ve pitch açıları bu değerden küçükse göz teması var sayılır
        self.eye_contact_threshold = 15.0  # derece
    
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
    
    def calculate_eye_contact_score(self, yaw: float, pitch: float, gaze_class: Optional[str] = None) -> float:
        """
        Yaw ve pitch açılarından veya gaze_class'tan göz teması skorunu hesaplar.
        Webcam modunda sadece gaze_class kullanılır (açı bazlı hesaplama devre dışı).
        
        Args:
            yaw: Yatay bakış açısı (derece) - webcam modunda kullanılmaz
            pitch: Dikey bakış açısı (derece) - webcam modunda kullanılmaz
            gaze_class: Gaze sınıfı (camera/left/right/up/down) - webcam modunda kullanılır
            
        Returns:
            Eye contact score (0.0 - 1.0)
        """
        # Webcam modu: Sadece gaze_class kullan (açı bazlı hesaplama devre dışı)
        if gaze_class is not None:
            return 1.0 if gaze_class == 'camera' else 0.0
        
        # Eski mod: Açı bazlı hesaplama (video analizi için)
        # Toplam gaze açısı
        gaze_angle = np.sqrt(yaw**2 + pitch**2)
        
        # Eğer gaze açısı threshold'dan küçükse, göz teması var
        if gaze_angle <= self.eye_contact_threshold:
            # Açı ne kadar küçükse, skor o kadar yüksek
            score = 1.0 - (gaze_angle / self.eye_contact_threshold)
        else:
            # Threshold'dan büyükse, skor düşer
            score = max(0.0, 1.0 - (gaze_angle - self.eye_contact_threshold) / 30.0)
        
        return max(0.0, min(1.0, score))  # 0-1 aralığına sınırla
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Tek bir frame'de göz teması analizi yapar.
        
        Args:
            frame: BGR formatında görüntü
            
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
        
        yaw = gaze_result['yaw']
        pitch = gaze_result['pitch']
        gaze_angle = gaze_result['gaze_angle']
        gaze_class = gaze_result.get('gaze_class', 'camera')
        
        # Eye contact score hesapla (gaze_class'a göre - webcam modu)
        # Camera sınıfı = göz teması var, açı bazlı hesaplama devre dışı
        eye_contact_score = self.calculate_eye_contact_score(yaw, pitch, gaze_class)
        
        return {
            'gaze': {
                'yaw': yaw,
                'pitch': pitch,
                'gaze_angle': gaze_angle,
                'gaze_class': gaze_class,
                'eye_contact_score': eye_contact_score
            },
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
                result = self.analyze_frame(frame)
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
        Göz teması metriklerini hesaplar.
        
        Args:
            eye_contact_results: Frame bazlı göz teması sonuçları
            
        Returns:
            Özet metrikler
        """
        valid_results = [r for r in eye_contact_results if r is not None and r.get('has_face', False)]
        
        if len(valid_results) == 0:
            total_frames = len(eye_contact_results)
            return {
                'average_eye_contact': 0.0,
                'average_eye_contact_percentage': 0.0,
                'eye_contact_percentage': 0.0,
                'consistency_score': 0.0,
                'average_gaze_angle': 0.0,
                'coverage': 0.0,
                'gaze_patterns': {},
                'total_frames': total_frames,
                'frames_with_face': 0
            }
        
        # Eye contact skorları ve gaze sınıfları
        eye_contact_scores = []
        gaze_angles = []
        gaze_classes = []
        
        for result in valid_results:
            gaze = result.get('gaze', {})
            if gaze:
                eye_contact_scores.append(gaze.get('eye_contact_score', 0.0))
                gaze_angles.append(gaze.get('gaze_angle', 0.0))
                gaze_classes.append(gaze.get('gaze_class', 'unknown'))
        
        if len(eye_contact_scores) == 0:
            total_frames = len(eye_contact_results)
            coverage = len(valid_results) / total_frames if total_frames > 0 else 0.0
            return {
                'average_eye_contact': 0.0,
                'average_eye_contact_percentage': 0.0,
                'eye_contact_percentage': 0.0,
                'consistency_score': 0.0,
                'average_gaze_angle': 0.0,
                'coverage': float(coverage),
                'gaze_patterns': {},
                'total_frames': total_frames,
                'frames_with_face': len(valid_results)
            }
        
        # Ortalama göz teması skoru
        avg_eye_contact = np.mean(eye_contact_scores)
        
        # Göz teması yüzdesi: "camera" sınıfında geçirilen frame oranı (0-100 arası)
        # Webcam analizinde sadece gerçek kamera bakışı eye contact olarak kabul edilir
        camera_frames = sum(1 for gc in gaze_classes if gc == 'camera')
        eye_contact_percentage = (camera_frames / len(gaze_classes)) * 100.0 if gaze_classes else 0.0
        
        # Tutarlılık skoru (standart sapmanın tersi)
        if len(eye_contact_scores) > 1:
            std_dev = np.std(eye_contact_scores)
            consistency_score = max(0.0, 1.0 - std_dev)  # Düşük std = yüksek tutarlılık
        else:
            consistency_score = 1.0
        
        # Ortalama gaze açısı
        avg_gaze_angle = np.mean(gaze_angles) if gaze_angles else 0.0
        
        # Coverage: yüz tespit edilen frame'lerin toplam frame'lere oranı
        total_frames = len(eye_contact_results)
        coverage = len(valid_results) / total_frames if total_frames > 0 else 0.0
        
        # Gaze patterns (gaze sınıflarına göre)
        gaze_patterns = {}
        if gaze_classes:
            gaze_class_counts = {
                'camera': sum(1 for gc in gaze_classes if gc == 'camera'),
                'left': sum(1 for gc in gaze_classes if gc == 'left'),
                'right': sum(1 for gc in gaze_classes if gc == 'right'),
                'up': sum(1 for gc in gaze_classes if gc == 'up'),
                'down': sum(1 for gc in gaze_classes if gc == 'down')
            }
            
            total = len(gaze_classes)
            gaze_patterns = {
                'camera_ratio': gaze_class_counts['camera'] / total if total > 0 else 0.0,
                'left_ratio': gaze_class_counts['left'] / total if total > 0 else 0.0,
                'right_ratio': gaze_class_counts['right'] / total if total > 0 else 0.0,
                'up_ratio': gaze_class_counts['up'] / total if total > 0 else 0.0,
                'down_ratio': gaze_class_counts['down'] / total if total > 0 else 0.0
            }
        
        return {
            'average_eye_contact': float(avg_eye_contact),
            'average_eye_contact_percentage': float(eye_contact_percentage),  # Pipeline'ın beklediği isim
            'eye_contact_percentage': float(eye_contact_percentage),  # Geriye uyumluluk için
            'consistency_score': float(consistency_score),
            'average_gaze_angle': float(avg_gaze_angle),
            'coverage': float(coverage),  # Pipeline'ın beklediği coverage
            'gaze_patterns': gaze_patterns,  # Pipeline'ın beklediği gaze_patterns
            'total_frames': total_frames,
            'frames_with_face': len(valid_results)
        }
