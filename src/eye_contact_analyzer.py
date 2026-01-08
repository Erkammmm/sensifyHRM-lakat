"""
Göz Teması Analizi Modülü
MediaPipe kullanarak göz teması ve bakış yönü analizi yapar.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
from collections import deque


class EyeContactAnalyzer:
    """
    Göz teması ve bakış yönü analizi yapan sınıf.
    MediaPipe Face Mesh kullanır.
    """
    
    # MediaPipe landmark indeksleri (gözler için)
    LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Yüz merkezi ve kamera yönü için landmark'lar
    NOSE_TIP = 4
    FOREHEAD_CENTER = 10
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Göz teması analizcisini başlatır.
        
        Args:
            min_detection_confidence: Minimum yüz tespit güveni
            min_tracking_confidence: Minimum takip güveni
        """
        # MediaPipe import - versiyon uyumluluğu için
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
        except AttributeError:
            # Yeni MediaPipe versiyonları için alternatif import
            import mediapipe.python.solutions.face_mesh as face_mesh_module
            import mediapipe.python.solutions.drawing_utils as drawing_utils_module
            self.mp_face_mesh = face_mesh_module
            self.mp_drawing = drawing_utils_module
        
        # Face Mesh modelini başlat
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Kamera parametreleri (dinamik olarak frame boyutuna göre ayarlanacak)
        # Varsayılan değerler, analyze_frame'de güncellenecek
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    def detect_face_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Frame'de yüz landmark'larını tespit eder.
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            Landmark bilgileri veya None
        """
        # MediaPipe RGB format bekliyor
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # İlk yüzü al (çoklu yüz desteklenmiyor şimdilik)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Landmark'ları numpy array'e çevir
        h, w = frame.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x * w, landmark.y * h, landmark.z * w])
        
        landmarks = np.array(landmarks)
        
        return {
            'landmarks': landmarks,
            'face_landmarks_mp': face_landmarks
        }
    
    def calculate_head_pose(self, landmarks: np.ndarray) -> Dict:
        """
        Baş pozisyonunu (head pose) hesaplar.
        
        Args:
            landmarks: Yüz landmark'ları (468x3)
            
        Returns:
            Baş pozisyonu bilgileri (pitch, yaw, roll)
        """
        # MediaPipe landmark indeksleri (doğru indeksler)
        # MediaPipe Face Mesh 468 landmark kullanır
        try:
            # Görüntü noktaları (landmark'lardan) - en az 4 nokta gerekli
            # MediaPipe landmark indeksleri: 0-467
            nose_tip_idx = 4  # Nose tip (doğru indeks)
            chin_idx = 175    # Chin
            left_eye_idx = 33  # Left eye left corner
            right_eye_idx = 263  # Right eye right corner
            left_mouth_idx = 61  # Left mouth corner
            right_mouth_idx = 291  # Right mouth corner
            
            # Landmark'ların geçerli olduğundan emin ol
            if landmarks.shape[0] < 468:
                return {'pitch': 0, 'yaw': 0, 'roll': 0, 'success': False}
            
            # Görüntü noktaları (2D koordinatlar - x, y)
            image_points = np.array([
                [landmarks[nose_tip_idx][0], landmarks[nose_tip_idx][1]],      # Nose tip
                [landmarks[chin_idx][0], landmarks[chin_idx][1]],              # Chin
                [landmarks[left_eye_idx][0], landmarks[left_eye_idx][1]],       # Left eye
                [landmarks[right_eye_idx][0], landmarks[right_eye_idx][1]],    # Right eye
                [landmarks[left_mouth_idx][0], landmarks[left_mouth_idx][1]],   # Left mouth
                [landmarks[right_mouth_idx][0], landmarks[right_mouth_idx][1]] # Right mouth
            ], dtype=np.float32)
            
            # 3D model noktaları (MediaPipe face mesh referans noktaları)
            # Basitleştirilmiş model noktaları (mm cinsinden)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ], dtype=np.float32)
            # SolvePnP ile baş pozisyonunu hesapla
            # En az 4 nokta gerekli, biz 6 nokta kullanıyoruz
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return {'pitch': 0, 'yaw': 0, 'roll': 0, 'success': False}
        except (IndexError, KeyError, cv2.error, Exception) as e:
            # Eğer hata olursa, varsayılan değerler döndür
            return {'pitch': 0, 'yaw': 0, 'roll': 0, 'success': False}
        
        # Rotation vector'ü Euler açılarına çevir
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Pitch, Yaw, Roll hesapla
        pitch = np.arcsin(-rotation_matrix[2][1]) * 180 / np.pi
        yaw = np.arctan2(rotation_matrix[2][0], rotation_matrix[2][2]) * 180 / np.pi
        roll = np.arctan2(rotation_matrix[0][1], rotation_matrix[1][1]) * 180 / np.pi
        
        return {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll),
            'success': True
        }
    
    def calculate_gaze_direction(self, landmarks: np.ndarray, head_pose: Dict) -> Dict:
        """
        Bakış yönünü hesaplar (head pose'u dikkate alarak).
        
        Args:
            landmarks: Yüz landmark'ları
            head_pose: Baş pozisyonu bilgileri (pitch, yaw, roll)
            
        Returns:
            Bakış yönü bilgileri
        """
        try:
            # Sol göz merkezi
            left_eye_landmarks = landmarks[self.LEFT_EYE_INDICES]
            left_eye_center = np.mean(left_eye_landmarks, axis=0)
            
            # Sağ göz merkezi
            right_eye_landmarks = landmarks[self.RIGHT_EYE_INDICES]
            right_eye_center = np.mean(right_eye_landmarks, axis=0)
            
            # Göz merkezi (iki gözün ortası) - 2D koordinatlar (x, y)
            eye_center_2d = (left_eye_center[:2] + right_eye_center[:2]) / 2
            
            # Kamera merkezi (frame merkezi)
            if self.camera_matrix is not None:
                camera_center_x = self.camera_matrix[0, 2]
                camera_center_y = self.camera_matrix[1, 2]
            else:
                camera_center_x = eye_center_2d[0]
                camera_center_y = eye_center_2d[1]
            
            # Head pose'u dikkate al (yaw ve pitch)
            yaw = head_pose.get('yaw', 0.0)  # Yatay açı
            pitch = head_pose.get('pitch', 0.0)  # Dikey açı
            
            # Göz merkezinden kamera merkezine vektör (2D)
            gaze_vector_2d = np.array([camera_center_x, camera_center_y]) - eye_center_2d
            
            # Frame boyutuna normalize et
            if self.camera_matrix is not None:
                frame_width = self.camera_matrix[0, 2] * 2
                frame_height = self.camera_matrix[1, 2] * 2
                frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
                
                # 2D mesafe (piksel cinsinden)
                distance_2d = np.linalg.norm(gaze_vector_2d)
                
                # Frame boyutuna göre normalize et (0-1 arası)
                # Frame'in yarısı kadar sapma = 1.0
                max_distance = np.sqrt((frame_width/2)**2 + (frame_height/2)**2)
                normalized_deviation_2d = min(1.0, distance_2d / max_distance) if max_distance > 0 else 1.0
                
                # Head pose'u dikkate alarak toplam sapma
                # Yaw ve pitch'i normalize et (0-1 arası)
                yaw_normalized = min(1.0, abs(yaw) / 45.0)  # 45 derece = 1.0
                pitch_normalized = min(1.0, abs(pitch) / 30.0)  # 30 derece = 1.0
                
                # Kombine sapma (2D + head pose)
                # 2D sapma daha az ağırlıklı, head pose daha önemli
                combined_deviation = min(1.0, (normalized_deviation_2d * 0.3 + yaw_normalized * 0.4 + pitch_normalized * 0.3))
                
                # Açı hesaplama (0-1 sapma = 0-25 derece)
                angle = combined_deviation * 25.0
            else:
                # Varsayılan: sadece head pose'a göre
                yaw_normalized = min(1.0, abs(yaw) / 45.0)
                pitch_normalized = min(1.0, abs(pitch) / 30.0)
                combined_deviation = min(1.0, (yaw_normalized * 0.5 + pitch_normalized * 0.5))
                angle = combined_deviation * 25.0
            
            # Göz teması skoru (0-1 arası, 0-25 derece arası iyi kabul edilir)
            # Daha esnek threshold (25 derece)
            eye_contact_score = max(0, 1.0 - (angle / 25.0))
            
            return {
                'gaze_vector': gaze_vector_2d.tolist(),
                'angle_degrees': float(angle),
                'eye_contact_score': float(eye_contact_score),
                'eye_center': eye_center_2d.tolist(),
                'head_pose_contribution': {
                    'yaw': float(yaw),
                    'pitch': float(pitch)
                }
            }
        except Exception as e:
            # Hata durumunda varsayılan değerler
            return {
                'gaze_vector': [0, 0],
                'angle_degrees': 30.0,
                'eye_contact_score': 0.0,
                'eye_center': [0, 0],
                'head_pose_contribution': {'yaw': 0.0, 'pitch': 0.0}
            }
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Tek bir frame'de göz teması analizi yapar.
        
        Args:
            frame: BGR formatında görüntü
            
        Returns:
            Analiz sonuçları veya None
        """
        # Kamera matrisini frame boyutuna göre ayarla
        h, w = frame.shape[:2]
        focal_length = w  # Focal length genişlik kadar (yaklaşık)
        center_x = w / 2.0
        center_y = h / 2.0
        
        self.camera_matrix = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        face_data = self.detect_face_landmarks(frame)
        
        if face_data is None:
            return None
        
        landmarks = face_data['landmarks']
        
        # Baş pozisyonu
        head_pose = self.calculate_head_pose(landmarks)
        
        # Bakış yönü (head pose'u dikkate alarak)
        gaze = self.calculate_gaze_direction(landmarks, head_pose)
        
        return {
            'head_pose': head_pose,
            'gaze': gaze,
            'has_face': True
        }
    
    def analyze_frames(self, frames: List[np.ndarray], sample_rate: int = 5) -> List[Optional[Dict]]:
        """
        Birden fazla frame'i analiz eder.
        
        Args:
            frames: Frame'lerin listesi
            sample_rate: Her N frame'de bir analiz yap (performans için, 1 = tüm frame'ler)
            
        Returns:
            Her frame için analiz sonuçları
        """
        results = []
        total_frames = len(frames)
        analyzed_count = 0
        success_count = 0
        
        # Her N frame'de bir analiz yap (performans için)
        for i, frame in enumerate(frames):
            if i % sample_rate == 0 or i == total_frames - 1:  # İlk, son ve her N. frame
                analyzed_count += 1
                result = self.analyze_frame(frame)
                if result is not None and result.get('has_face', False):
                    success_count += 1
                results.append(result)
            else:
                # Analiz edilmeyen frame'ler için None (interpolation yapma)
                results.append(None)
        
        # Debug bilgisi
        if analyzed_count > 0:
            success_rate = (success_count / analyzed_count) * 100
            print(f"[EyeContactAnalyzer] {analyzed_count} frame analiz edildi, {success_count} başarılı ({success_rate:.1f}%)")
        
        return results
    
    def calculate_eye_contact_metrics(self, analysis_results: List[Optional[Dict]]) -> Dict:
        """
        Göz teması metriklerini hesaplar.
        
        Args:
            analysis_results: Frame analiz sonuçları
            
        Returns:
            Göz teması metrikleri
        """
        # Geçerli sonuçları filtrele
        valid_results = [r for r in analysis_results if r is not None and r.get('has_face', False)]
        
        if not valid_results:
            return {
                'average_eye_contact_percentage': 0.0,
                'consistency_score': 0.0,
                'total_frames_analyzed': 0,
                'total_frames': len(analysis_results)
            }
        
        # Göz teması skorlarını topla
        eye_contact_scores = []
        gaze_angles = []
        
        for result in valid_results:
            gaze = result.get('gaze', {})
            eye_contact_score = gaze.get('eye_contact_score', 0.0)
            angle = gaze.get('angle_degrees', 90.0)
            
            eye_contact_scores.append(eye_contact_score)
            gaze_angles.append(angle)
        
        # Debug: İlk birkaç skoru göster
        if len(eye_contact_scores) > 0 and not hasattr(self, '_debug_shown'):
            print(f"[EyeContactAnalyzer] İlk 5 göz teması skoru: {eye_contact_scores[:5]}")
            print(f"[EyeContactAnalyzer] İlk 5 gaze açısı: {gaze_angles[:5]}")
            self._debug_shown = True
        
        # Ortalama göz teması yüzdesi
        if len(eye_contact_scores) > 0:
            avg_eye_contact = np.mean(eye_contact_scores) * 100
        else:
            avg_eye_contact = 0.0
        
        # Tutarlılık skoru (düşük standart sapma = yüksek tutarlılık)
        if len(eye_contact_scores) > 1:
            std_dev = np.std(eye_contact_scores)
            consistency_score = max(0, 1.0 - (std_dev / 0.5))  # 0.5 std dev threshold
        else:
            consistency_score = 1.0
        
        # Bakış yönü dağılımı
        gaze_patterns = {
            'direct': sum(1 for angle in gaze_angles if angle < 15),
            'slight_deviation': sum(1 for angle in gaze_angles if 15 <= angle < 30),
            'moderate_deviation': sum(1 for angle in gaze_angles if 30 <= angle < 45),
            'significant_deviation': sum(1 for angle in gaze_angles if angle >= 45)
        }
        
        return {
            'average_eye_contact_percentage': float(avg_eye_contact),
            'consistency_score': float(consistency_score),
            'gaze_patterns': gaze_patterns,
            'average_gaze_angle': float(np.mean(gaze_angles)),
            'total_frames_analyzed': len(valid_results),
            'total_frames': len(analysis_results),
            'coverage': len(valid_results) / len(analysis_results) if analysis_results else 0.0
        }


if __name__ == "__main__":
    # Test kodu
    analyzer = EyeContactAnalyzer()
    print("EyeContactAnalyzer modülü hazır.")
