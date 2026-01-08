"""
Duygu Analizi Modülü
DeepFace kullanarak yüz ifadelerinden duygu tespiti yapar.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
import cv2
from collections import Counter
from PIL import Image
import tempfile
import os


class EmotionAnalyzer:
    """
    Yüz ifadelerinden duygu analizi yapan sınıf.
    DeepFace pretrained modelini kullanır.
    """
    
    # Türkçe duygu isimleri mapping
    EMOTION_TRANSLATION = {
        'angry': 'kızgınlık',
        'disgust': 'iğrenme',
        'fear': 'korku',
        'happy': 'mutluluk',
        'sad': 'üzüntü',
        'surprise': 'şaşkınlık',
        'neutral': 'nötr'
    }
    
    def __init__(self, model_name: str = "VGG-Face", enforce_detection: bool = False):
        """
        Duygu analizcisini başlatır.
        
        Args:
            model_name: DeepFace model adı (VGG-Face, OpenFace, FaceNet, etc.)
            enforce_detection: Yüz bulunamazsa hata ver (False ise None döner)
        """
        self.model_name = model_name
        self.enforce_detection = enforce_detection
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Tek bir frame'deki yüz özelliklerini analiz eder (age, gender, emotion, race).
        
        Args:
            frame: BGR formatında görüntü (OpenCV formatı)
            
        Returns:
            Yüz özelliklerini içeren dictionary (emotion, age, gender, race) veya None (yüz bulunamazsa)
        """
        try:
            # DeepFace numpy array'i direkt kabul etmiyor, geçici dosya kullan
            # BGR'den RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Geçici dosya oluştur (DeepFace daha güvenilir çalışır)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                # PIL Image'e çevir ve kaydet
                pil_image = Image.fromarray(rgb_frame)
                pil_image.save(tmp_path, 'JPEG', quality=95)
            
            try:
                # DeepFace analizi - tüm özellikleri al (age, gender, emotion, race)
                result = DeepFace.analyze(
                    img_path=tmp_path,
                    actions=['age', 'gender', 'emotion', 'race'],
                    enforce_detection=self.enforce_detection,
                    silent=False  # Hata mesajlarını görmek için
                )
                
                # Sonuç formatını düzenle
                if isinstance(result, list):
                    result = result[0]
                
                # Tüm özellikleri döndür
                return {
                    'emotion': result.get('emotion', {}),
                    'age': result.get('age', 0),
                    'gender': result.get('dominant_gender', 'unknown'),
                    'race': result.get('dominant_race', 'unknown'),
                    'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0.0) if isinstance(result.get('gender'), dict) else 0.0,
                    'race_confidence': result.get('race', {}).get(result.get('dominant_race', ''), 0.0) if isinstance(result.get('race'), dict) else 0.0
                }
            finally:
                # Geçici dosyayı sil
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            # Hata mesajını göster (debug için)
            # Sadece ilk birkaç hatada göster (çok fazla log olmasın)
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            
            if self._error_count < 3:
                print(f"[EmotionAnalyzer] Hata (frame analizi): {str(e)}")
                self._error_count += 1
            
            # Yüz bulunamadı veya başka bir hata
            if not self.enforce_detection:
                return None
            return None
    
    def analyze_frames(self, frames: List[np.ndarray], sample_rate: int = 5) -> List[Optional[Dict[str, float]]]:
        """
        Birden fazla frame'i analiz eder.
        
        Args:
            frames: Frame'lerin listesi
            sample_rate: Her N frame'de bir analiz yap (performans için, 1 = tüm frame'ler)
            
        Returns:
            Her frame için duygu skorları listesi
        """
        results = []
        total_frames = len(frames)
        analyzed_count = 0
        success_count = 0
        
        # Her N frame'de bir analiz yap (performans için)
        for i, frame in enumerate(frames):
            if i % sample_rate == 0 or i == total_frames - 1:  # İlk, son ve her N. frame
                analyzed_count += 1
                emotion_result = self.analyze_frame(frame)
                if emotion_result is not None:
                    success_count += 1
                results.append(emotion_result)
            else:
                # Analiz edilmeyen frame'ler için None (interpolation yapma, yanlış sonuçlara yol açabilir)
                results.append(None)
        
        # Debug bilgisi
        if analyzed_count > 0:
            success_rate = (success_count / analyzed_count) * 100
            print(f"[EmotionAnalyzer] {analyzed_count} frame analiz edildi, {success_count} başarılı ({success_rate:.1f}%)")
        
        return results
    
    def calculate_emotion_trends(self, emotion_results: List[Optional[Dict[str, float]]]) -> Dict:
        """
        Duygu sonuçlarından trend analizi yapar.
        
        Args:
            emotion_results: Frame'ler için duygu skorları listesi
            
        Returns:
            Trend analizi sonuçları
        """
        # Geçerli sonuçları filtrele (None olmayanlar)
        valid_results = [r for r in emotion_results if r is not None]
        
        if not valid_results:
            return {
                'dominant_emotion': None,
                'emotion_distribution': {},
                'stability_score': 0.0,
                'emotion_timeline': [],
                'total_frames_analyzed': 0,
                'total_frames': len(emotion_results)
            }
        
        # Her frame için dominant duyguyu bul
        dominant_emotions = []
        for result in valid_results:
            if result and 'emotion' in result:
                emotion_scores = result['emotion']
                if emotion_scores:
                    dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    dominant_emotions.append(dominant)
        
        # En sık görülen duygu
        emotion_counter = Counter(dominant_emotions)
        most_common_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else None
        
        # Duygu dağılımını hesapla (ortalama skorlar)
        emotion_distribution = {}
        for emotion in self.EMOTION_TRANSLATION.keys():
            scores = []
            for r in valid_results:
                if r and 'emotion' in r:
                    emotion_scores = r['emotion']
                    if emotion_scores and emotion in emotion_scores:
                        scores.append(emotion_scores[emotion])
            if scores:
                emotion_distribution[emotion] = {
                    'average': np.mean(scores),
                    'max': np.max(scores),
                    'min': np.min(scores),
                    'std': np.std(scores)
                }
        
        # Stabilite skoru hesapla (duygu değişkenliği)
        # Düşük değişkenlik = yüksek stabilite
        if len(dominant_emotions) > 1:
            # Dominant duygu değişim sıklığı
            changes = sum(1 for i in range(1, len(dominant_emotions)) 
                         if dominant_emotions[i] != dominant_emotions[i-1])
            stability_score = 1.0 - (changes / len(dominant_emotions))
        else:
            stability_score = 1.0
        
        # Timeline oluştur (her frame için dominant duygu)
        emotion_timeline = []
        for i, result in enumerate(emotion_results):
            if result and 'emotion' in result:
                emotion_scores = result['emotion']
                if emotion_scores:
                    dominant = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    emotion_timeline.append({
                        'frame': i,
                        'dominant_emotion': dominant,
                        'emotion_scores': emotion_scores,
                        'age': result.get('age', 0),
                        'gender': result.get('gender', 'unknown'),
                        'race': result.get('race', 'unknown')
                    })
        
        return {
            'dominant_emotion': most_common_emotion,
            'emotion_distribution': emotion_distribution,
            'stability_score': float(stability_score),
            'emotion_timeline': emotion_timeline,
            'total_frames_analyzed': len(valid_results),
            'total_frames': len(emotion_results)
        }
    
    def get_emotion_summary(self, emotion_results: List[Optional[Dict[str, float]]]) -> Dict:
        """
        Duygu analizi özeti oluşturur.
        
        Args:
            emotion_results: Frame'ler için duygu skorları listesi
            
        Returns:
            Özet rapor
        """
        trends = self.calculate_emotion_trends(emotion_results)
        
        # Güvenli erişim için
        total_frames = trends.get('total_frames', len(emotion_results))
        total_frames_analyzed = trends.get('total_frames_analyzed', 0)
        
        # Türkçe çevirileri ekle
        summary = {
            'dominant_emotion': trends.get('dominant_emotion'),
            'dominant_emotion_tr': self.EMOTION_TRANSLATION.get(trends.get('dominant_emotion', 'neutral'), 'bilinmiyor'),
            'emotion_distribution': {
                self.EMOTION_TRANSLATION.get(k, k): v 
                for k, v in trends.get('emotion_distribution', {}).items()
            },
            'stability_score': trends.get('stability_score', 0.0),
            'coverage': total_frames_analyzed / total_frames if total_frames > 0 else 0.0
        }
        
        return summary


if __name__ == "__main__":
    # Test kodu
    analyzer = EmotionAnalyzer()
    print("EmotionAnalyzer modülü hazır.")
