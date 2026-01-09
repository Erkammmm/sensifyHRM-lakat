"""
Ana Pipeline Modülü
Tüm analiz modüllerini koordine eder ve sonuçları birleştirir.
"""

import numpy as np
from typing import Dict, List, Optional
import uuid
from datetime import datetime
from collections import Counter

from .video_processor import VideoProcessor
from .emotion_analyzer import EmotionAnalyzer
from .eye_contact_analyzer import EyeContactAnalyzer
from .voice_analyzer import VoiceAnalyzer
from .behavioral_analyzer import BehavioralAnalyzer


class InterviewAnalysisPipeline:
    """
    Mülakat analiz pipeline'ı.
    Video'yu işler, tüm analizleri yapar ve rapor oluşturur.
    """
    
    def __init__(self):
        """Pipeline'ı başlatır ve modülleri yükler."""
        self.video_processor = VideoProcessor()
        self.emotion_analyzer = EmotionAnalyzer()
        self.eye_contact_analyzer = EyeContactAnalyzer()
        self.voice_analyzer = VoiceAnalyzer()
        self.behavioral_analyzer = BehavioralAnalyzer()
    
    def process_interview(self, video_path: str, interview_id: Optional[str] = None) -> Dict:
        """
        Mülakat videosunu işler ve analiz eder.
        
        Args:
            video_path: Video dosyasının yolu
            interview_id: Mülakat ID (None ise otomatik oluşturulur)
            
        Returns:
            Analiz sonuçları dictionary'si
        """
        if interview_id is None:
            interview_id = str(uuid.uuid4())
        
        print(f"[Pipeline] Mülakat analizi başlatılıyor: {interview_id}")
        
        # 1. Video bilgilerini al
        print("[Pipeline] Video bilgileri alınıyor...")
        video_info = self.video_processor.get_video_info(video_path)
        
        # 2. Frame'leri çıkar
        print("[Pipeline] Frame'ler çıkarılıyor...")
        frames = self.video_processor.extract_frames(video_path)
        print(f"[Pipeline] {len(frames)} frame çıkarıldı.")
        
        # 3. Duygu analizi (DeepFace - age, gender, emotion, race)
        print("[Pipeline] Duygu analizi yapılıyor...")
        # Her 5 frame'de bir analiz yap (performans için, tüm frame'ler çok yavaş olabilir)
        emotion_results = self.emotion_analyzer.analyze_frames(frames, sample_rate=5)
        emotion_summary = self.emotion_analyzer.get_emotion_summary(emotion_results)
        print(f"[Pipeline] Duygu analizi tamamlandı. {len([r for r in emotion_results if r is not None])}/{len(emotion_results)} frame'de yüz tespit edildi.")
        
        # 4. Göz teması analizi (MediaPipe Face Mesh)
        print("[Pipeline] Göz teması analizi yapılıyor...")
        # Her 5 frame'de bir analiz yap (performans için)
        eye_contact_results = self.eye_contact_analyzer.analyze_frames(frames, sample_rate=5)
        eye_contact_metrics = self.eye_contact_analyzer.calculate_eye_contact_metrics(eye_contact_results)
        print(f"[Pipeline] Göz teması analizi tamamlandı. {len([r for r in eye_contact_results if r is not None])}/{len(eye_contact_results)} frame'de yüz tespit edildi.")
        
        # 5. Ses analizi (librosa - basit)
        print("[Pipeline] Ses analizi yapılıyor...")
        voice_summary = None
        try:
            # Video'dan sesi çıkar
            audio_path = self.video_processor.extract_audio(video_path)
            # Ses analizini yap
            voice_analysis_result = self.voice_analyzer.analyze_audio(audio_path)
            voice_summary = voice_analysis_result.get('summary', {})
            # Geçici ses dosyasını sil
            import os
            if os.path.exists(audio_path):
                os.remove(audio_path)
            print("[Pipeline] Ses analizi tamamlandı.")
        except Exception as e:
            print(f"[Pipeline] Ses analizi hatası: {str(e)}")
            voice_summary = {
                'stress_level': 0.0,
                'confidence_score': 0.0,
                'error': str(e)
            }
        
        # 6. Davranışsal analiz (basit rule-based)
        print("[Pipeline] Davranışsal analiz yapılıyor...")
        behavioral_summary = None
        try:
            behavioral_result = self.behavioral_analyzer.analyze_behavior(
                emotion_results=emotion_results,
                eye_contact_results=eye_contact_results,
                voice_summary=voice_summary,
                emotion_summary=emotion_summary,
                eye_contact_metrics=eye_contact_metrics,
                video_info=video_info
            )
            behavioral_summary = behavioral_result.get('summary', {})
            print("[Pipeline] Davranışsal analiz tamamlandı.")
        except Exception as e:
            print(f"[Pipeline] Davranışsal analiz hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            behavioral_summary = {
                'suspicion_score': 0.0,
                'risk_level': 'low',
                'error': str(e)
            }
        
        # 7. Sonuçları birleştir
        print("[Pipeline] Sonuçlar birleştiriliyor...")
        report = self._generate_report(
            interview_id=interview_id,
            video_info=video_info,
            emotion_results=emotion_results,
            emotion_summary=emotion_summary,
            eye_contact_metrics=eye_contact_metrics,
            voice_summary=voice_summary,
            behavioral_summary=behavioral_summary
        )
        
        print(f"[Pipeline] Analiz tamamlandı: {interview_id}")
        
        return report
    
    def _generate_report(self, 
                        interview_id: str,
                        video_info: Dict,
                        emotion_results: List[Optional[Dict]],
                        emotion_summary: Dict,
                        eye_contact_metrics: Dict,
                        voice_summary: Optional[Dict] = None,
                        behavioral_summary: Optional[Dict] = None) -> Dict:
        """
        Analiz sonuçlarından rapor oluşturur.
        
        Args:
            interview_id: Mülakat ID
            video_info: Video bilgileri
            emotion_results: Frame bazlı duygu sonuçları (age, gender, emotion, race içerir)
            emotion_summary: Duygu analizi özeti
            eye_contact_metrics: Göz teması metrikleri
            voice_summary: Ses analizi özeti (opsiyonel)
            behavioral_summary: Davranışsal analiz özeti (opsiyonel)
            
        Returns:
            Yapılandırılmış rapor
        """
        # Genel değerlendirme skoru hesapla
        engagement_score = self._calculate_engagement_score(
            emotion_summary,
            eye_contact_metrics,
            voice_summary
        )
        
        # Öneriler oluştur
        recommendations = self._generate_recommendations(
            emotion_summary,
            eye_contact_metrics,
            voice_summary,
            behavioral_summary
        )
        
        # Age, Gender, Race bilgilerini çıkar
        age_info = self._extract_age_info(emotion_results)
        gender_info = self._extract_gender_info(emotion_results)
        race_info = self._extract_race_info(emotion_results)
        
        report = {
            "interview_id": interview_id,
            "duration_seconds": video_info.get('duration_seconds', 0),
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "video_info": {
                "fps": video_info.get('fps', 0),
                "resolution": {
                    "width": video_info.get('width', 0),
                    "height": video_info.get('height', 0)
                }
            },
            "emotion_analysis": {
                "dominant_emotion": emotion_summary.get('dominant_emotion'),
                "dominant_emotion_tr": emotion_summary.get('dominant_emotion_tr'),
                "emotion_distribution": emotion_summary.get('emotion_distribution', {}),
                "stability_score": emotion_summary.get('stability_score', 0.0),
                "coverage": emotion_summary.get('coverage', 0.0),
                "age_info": age_info,
                "gender_info": gender_info,
                "race_info": race_info
            },
            # Göz teması analizi: sadeleştirilmiş yapı
            # {
            #   "gaze_counts": { ... },
            #   "gaze_ratios": { ... }
            # }
            "eye_contact_analysis": eye_contact_metrics if eye_contact_metrics else {},
            "voice_analysis": voice_summary if voice_summary else {
                "status": "not_available",
                "note": "Ses analizi yapılamadı"
            },
            "behavioral_analysis": behavioral_summary if behavioral_summary else {
                "status": "not_available",
                "note": "Davranışsal analiz yapılamadı"
            },
            "overall_assessment": {
                "engagement_score": engagement_score,
                "recommendations": recommendations
            }
        }
        
        return report
    
    def _extract_age_info(self, emotion_results: List[Optional[Dict]]) -> Dict:
        """Age bilgilerini çıkarır."""
        ages = []
        for result in emotion_results:
            if result and 'age' in result:
                age = result['age']
                if isinstance(age, (int, float)) and age > 0:
                    ages.append(float(age))
        
        if not ages:
            return {"status": "not_available", "note": "Yaş bilgisi bulunamadı"}
        
        return {
            "average_age": float(np.mean(ages)),
            "min_age": float(np.min(ages)),
            "max_age": float(np.max(ages)),
            "std_age": float(np.std(ages)),
            "total_detections": len(ages)
        }
    
    def _extract_gender_info(self, emotion_results: List[Optional[Dict]]) -> Dict:
        """Gender bilgilerini çıkarır."""
        genders = []
        gender_confidences = []
        
        for result in emotion_results:
            if result and 'gender' in result:
                gender = result.get('gender', 'unknown')
                confidence = result.get('gender_confidence', 0.0)
                if gender != 'unknown':
                    genders.append(gender)
                    if confidence > 0:
                        gender_confidences.append(confidence)
        
        if not genders:
            return {"status": "not_available", "note": "Cinsiyet bilgisi bulunamadı"}
        
        # En sık görülen cinsiyet
        gender_counter = Counter(genders)
        dominant_gender = gender_counter.most_common(1)[0][0] if gender_counter else 'unknown'
        
        return {
            "dominant_gender": dominant_gender,
            "gender_distribution": dict(gender_counter),
            "average_confidence": float(np.mean(gender_confidences)) if gender_confidences else 0.0,
            "total_detections": len(genders)
        }
    
    def _extract_race_info(self, emotion_results: List[Optional[Dict]]) -> Dict:
        """Race bilgilerini çıkarır."""
        races = []
        race_confidences = []
        
        for result in emotion_results:
            if result and 'race' in result:
                race = result.get('race', 'unknown')
                confidence = result.get('race_confidence', 0.0)
                if race != 'unknown':
                    races.append(race)
                    if confidence > 0:
                        race_confidences.append(confidence)
        
        if not races:
            return {"status": "not_available", "note": "Irk bilgisi bulunamadı"}
        
        # En sık görülen ırk
        race_counter = Counter(races)
        dominant_race = race_counter.most_common(1)[0][0] if race_counter else 'unknown'
        
        return {
            "dominant_race": dominant_race,
            "race_distribution": dict(race_counter),
            "average_confidence": float(np.mean(race_confidences)) if race_confidences else 0.0,
            "total_detections": len(races)
        }
    
    def _calculate_engagement_score(self, 
                                   emotion_summary: Dict,
                                   eye_contact_metrics: Dict,
                                   voice_summary: Optional[Dict] = None) -> float:
        """
        Genel katılım skoru hesaplar (0-1 arası).
        
        Args:
            emotion_summary: Duygu analizi özeti
            eye_contact_metrics: Göz teması metrikleri
            voice_summary: Ses analizi özeti (opsiyonel)
            
        Returns:
            Katılım skoru (0-1)
        """
        # Duygu skoru (nötr/mutlu duygular pozitif)
        dominant_emotion = emotion_summary.get('dominant_emotion', 'neutral')
        emotion_scores = {
            'happy': 1.0,
            'neutral': 0.7,
            'surprise': 0.6,
            'sad': 0.4,
            'fear': 0.3,
            'angry': 0.2,
            'disgust': 0.2
        }
        emotion_score = emotion_scores.get(dominant_emotion, 0.5)
        
        # Duygu stabilitesi
        stability_score = emotion_summary.get('stability_score', 0.5)
        
        # Göz teması skoru (sadeleştirilmiş: sadece MobileGaze sınıfları)
        gaze_ratios = eye_contact_metrics.get('gaze_ratios', {}) if eye_contact_metrics else {}
        # Kamera yönü: modelin kamera için kullandığı etiket ("camera" veya "center")
        eye_contact_pct = gaze_ratios.get('camera', gaze_ratios.get('center', 0.0))
        eye_consistency = 0.5  # Artık ayrı bir consistency metriği hesaplanmıyor, nötr değer
        
        # Ses skorları (varsa)
        if voice_summary and 'error' not in voice_summary:
            confidence_score = voice_summary.get('confidence_score', 0.5)
            stress_score = voice_summary.get('stress_level', 0.5)
            # Stres tersine çevir (düşük stres = yüksek skor)
            stress_inverted = 1.0 - stress_score
            # Ağırlıklı ortalama (ses dahil)
            engagement_score = (
                emotion_score * 0.25 +
                stability_score * 0.15 +
                eye_contact_pct * 0.30 +
                confidence_score * 0.15 +
                stress_inverted * 0.15
            )
        else:
            # Ağırlıklı ortalama (ses yok)
            engagement_score = (
                emotion_score * 0.30 +
                stability_score * 0.20 +
                eye_contact_pct * 0.50
            )
        
        return float(np.clip(engagement_score, 0.0, 1.0))
    
    def _generate_recommendations(self,
                                 emotion_summary: Dict,
                                 eye_contact_metrics: Dict,
                                 voice_summary: Optional[Dict],
                                 behavioral_summary: Optional[Dict]) -> List[str]:
        """
        Analiz sonuçlarına göre öneriler oluşturur.
        
        Args:
            emotion_summary: Duygu analizi özeti
            eye_contact_metrics: Göz teması metrikleri
            voice_summary: Ses analizi özeti (opsiyonel)
            behavioral_summary: Davranışsal analiz özeti (opsiyonel)
            
        Returns:
            Öneriler listesi
        """
        recommendations = []
        
        # Göz teması önerileri (sadeleştirildi - sadece MobileGaze sınıfları raporlanır,
        # ekstra yorum üretilmez)
        
        # Duygu stabilitesi önerileri
        stability_score = emotion_summary.get('stability_score', 0.5)
        if stability_score < 0.5:
            recommendations.append("Duygu değişkenliği yüksek. Adayın stres seviyesi gözlemlenmeli.")
        
        # Ses analizi önerileri
        if voice_summary and 'error' not in voice_summary:
            stress_level = voice_summary.get('stress_level', 0.0)
            confidence_score = voice_summary.get('confidence_score', 0.0)
            
            if stress_level > 0.6:
                recommendations.append(f"Ses analizi yüksek stres seviyesi gösteriyor ({stress_level:.2f}). Adayın rahatlatılması önerilir.")
            
            if confidence_score < 0.4:
                recommendations.append(f"Ses analizi düşük güven skoru gösteriyor ({confidence_score:.2f}). Adayın kendine güveni değerlendirilmeli.")
        
        # Davranışsal analiz önerileri
        if behavioral_summary and 'error' not in behavioral_summary:
            suspicion_score = behavioral_summary.get('suspicion_score', 0.0)
            reading_suspicion = behavioral_summary.get('reading_suspicion', 0.0)
            
            if reading_suspicion > 0.5:
                recommendations.append("Okuma/cheating şüphesi tespit edildi. Adayın davranışları gözlemlenmeli.")
            
            if suspicion_score > 0.6:
                recommendations.append("Yüksek şüphe skoru tespit edildi. Detaylı değerlendirme önerilir.")
        
        if not recommendations:
            recommendations.append("Genel olarak iyi bir performans gözlemlendi.")
        
        return recommendations


if __name__ == "__main__":
    # Test kodu
    pipeline = InterviewAnalysisPipeline()
    print("InterviewAnalysisPipeline modülü hazır.")
