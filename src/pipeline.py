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
from .mediapipe_face_gaze_analyzer import MediapipeFaceGazeAnalyzer
from .voice_analyzer import VoiceAnalyzer
from .frame_summarizer import FrameAnalysisSummarizer


class InterviewAnalysisPipeline:
    """
    Mülakat analiz pipeline'ı.
    Video'yu işler, tüm analizleri yapar ve rapor oluşturur.
    """
    
    def __init__(self):
        """Pipeline'ı başlatır ve modülleri yükler."""
        self.video_processor = VideoProcessor()
        # DeepFace + MobileGaze yerine tamamen Mediapipe tabanlı analizör
        self.face_gaze_analyzer = MediapipeFaceGazeAnalyzer(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            frame_skip=3,  # FRAME_SKIP: her 3 karede bir analiz
        )
        self.voice_analyzer = VoiceAnalyzer()
    
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
        
        # 3. Mediapipe tabanlı yüz + ham özellikler + bakış yönü analizi
        print("[Pipeline] Mediapipe yüz + ham özellikler + bakış yönü analizi yapılıyor...")
        fps = video_info.get("fps", 0) or 30.0
        frame_analysis = self.face_gaze_analyzer.analyze_frames(frames, fps=fps)
        valid_face_frames = len([r for r in frame_analysis if r is not None])
        print(f"[Pipeline] Mediapipe analizi tamamlandı. {valid_face_frames}/{len(frame_analysis)} frame'de yüz tespit edildi.")
        
        # 4. Ses analizi (librosa - sadece raw özellikler)
        print("[Pipeline] Ses analizi yapılıyor...")
        voice_summary = None
        try:
            # Video'dan sesi çıkar
            audio_path = self.video_processor.extract_audio(video_path)
            # Ses analizini yap
            voice_analysis_result = self.voice_analyzer.analyze_audio(audio_path)
            # Artık sadece raw_voice_features kullanıyoruz
            voice_summary = voice_analysis_result.get('raw_voice_features', {})
            # Geçici ses dosyasını sil
            import os
            if os.path.exists(audio_path):
                os.remove(audio_path)
            print("[Pipeline] Ses analizi tamamlandı.")
        except Exception as e:
            print(f"[Pipeline] Ses analizi hatası: {str(e)}")
            voice_summary = {
                'status': 'error',
                'message': str(e),
                'raw_voice_features': {}
            }
        
        # 5. Frame analizini özetle
        print("[Pipeline] Frame analizi özetleniyor...")
        frame_summary = FrameAnalysisSummarizer.summarize(frame_analysis)
        
        # 6. Sonuçları birleştir
        print("[Pipeline] Sonuçlar birleştiriliyor...")
        report = self._generate_report(
            interview_id=interview_id,
            video_info=video_info,
            frame_analysis=frame_analysis,
            frame_summary=frame_summary,
            voice_summary=voice_summary
        )
        
        print(f"[Pipeline] Analiz tamamlandı: {interview_id}")
        
        return report
    
    def _generate_report(
        self,
        interview_id: str,
        video_info: Dict,
        frame_analysis: List[Optional[Dict]],
        frame_summary: Dict,
        voice_summary: Optional[Dict] = None,
    ) -> Dict:
        """
        Analiz sonuçlarından rapor oluşturur.
        
        Args:
            interview_id: Mülakat ID
            video_info: Video bilgileri
            frame_analysis: Her işlenen frame için
                {
                    "timestamp": float,
                    "raw_features": {
                        "mouth_width_norm": float,
                        "mouth_height_norm": float,
                        "eye_opening_norm": float,
                        "brow_distance_norm": float,
                        "jaw_open_norm": float,
                    },
                    "gaze": {...}
                }
            voice_summary: Ses analizi özeti (opsiyonel, raw voice features)
            
        Returns:
            Yapılandırılmış rapor
        """
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
            # Frame bazlı Mediapipe analizi (ham özellikler + gaze vektörü)
            # Not: Ham veri çok kalabalık olabilir, Gemini API için frame_summary kullanılmalı
            "frame_analysis": [
                fa for fa in frame_analysis if fa is not None
            ],
            # Frame analizi özeti (Gemini API için sadeleştirilmiş)
            "frame_summary": frame_summary,
            # Ses analizi: sadece raw_voice_features içeren sade yapı
            "voice_analysis": voice_summary if voice_summary else {
                "status": "not_available",
                "note": "Ses analizi yapılamadı"
            }
        }
        
        return report
    


if __name__ == "__main__":
    # Test kodu
    pipeline = InterviewAnalysisPipeline()
    print("InterviewAnalysisPipeline modülü hazır.")
