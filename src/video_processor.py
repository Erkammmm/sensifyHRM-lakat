"""
Video İşleme Modülü
Video dosyalarını frame'lere ayırır ve ön işleme yapar.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class VideoProcessor:
    """
    Video dosyalarını işleyen sınıf.
    Video'yu frame'lere ayırır ve yüz tespiti için hazırlar.
    """
    
    def __init__(self, target_fps: int = 30, target_width: int = 1280, target_height: int = 720):
        """
        Video işlemciyi başlatır.
        
        Args:
            target_fps: Hedef frame rate (saniyede frame sayısı)
            target_width: Hedef genişlik (piksel)
            target_height: Hedef yükseklik (piksel)
        """
        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Video dosyasından frame'leri çıkarır.
        
        Args:
            video_path: Video dosyasının yolu
            
        Returns:
            Frame'lerin listesi (numpy array'ler)
        """
        # Video dosyasını aç
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        # Video özelliklerini al
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Frame'leri saklamak için liste
        frames = []
        
        # Frame skip hesaplama (eğer orijinal FPS hedef FPS'den yüksekse)
        frame_skip = max(1, int(original_fps / self.target_fps))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sadece belirli frame'leri al (FPS normalizasyonu için)
            if frame_count % frame_skip == 0:
                # Frame'i yeniden boyutlandır
                resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                frames.append(resized_frame)
            
            frame_count += 1
        
        cap.release()
        
        return frames
    
    def extract_frames_with_timestamps(self, video_path: str) -> List[Tuple[np.ndarray, float]]:
        """
        Video'dan frame'leri timestamp'leriyle birlikte çıkarır.
        
        Args:
            video_path: Video dosyasının yolu
            
        Returns:
            (frame, timestamp) tuple'larının listesi
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(original_fps / self.target_fps))
        
        frames_with_timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Timestamp hesapla (saniye cinsinden)
                timestamp = frame_count / original_fps
                
                resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                frames_with_timestamps.append((resized_frame, timestamp))
            
            frame_count += 1
        
        cap.release()
        
        return frames_with_timestamps
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Video dosyası hakkında bilgi döndürür.
        
        Args:
            video_path: Video dosyasının yolu
            
        Returns:
            Video bilgilerini içeren dictionary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Video dosyası açılamadı: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        
        return info
    
    def extract_audio(self, video_path: str, output_audio_path: Optional[str] = None) -> str:
        """
        Video'dan sesi çıkarır (FFmpeg kullanarak).
        
        Args:
            video_path: Video dosyasının yolu
            output_audio_path: Çıktı ses dosyası yolu (None ise otomatik oluşturulur)
            
        Returns:
            Çıktı ses dosyasının yolu
        """
        import subprocess
        
        if output_audio_path is None:
            # Otomatik dosya adı oluştur
            base_name = os.path.splitext(video_path)[0]
            output_audio_path = f"{base_name}_audio.wav"
        
        # FFmpeg ile ses çıkarma
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Video stream'i yok say
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono channel
            '-y',  # Overwrite output file
            output_audio_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
            return output_audio_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Ses çıkarma hatası: {e}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg bulunamadı. Lütfen FFmpeg'i yükleyin.")


if __name__ == "__main__":
    # Test kodu
    processor = VideoProcessor()
    # Test için bir video yolu gerekli
    print("VideoProcessor modülü hazır.")
