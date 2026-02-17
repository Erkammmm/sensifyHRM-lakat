"""
Video İşleme Modülü
Video dosyalarını frame'lere ayırır ve ön işleme yapar.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import os
import math


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
        if not math.isfinite(original_fps) or original_fps <= 0:
            original_fps = float(self.target_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Uzun videolar için hedef FPS adaptasyonu (bellek kullanımını azaltır)
        duration_seconds = (total_frames / original_fps) if original_fps > 0 else 0.0
        effective_target_fps = float(self.target_fps)
        if duration_seconds >= 1800:  # 30 dk+
            effective_target_fps = min(effective_target_fps, 2.0)
        elif duration_seconds >= 1200:  # 20 dk+
            effective_target_fps = min(effective_target_fps, 5.0)
        elif duration_seconds >= 600:  # 10 dk+
            effective_target_fps = min(effective_target_fps, 10.0)

        # Frame'leri saklamak için liste
        frames: List[np.ndarray] = []

        # Frame skip hesaplama (eğer orijinal FPS hedef FPS'den yüksekse)
        frame_skip = max(1, int(original_fps / effective_target_fps)) if effective_target_fps > 0 else 1
        
        frame_count = 0
        target_width = self.target_width
        target_height = self.target_height

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Sadece belirli frame'leri al (FPS normalizasyonu için)
            if frame_count % frame_skip == 0:
                # Frame'i yeniden boyutlandır (bellek hatasında çözünürlüğü düşür)
                try:
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                except Exception:
                    # Daha küçük çözünürlüğe düş
                    target_width = max(320, target_width // 2)
                    target_height = max(180, target_height // 2)
                    resized_frame = cv2.resize(frame, (target_width, target_height))
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
        if not math.isfinite(original_fps) or original_fps <= 0:
            original_fps = float(self.target_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = (total_frames / original_fps) if original_fps > 0 else 0.0
        effective_target_fps = float(self.target_fps)
        if duration_seconds >= 1800:
            effective_target_fps = min(effective_target_fps, 2.0)
        elif duration_seconds >= 1200:
            effective_target_fps = min(effective_target_fps, 5.0)
        elif duration_seconds >= 600:
            effective_target_fps = min(effective_target_fps, 10.0)

        frame_skip = max(1, int(original_fps / effective_target_fps)) if effective_target_fps > 0 else 1
        frames_with_timestamps: List[Tuple[np.ndarray, float]] = []
        frame_count = 0
        target_width = self.target_width
        target_height = self.target_height

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                # Timestamp hesapla (saniye cinsinden)
                timestamp = frame_count / original_fps
                try:
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                except Exception:
                    target_width = max(320, target_width // 2)
                    target_height = max(180, target_height // 2)
                    resized_frame = cv2.resize(frame, (target_width, target_height))
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
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if not math.isfinite(fps) or fps <= 0:
            fps = 0.0
        if not math.isfinite(frame_count) or frame_count < 0:
            frame_count = 0.0
        if not math.isfinite(width) or width < 0:
            width = 0.0
        if not math.isfinite(height) or height < 0:
            height = 0.0

        duration_seconds = (frame_count / fps) if fps > 0 else 0.0

        info = {
            'fps': float(fps),
            'frame_count': int(frame_count) if frame_count == frame_count else 0,
            'width': int(width) if width == width else 0,
            'height': int(height) if height == height else 0,
            'duration_seconds': float(duration_seconds)
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
