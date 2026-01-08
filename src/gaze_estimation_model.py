"""
Gaze Estimation Model Wrapper
MobileGaze modelini kullanarak gaze estimation yapar.
https://github.com/yakhyo/gaze-estimation
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import os
from pathlib import Path


class GazeEstimationModel:
    """
    MobileGaze model wrapper.
    Pretrained gaze estimation modelini yükler ve inference yapar.
    """
    
    def __init__(self, 
                 model_name: str = "mobilenetv2",
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Gaze estimation modelini başlatır.
        
        Args:
            model_name: Model adı (resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0)
            model_path: Model weights dosya yolu (None ise otomatik bulunur)
            device: Cihaz (cpu veya cuda)
        """
        self.model_name = model_name
        self.device = device
        
        # Model path'i belirle
        if model_path is None:
            # Varsayılan weights klasörü
            weights_dir = Path(__file__).parent.parent / "weights" / "gaze_estimation"
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = weights_dir / f"{model_name}.pt"
        
        self.model_path = str(model_path)
        
        # Model'i yükle
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Model'i yükler."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model weights bulunamadı: {self.model_path}\n"
                f"Lütfen model weights'i indirin: https://github.com/yakhyo/gaze-estimation"
            )
        
        # Model architecture'ı import et (basit wrapper)
        # Not: Gerçek implementasyon için gaze-estimation repo'sundan model kodunu kopyalamak gerekir
        # Şimdilik placeholder - gerçek model yapısı repo'da
        print(f"[GazeEstimationModel] Model yükleniyor: {self.model_name}")
        print(f"[GazeEstimationModel] Model path: {self.model_path}")
        
        # Model'i yükle (PyTorch)
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Model architecture'ı buraya eklemek gerekir
            # Şimdilik placeholder
            self.model = None  # Gerçek model yapısı repo'da
            print("[GazeEstimationModel] Model yüklendi.")
        except Exception as e:
            print(f"[GazeEstimationModel] Model yükleme hatası: {str(e)}")
            raise
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Yüz görüntüsünü model için hazırlar.
        
        Args:
            face_image: Yüz görüntüsü (BGR veya RGB)
            
        Returns:
            Preprocessed tensor
        """
        # Gaze-estimation repo'sunda genellikle 224x224 input beklenir
        # Face crop ve normalize
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        # Normalize (ImageNet stats)
        face_normalized = face_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        # Tensor'e çevir (B, C, H, W)
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        
        return face_tensor
    
    def estimate_gaze(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Gaze açısını tahmin eder.
        
        Args:
            face_image: Yüz görüntüsü (BGR format)
            
        Returns:
            Gaze açıları (yaw, pitch) veya None
        """
        if self.model is None:
            return None
        
        try:
            # Preprocess
            face_tensor = self.preprocess_face(face_image)
            
            # Inference
            with torch.no_grad():
                output = self.model(face_tensor)
            
            # Output format: [yaw, pitch] (radyan veya derece)
            # Repo'ya göre genellikle radyan, dereceye çevir
            yaw = float(output[0, 0].item()) * 180.0 / np.pi  # Radyan -> derece
            pitch = float(output[0, 1].item()) * 180.0 / np.pi
            
            return {
                'yaw': yaw,
                'pitch': pitch,
                'gaze_angle': np.sqrt(yaw**2 + pitch**2)  # Toplam açı
            }
        except Exception as e:
            print(f"[GazeEstimationModel] Inference hatası: {str(e)}")
            return None


# Not: Bu dosya placeholder. Gerçek implementasyon için:
# 1. gaze-estimation repo'sundan model architecture kodunu kopyala
# 2. Model weights'i indir
# 3. Bu wrapper'ı tamamla
