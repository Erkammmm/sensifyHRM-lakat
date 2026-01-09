"""
Gaze Estimation Model Wrapper
Gaze estimation modelini kullanarak yaw-pitch tahmini yapar.
https://github.com/yakhyo/gaze-estimation

Model weights otomatik olarak indirilir (ilk kullanımda).
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import os
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Model architecture'ları import et - lazy import (sadece gerektiğinde)
# torchvision import'u model dosyalarında yapılıyor, burada sadece model fonksiyonlarını import ediyoruz
def _import_models():
    """Model architecture'ları lazy import eder."""
    try:
        from src.models import resnet18, resnet34, resnet50, mobilenet_v2, mobileone_s0
        return resnet18, resnet34, resnet50, mobilenet_v2, mobileone_s0
    except ImportError as e1:
        # Alternatif import path
        try:
            import sys
            from pathlib import Path
            models_path = Path(__file__).parent / "models"
            if models_path.exists():
                sys.path.insert(0, str(Path(__file__).parent))
                from models import resnet18, resnet34, resnet50, mobilenet_v2, mobileone_s0
                return resnet18, resnet34, resnet50, mobilenet_v2, mobileone_s0
            else:
                raise ImportError(f"Models klasörü bulunamadı: {models_path}")
        except ImportError as e2:
            raise ImportError(
                f"Model dosyaları bulunamadı. Lütfen gaze-estimation repo'sundan "
                f"model dosyalarını src/models/ klasörüne kopyalayın.\n"
                f"Import hatası 1: {e1}\n"
                f"Import hatası 2: {e2}"
            )


class GazeEstimationModel:
    """
    MobileGaze model wrapper.
    Pretrained gaze estimation modelini yükler ve inference yapar.
    Model weights otomatik olarak indirilir (ilk kullanımda).
    """
    
    # Model weights URL'leri (GitHub releases veya direct download)
    # Not: Gerçek URL'ler repo'ya göre güncellenmeli
    MODEL_URLS = {
        "resnet18": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18.pt",
        "resnet34": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34.pt",
        "resnet50": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet50.pt",
        "mobilenetv2": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2.pt",
        "mobileone_s0": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0.pt"
    }
    
    def __init__(self, 
                 model_name: str = "mobilenetv2",
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Gaze estimation modelini başlatır.
        
        Args:
            model_name: Model adı (resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0)
            model_path: Model weights dosya yolu (None ise otomatik bulunur/indirilir)
            device: Cihaz (cpu veya cuda)
        """
        self.model_name = model_name
        self.device = device
        
        # Model path'i belirle
        if model_path is None:
            # Varsayılan weights klasörü (proje kökünde)
            weights_dir = Path(__file__).parent.parent / "weights" / "gaze_estimation"
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = weights_dir / f"{model_name}.pt"
        
        self.model_path = Path(model_path)
        
        # Model weights yoksa otomatik indir
        if not self.model_path.exists():
            self._download_model()
        
        # Model'i yükle
        self.model = None
        self.num_classes = None  # Checkpoint'ten çıkarılacak
        self._load_model()
    
    def _download_model(self):
        """Model weights'i otomatik indirir."""
        if self.model_name not in self.MODEL_URLS:
            raise ValueError(
                f"Bilinmeyen model: {self.model_name}\n"
                f"Desteklenen modeller: {list(self.MODEL_URLS.keys())}"
            )
        
        url = self.MODEL_URLS[self.model_name]
        print(f"[GazeEstimationModel] Model weights indiriliyor: {self.model_name}")
        print(f"[GazeEstimationModel] URL: {url}")
        
        try:
            # Progress bar ile indir
            def download_progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\r[GazeEstimationModel] İndiriliyor: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, str(self.model_path), reporthook=download_progress_hook)
            print(f"\n[GazeEstimationModel] Model weights indirildi: {self.model_path}")
        except Exception as e:
            print(f"\n[GazeEstimationModel] İndirme hatası: {str(e)}")
            print(f"[GazeEstimationModel] Manuel indirme için: {url}")
            raise FileNotFoundError(
                f"Model weights indirilemedi. Lütfen manuel olarak indirin:\n"
                f"URL: {url}\n"
                f"Hedef: {self.model_path}"
            )
    
    def _load_model(self):
        """Model'i yükler."""
        print(f"[GazeEstimationModel] Model yükleniyor: {self.model_name}")
        print(f"[GazeEstimationModel] Model path: {self.model_path}")
        
        try:
            # Model architecture'ları import et (lazy import)
            resnet18, resnet34, resnet50, mobilenet_v2, mobileone_s0 = _import_models()
            
            # Checkpoint'i yükle
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            
            # Checkpoint'ten num_classes (bins) değerini çıkar
            # State dict'ten fc_yaw veya fc_pitch weight shape'inden anla
            bins = 66  # Varsayılan
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            
            # State dict'ten bins sayısını çıkar
            if isinstance(state_dict, dict):
                # fc_yaw.weight veya fc_pitch.weight shape'inden bins sayısını bul
                for key in state_dict.keys():
                    if 'fc_yaw.weight' in key or 'fc_pitch.weight' in key:
                        weight_shape = state_dict[key].shape
                        if len(weight_shape) >= 1:
                            bins = int(weight_shape[0])  # İlk dimension = num_classes
                            print(f"[GazeEstimationModel] Checkpoint'ten num_classes tespit edildi: {bins}")
                            break
                
                # Eğer checkpoint'te metadata varsa onu kullan
                if isinstance(checkpoint, dict):
                    if 'num_classes' in checkpoint:
                        bins = int(checkpoint['num_classes'])
                        print(f"[GazeEstimationModel] Checkpoint metadata'dan num_classes: {bins}")
                    elif 'bins' in checkpoint:
                        bins = int(checkpoint['bins'])
                        print(f"[GazeEstimationModel] Checkpoint metadata'dan bins: {bins}")
            
            # num_classes'ı sakla (inference için gerekli)
            self.num_classes = bins
            
            # Model architecture'ı oluştur
            if self.model_name == "resnet18":
                self.model = resnet18(pretrained=False, num_classes=bins)
            elif self.model_name == "resnet34":
                self.model = resnet34(pretrained=False, num_classes=bins)
            elif self.model_name == "resnet50":
                self.model = resnet50(pretrained=False, num_classes=bins)
            elif self.model_name == "mobilenetv2":
                self.model = mobilenet_v2(pretrained=False, num_classes=bins)
            elif self.model_name == "mobileone_s0":
                self.model = mobileone_s0(pretrained=False, num_classes=bins, inference_mode=True)
            else:
                raise ValueError(f"Desteklenmeyen model: {self.model_name}")
            
            # State dict'i yükle
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                # Eğer direkt state_dict ise
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.to(self.device)
            # Model float64 (double) bekliyor, bu yüzden double() kullanıyoruz
            self.model = self.model.double()
            self.model.eval()
            print(f"[GazeEstimationModel] Model yüklendi: {self.model_name} (num_classes={bins})")
        except Exception as e:
            print(f"[GazeEstimationModel] Model yükleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
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
        # Model float64 bekliyor, bu yüzden float64 kullanıyoruz
        face_normalized = face_resized.astype(np.float64) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float64)
        face_normalized = (face_normalized - mean) / std
        
        # Tensor'e çevir (B, C, H, W) - float64 (double) olarak
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        # Model float64 bekliyor, bu yüzden double() kullanıyoruz
        face_tensor = face_tensor.double()
        
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
            
            # Model output format: (pitch, yaw) tuple
            # Modeller classification yapıyor (bins sayısı kadar class)
            # Softmax + argmax ile açıya çevirmemiz gerekiyor
            if isinstance(output, tuple) and len(output) == 2:
                pitch_logits, yaw_logits = output
                
                # Softmax + argmax ile class index'i bul
                pitch_probs = torch.softmax(pitch_logits, dim=1)
                yaw_probs = torch.softmax(yaw_logits, dim=1)
                
                pitch_class = torch.argmax(pitch_probs, dim=1)
                yaw_class = torch.argmax(yaw_probs, dim=1)
                
                # Class index'i açıya çevir
                # Bins: -90 ile +90 arası (veya -angle ile +angle arası)
                # Binwidth genellikle 2 veya 4 derece
                angle_range = 90  # Varsayılan: -90 ile +90 derece
                num_bins = self.num_classes if self.num_classes else pitch_logits.shape[1]  # Model'den bins sayısını al
                binwidth = (2 * angle_range) / num_bins  # Bin genişliği
                
                # Class index'i açıya çevir (center of bin)
                pitch = float(pitch_class[0].item()) * binwidth - angle_range + (binwidth / 2)
                yaw = float(yaw_class[0].item()) * binwidth - angle_range + (binwidth / 2)
            else:
                # Eğer farklı format ise (regression)
                pitch = float(output[0, 0].item())
                yaw = float(output[0, 1].item())
            
            # Açıları derece cinsinden al (zaten derece olmalı, ama kontrol et)
            # Eğer radyan ise dereceye çevir
            if abs(pitch) > np.pi or abs(yaw) > np.pi:
                # Zaten derece
                pass
            else:
                # Radyan, dereceye çevir
                pitch = pitch * 180.0 / np.pi
                yaw = yaw * 180.0 / np.pi
            
            # Gaze angle hesapla (toplam sapma açısı)
            gaze_angle = np.sqrt(yaw**2 + pitch**2)
            
            # Gaze sınıfını belirle (camera/left/right/up/down)
            # Camera: yaw ve pitch küçükse (kameraya bakıyor)
            # Left: yaw negatif ve büyükse
            # Right: yaw pozitif ve büyükse
            # Up: pitch pozitif ve büyükse
            # Down: pitch negatif ve büyükse
            threshold = 15.0  # derece
            
            if abs(yaw) <= threshold and abs(pitch) <= threshold:
                gaze_class = "camera"
            elif abs(yaw) > abs(pitch):
                # Yatay bakış daha dominant
                if yaw < -threshold:
                    gaze_class = "left"
                elif yaw > threshold:
                    gaze_class = "right"
                else:
                    gaze_class = "camera"
            else:
                # Dikey bakış daha dominant
                if pitch > threshold:
                    gaze_class = "up"
                elif pitch < -threshold:
                    gaze_class = "down"
                else:
                    gaze_class = "camera"
            
            return {
                'yaw': yaw,
                'pitch': pitch,
                'gaze_angle': gaze_angle,
                'gaze_class': gaze_class
            }
        except Exception as e:
            print(f"[GazeEstimationModel] Inference hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# Not: Bu dosya placeholder. Gerçek implementasyon için:
# 1. gaze-estimation repo'sundan model architecture kodunu kopyala
# 2. Model weights'i indir
# 3. Bu wrapper'ı tamamla
