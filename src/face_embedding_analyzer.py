"""
Yüz Embedding Analizi Modülü
DeepFace Facenet512 kullanarak yüz embedding'leri çıkarır.
Identity-agnostic yüz temsil vektörleri üretir (duygu etiketi değil, regülasyon/tutarlılık için).
"""

import numpy as np
from typing import List, Dict, Optional
from deepface import DeepFace
import cv2
from PIL import Image
import tempfile
import os


class FaceEmbeddingAnalyzer:
    """
    Yüz embedding analizi yapan sınıf.
    DeepFace Facenet512 kullanarak 512 boyutlu yüz embedding'leri çıkarır.
    Bu embedding'ler duygu etiketi için değil, yüz regülasyonu ve tutarlılık ölçümü için kullanılır.
    """
    
    def __init__(self, model_name: str = "Facenet512", enforce_detection: bool = False):
        """
        Yüz embedding analizcisini başlatır.
        
        Args:
            model_name: DeepFace model adı (Facenet512 önerilir)
            enforce_detection: Yüz bulunamazsa hata ver (False ise None döner)
        """
        self.model_name = model_name
        self.enforce_detection = enforce_detection
    
    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Tek bir frame'den yüz embedding'i çıkarır.
        
        Args:
            frame: BGR formatında görüntü (OpenCV formatı)
            
        Returns:
            512 boyutlu embedding vektörü veya None (yüz bulunamazsa)
        """
        try:
            # BGR'den RGB'ye çevir
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Geçici dosya oluştur
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                pil_image = Image.fromarray(rgb_frame)
                pil_image.save(tmp_path, 'JPEG', quality=95)
            
            try:
                # DeepFace represent fonksiyonu ile embedding çıkar
                embedding_objs = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=self.model_name,
                    enforce_detection=self.enforce_detection
                )
                
                # Sonuç formatını düzenle
                if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                    embedding = embedding_objs[0].get('embedding', None)
                    if embedding:
                        return np.array(embedding)
                
                return None
            finally:
                # Geçici dosyayı sil
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
        except Exception as e:
            # Hata durumunda None döndür
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            
            if self._error_count < 3:
                print(f"[FaceEmbeddingAnalyzer] Hata (embedding çıkarma): {str(e)}")
                self._error_count += 1
            
            return None
    
    def extract_embeddings(self, frames: List[np.ndarray], sample_rate: int = 5) -> List[Optional[np.ndarray]]:
        """
        Birden fazla frame'den yüz embedding'leri çıkarır.
        
        Args:
            frames: Frame'lerin listesi
            sample_rate: Her N frame'de bir analiz yap (performans için, 1 = tüm frame'ler)
            
        Returns:
            Her frame için embedding vektörleri listesi (None = yüz bulunamadı)
        """
        embeddings = []
        total_frames = len(frames)
        extracted_count = 0
        success_count = 0
        
        # Her N frame'de bir analiz yap
        for i, frame in enumerate(frames):
            if i % sample_rate == 0 or i == total_frames - 1:
                extracted_count += 1
                embedding = self.extract_embedding(frame)
                if embedding is not None:
                    success_count += 1
                embeddings.append(embedding)
            else:
                # Analiz edilmeyen frame'ler için None
                embeddings.append(None)
        
        # Debug bilgisi
        if extracted_count > 0:
            success_rate = (success_count / extracted_count) * 100
            print(f"[FaceEmbeddingAnalyzer] {extracted_count} frame'den embedding çıkarıldı, {success_count} başarılı ({success_rate:.1f}%)")
        
        return embeddings
    
    def calculate_embedding_stability(self, embeddings: List[Optional[np.ndarray]]) -> Dict:
        """
        Embedding'lerin zamansal stabilitesini hesaplar.
        Düşük varyans = yüksek stabilite (tutarlı yüz ifadesi).
        
        Args:
            embeddings: Embedding vektörleri listesi
            
        Returns:
            Stabilite metrikleri
        """
        # Geçerli embedding'leri filtrele
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if len(valid_embeddings) < 2:
            return {
                'stability_score': 0.0,
                'mean_variance': 1.0,
                'temporal_consistency': 0.0,
                'total_embeddings': len(valid_embeddings)
            }
        
        # Embedding'leri numpy array'e çevir
        embedding_matrix = np.array(valid_embeddings)  # Shape: (n_frames, 512)
        
        # Her boyut için varyans hesapla
        variances = np.var(embedding_matrix, axis=0)  # Shape: (512,)
        mean_variance = np.mean(variances)
        
        # Stabilite skoru (düşük varyans = yüksek stabilite)
        # Variance'ı normalize et (0-1 arası, 1 = çok değişken, 0 = çok stabil)
        # Facenet512 embedding'leri genellikle -1 ile 1 arasında normalize edilmiş
        # Bu yüzden variance 0-4 arası olabilir, normalize edelim
        normalized_variance = min(1.0, mean_variance / 0.5)  # 0.5 threshold
        stability_score = 1.0 - normalized_variance
        
        # Temporal consistency (ardışık embedding'ler arası benzerlik)
        if len(valid_embeddings) > 1:
            similarities = []
            for i in range(len(valid_embeddings) - 1):
                # Cosine similarity
                emb1 = valid_embeddings[i]
                emb2 = valid_embeddings[i + 1]
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(similarity)
            
            temporal_consistency = np.mean(similarities) if similarities else 0.0
        else:
            temporal_consistency = 0.0
        
        return {
            'stability_score': float(stability_score),
            'mean_variance': float(mean_variance),
            'temporal_consistency': float(temporal_consistency),
            'total_embeddings': len(valid_embeddings),
            'embedding_dimension': embedding_matrix.shape[1] if len(valid_embeddings) > 0 else 0
        }


if __name__ == "__main__":
    # Test kodu
    analyzer = FaceEmbeddingAnalyzer()
    print("FaceEmbeddingAnalyzer modülü hazır.")
    print(f"Model: {analyzer.model_name}")
