"""
Ses Embedding Analizi Modülü
wav2vec 2.0 kullanarak ses embedding'leri çıkarır.
Prosodik kontrol ve bilişsel yük (cognitive load) çıkarımı için.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import librosa
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch


class VoiceEmbeddingAnalyzer:
    """
    Ses embedding analizi yapan sınıf.
    wav2vec 2.0 pretrained model kullanarak ses embedding'leri çıkarır.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Ses embedding analizcisini başlatır.
        
        Args:
            model_name: HuggingFace model adı (wav2vec2-base-960h önerilir)
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Model'i lazy loading ile yükle (ilk kullanımda)."""
        if not self._model_loaded:
            try:
                print(f"[VoiceEmbeddingAnalyzer] Model yükleniyor: {self.model_name}")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2Model.from_pretrained(self.model_name)
                self.model.eval()  # Evaluation mode
                self._model_loaded = True
                print("[VoiceEmbeddingAnalyzer] Model yüklendi.")
            except Exception as e:
                print(f"[VoiceEmbeddingAnalyzer] Model yükleme hatası: {str(e)}")
                raise
    
    def extract_embedding(self, audio_path: str, segment_duration: float = 2.0) -> Optional[np.ndarray]:
        """
        Ses dosyasından embedding çıkarır.
        
        Args:
            audio_path: Ses dosyasının yolu
            segment_duration: Her segment'in süresi (saniye)
            
        Returns:
            Ortalama embedding vektörü veya None
        """
        try:
            # Model'i yükle
            self._load_model()
            
            # Ses dosyasını yükle
            audio, sr = librosa.load(audio_path, sr=16000)  # wav2vec2 16kHz bekliyor
            
            if len(audio) == 0:
                return None
            
            # Ses dosyasını segment'lere böl (2 saniyelik)
            segment_length = int(segment_duration * sr)
            segments = []
            
            for i in range(0, len(audio), segment_length):
                segment = audio[i:i + segment_length]
                if len(segment) >= segment_length // 2:  # En az yarım segment
                    # Padding gerekirse
                    if len(segment) < segment_length:
                        segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
                    segments.append(segment)
            
            if not segments:
                return None
            
            # Her segment için embedding çıkar
            all_embeddings = []
            
            with torch.no_grad():
                for segment in segments:
                    # Process input
                    inputs = self.processor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
                    
                    # Model inference
                    outputs = self.model(**inputs)
                    
                    # Hidden states'ten embedding al (son katman)
                    # Shape: (batch_size, sequence_length, hidden_size)
                    hidden_states = outputs.last_hidden_state
                    
                    # Ortalama pooling (sequence_length üzerinden)
                    embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
                    all_embeddings.append(embedding)
            
            # Tüm segment'lerin ortalamasını al
            if all_embeddings:
                mean_embedding = np.mean(all_embeddings, axis=0)
                return mean_embedding
            
            return None
            
        except Exception as e:
            print(f"[VoiceEmbeddingAnalyzer] Hata (embedding çıkarma): {str(e)}")
            return None
    
    def extract_temporal_embeddings(self, audio_path: str, window_duration: float = 2.0, 
                                   hop_duration: float = 1.0) -> List[np.ndarray]:
        """
        Ses dosyasından zamansal embedding'ler çıkarır (sliding window).
        
        Args:
            audio_path: Ses dosyasının yolu
            window_duration: Pencere süresi (saniye)
            hop_duration: Pencere kaydırma süresi (saniye)
            
        Returns:
            Her pencere için embedding vektörleri listesi
        """
        try:
            # Model'i yükle
            self._load_model()
            
            # Ses dosyasını yükle
            audio, sr = librosa.load(audio_path, sr=16000)
            
            if len(audio) == 0:
                return []
            
            window_length = int(window_duration * sr)
            hop_length = int(hop_duration * sr)
            
            embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(audio) - window_length + 1, hop_length):
                    segment = audio[i:i + window_length]
                    
                    # Process input
                    inputs = self.processor(segment, sampling_rate=sr, return_tensors="pt", padding=True)
                    
                    # Model inference
                    outputs = self.model(**inputs)
                    
                    # Hidden states'ten embedding al
                    hidden_states = outputs.last_hidden_state
                    
                    # Ortalama pooling
                    embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
                    embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            print(f"[VoiceEmbeddingAnalyzer] Hata (temporal embedding çıkarma): {str(e)}")
            return []
    
    def calculate_cognitive_load(self, embeddings: List[np.ndarray]) -> Dict:
        """
        Embedding'lerden bilişsel yük (cognitive load) çıkarımı yapar.
        Yüksek değişkenlik = yüksek bilişsel yük.
        
        Args:
            embeddings: Embedding vektörleri listesi
            
        Returns:
            Bilişsel yük metrikleri
        """
        if len(embeddings) < 2:
            return {
                'cognitive_load_score': 0.0,
                'prosodic_variability': 0.0,
                'temporal_stability': 0.0
            }
        
        # Embedding'leri numpy array'e çevir
        embedding_matrix = np.array(embeddings)  # Shape: (n_windows, hidden_size)
        
        # Prosodik değişkenlik (her boyut için varyans)
        variances = np.var(embedding_matrix, axis=0)
        mean_variance = np.mean(variances)
        
        # Bilişsel yük skoru (yüksek varyans = yüksek bilişsel yük)
        # Normalize et (0-1 arası)
        normalized_variance = min(1.0, mean_variance / 0.3)  # Threshold
        cognitive_load_score = normalized_variance
        
        # Temporal stability (ardışık embedding'ler arası benzerlik)
        similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)
        
        temporal_stability = np.mean(similarities) if similarities else 0.0
        
        return {
            'cognitive_load_score': float(cognitive_load_score),
            'prosodic_variability': float(mean_variance),
            'temporal_stability': float(temporal_stability)
        }


if __name__ == "__main__":
    # Test kodu
    analyzer = VoiceEmbeddingAnalyzer()
    print("VoiceEmbeddingAnalyzer modülü hazır.")
    print(f"Model: {analyzer.model_name}")
