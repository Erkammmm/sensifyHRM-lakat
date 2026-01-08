"""
Temporal Transformer Modülü
Multi-modal embedding'leri (gaze, head pose, face embedding, voice embedding) 
attention mekanizması ile birleştirerek behavioral embedding üretir.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from collections import deque


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer modeli.
    Multi-modal embedding'leri attention mekanizması ile birleştirir.
    """
    
    def __init__(self, 
                 face_embedding_dim: int = 512,
                 voice_embedding_dim: int = 768,  # wav2vec2-base hidden size
                 gaze_feature_dim: int = 4,  # eye_contact, gaze_angle, pitch, yaw
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Temporal Transformer'ı başlatır.
        
        Args:
            face_embedding_dim: Yüz embedding boyutu (Facenet512 = 512)
            voice_embedding_dim: Ses embedding boyutu (wav2vec2-base = 768)
            gaze_feature_dim: Gaze/head pose feature boyutu
            hidden_dim: Transformer hidden dimension
            num_heads: Attention head sayısı
            num_layers: Transformer layer sayısı
            dropout: Dropout oranı
        """
        super(TemporalTransformer, self).__init__()
        
        self.face_embedding_dim = face_embedding_dim
        self.voice_embedding_dim = voice_embedding_dim
        self.gaze_feature_dim = gaze_feature_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layers (her modalite için)
        self.face_projection = nn.Linear(face_embedding_dim, hidden_dim)
        self.voice_projection = nn.Linear(voice_embedding_dim, hidden_dim)
        self.gaze_projection = nn.Linear(gaze_feature_dim, hidden_dim)
        
        # Positional encoding (basit sinüs/cosinüs)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                face_embeddings: Optional[torch.Tensor] = None,
                voice_embeddings: Optional[torch.Tensor] = None,
                gaze_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            face_embeddings: (batch_size, seq_len, face_embedding_dim)
            voice_embeddings: (batch_size, seq_len, voice_embedding_dim)
            gaze_features: (batch_size, seq_len, gaze_feature_dim)
            
        Returns:
            Behavioral embedding: (batch_size, seq_len, hidden_dim)
        """
        batch_size = None
        seq_len = None
        
        # Batch size ve sequence length'i belirle
        if face_embeddings is not None:
            batch_size, seq_len = face_embeddings.shape[:2]
        elif voice_embeddings is not None:
            batch_size, seq_len = voice_embeddings.shape[:2]
        elif gaze_features is not None:
            batch_size, seq_len = gaze_features.shape[:2]
        else:
            raise ValueError("En az bir modalite sağlanmalı")
        
        # Her modaliteyi project et ve birleştir
        projected_features = []
        
        if face_embeddings is not None:
            face_proj = self.face_projection(face_embeddings)
            projected_features.append(face_proj)
        
        if voice_embeddings is not None:
            voice_proj = self.voice_projection(voice_embeddings)
            projected_features.append(voice_proj)
        
        if gaze_features is not None:
            gaze_proj = self.gaze_projection(gaze_features)
            projected_features.append(gaze_proj)
        
        # Feature'ları topla (multi-modal fusion)
        combined_features = torch.stack(projected_features, dim=2)  # (batch, seq, num_modalities, hidden)
        combined_features = torch.mean(combined_features, dim=2)  # (batch, seq, hidden) - average pooling
        
        # Positional encoding ekle
        combined_features = self.pos_encoding(combined_features)
        
        # Transformer encoder
        output = self.transformer_encoder(combined_features)
        
        # Output projection
        output = self.output_projection(output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Sinüs/cosinüs positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BehavioralEmbeddingExtractor:
    """
    Behavioral embedding çıkaran wrapper sınıf.
    Temporal Transformer kullanarak multi-modal embedding'leri birleştirir.
    """
    
    def __init__(self, 
                 face_embedding_dim: int = 512,
                 voice_embedding_dim: int = 768,
                 gaze_feature_dim: int = 4,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 2):
        """
        Behavioral embedding extractor'ı başlatır.
        
        Args:
            face_embedding_dim: Yüz embedding boyutu
            voice_embedding_dim: Ses embedding boyutu
            gaze_feature_dim: Gaze feature boyutu
            hidden_dim: Transformer hidden dimension
            num_heads: Attention head sayısı
            num_layers: Transformer layer sayısı
        """
        self.model = TemporalTransformer(
            face_embedding_dim=face_embedding_dim,
            voice_embedding_dim=voice_embedding_dim,
            gaze_feature_dim=gaze_feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        self.model.eval()  # Evaluation mode
    
    def extract_behavioral_embeddings(self,
                                     face_embeddings: Optional[List[np.ndarray]] = None,
                                     voice_embeddings: Optional[List[np.ndarray]] = None,
                                     gaze_features: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Multi-modal embedding'lerden behavioral embedding çıkarır.
        
        Args:
            face_embeddings: Yüz embedding'leri listesi (her eleman 512 boyutlu)
            voice_embeddings: Ses embedding'leri listesi (her eleman 768 boyutlu)
            gaze_features: Gaze feature'ları listesi (her eleman [eye_contact, gaze_angle, pitch, yaw])
            
        Returns:
            Behavioral embedding'ler: (seq_len, hidden_dim)
        """
        # Sequence length'i belirle
        seq_len = 0
        if face_embeddings:
            seq_len = len(face_embeddings)
        elif voice_embeddings:
            seq_len = len(voice_embeddings)
        elif gaze_features:
            seq_len = len(gaze_features)
        else:
            raise ValueError("En az bir modalite sağlanmalı")
        
        # Tensor'lara çevir
        face_tensor = None
        voice_tensor = None
        gaze_tensor = None
        
        if face_embeddings:
            # None'ları sıfır vektörü ile değiştir
            face_arrays = [emb if emb is not None else np.zeros(512) for emb in face_embeddings]
            face_tensor = torch.FloatTensor(np.array(face_arrays)).unsqueeze(0)  # (1, seq_len, 512)
        
        if voice_embeddings:
            voice_arrays = [emb if emb is not None else np.zeros(768) for emb in voice_embeddings]
            voice_tensor = torch.FloatTensor(np.array(voice_arrays)).unsqueeze(0)  # (1, seq_len, 768)
        
        if gaze_features:
            gaze_arrays = [feat if feat is not None else np.zeros(4) for feat in gaze_features]
            gaze_tensor = torch.FloatTensor(np.array(gaze_arrays)).unsqueeze(0)  # (1, seq_len, 4)
        
        # Model inference
        with torch.no_grad():
            behavioral_embeddings = self.model(
                face_embeddings=face_tensor,
                voice_embeddings=voice_tensor,
                gaze_features=gaze_tensor
            )
        
        # Numpy'ye çevir ve batch dimension'ı kaldır
        behavioral_embeddings = behavioral_embeddings.squeeze(0).numpy()  # (seq_len, hidden_dim)
        
        return behavioral_embeddings
    
    def calculate_behavioral_score(self, behavioral_embeddings: np.ndarray) -> Dict:
        """
        Behavioral embedding'lerden genel skor hesaplar.
        
        Args:
            behavioral_embeddings: Behavioral embedding'ler (seq_len, hidden_dim)
            
        Returns:
            Behavioral skor metrikleri
        """
        if len(behavioral_embeddings) == 0:
            return {
                'behavioral_consistency': 0.0,
                'temporal_stability': 0.0,
                'engagement_score': 0.0
            }
        
        # Temporal stability (ardışık embedding'ler arası benzerlik)
        similarities = []
        for i in range(len(behavioral_embeddings) - 1):
            emb1 = behavioral_embeddings[i]
            emb2 = behavioral_embeddings[i + 1]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)
        
        temporal_stability = np.mean(similarities) if similarities else 0.0
        
        # Behavioral consistency (varyans)
        variances = np.var(behavioral_embeddings, axis=0)
        mean_variance = np.mean(variances)
        behavioral_consistency = 1.0 - min(1.0, mean_variance / 0.5)  # Normalize
        
        # Engagement score (basit heuristik - embedding norm'u)
        embedding_norms = np.linalg.norm(behavioral_embeddings, axis=1)
        engagement_score = float(np.mean(embedding_norms) / 10.0)  # Normalize
        engagement_score = min(1.0, engagement_score)
        
        return {
            'behavioral_consistency': float(behavioral_consistency),
            'temporal_stability': float(temporal_stability),
            'engagement_score': float(engagement_score)
        }


if __name__ == "__main__":
    # Test kodu
    extractor = BehavioralEmbeddingExtractor()
    print("TemporalTransformer modülü hazır.")
    print("Behavioral embedding extractor hazır.")
