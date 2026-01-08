"""
Ses Analizi Modülü
Ses dosyalarından stres, güven ve konuşma özelliklerini analiz eder.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class VoiceAnalyzer:
    """
    Ses analizi yapan sınıf.
    Pitch, energy, MFCC ve prosodic features kullanarak stres ve güven skorları hesaplar.
    """
    
    def __init__(self, sample_rate: int = 16000, frame_length: int = 2048, hop_length: int = 512):
        """
        Ses analizcisini başlatır.
        
        Args:
            sample_rate: Ses örnekleme hızı (Hz)
            frame_length: Frame uzunluğu (FFT için)
            hop_length: Frame atlama uzunluğu
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Ses dosyasını yükler ve normalize eder.
        
        Args:
            audio_path: Ses dosyasının yolu
            
        Returns:
            (audio_array, sample_rate) tuple
        """
        try:
            # Librosa ile ses yükleme (otomatik resample)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Ses dosyası yüklenemedi: {str(e)}")
    
    def extract_pitch(self, audio: np.ndarray) -> Dict:
        """
        Pitch (perde) özelliklerini çıkarır.
        Yüksek pitch değişkenliği stres göstergesi olabilir.
        
        Args:
            audio: Ses sinyali
            
        Returns:
            Pitch özellikleri dictionary'si
        """
        # Pitch tespiti (pyin algoritması - daha doğru)
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            fmin=50,  # Minimum frekans (Hz)
            fmax=400  # Maksimum frekans (Hz)
        )
        
        # Pitch değerlerini çıkar (0 olmayan değerler)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) == 0:
            return {
                'mean_pitch': 0.0,
                'std_pitch': 0.0,
                'pitch_range': 0.0,
                'pitch_variability': 0.0
            }
        
        pitch_values = np.array(pitch_values)
        
        # Pitch özellikleri
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        pitch_range = np.max(pitch_values) - np.min(pitch_values)
        
        # Pitch değişkenliği (coefficient of variation)
        pitch_variability = std_pitch / mean_pitch if mean_pitch > 0 else 0.0
        
        return {
            'mean_pitch': float(mean_pitch),
            'std_pitch': float(std_pitch),
            'pitch_range': float(pitch_range),
            'pitch_variability': float(pitch_variability),
            'pitch_values': pitch_values.tolist()[:100]  # İlk 100 değer (örnek)
        }
    
    def extract_energy(self, audio: np.ndarray) -> Dict:
        """
        Enerji (ses seviyesi) özelliklerini çıkarır.
        
        Args:
            audio: Ses sinyali
            
        Returns:
            Enerji özellikleri dictionary'si
        """
        # RMS (Root Mean Square) enerji
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # Enerji özellikleri
        mean_energy = np.mean(rms)
        std_energy = np.std(rms)
        max_energy = np.max(rms)
        min_energy = np.min(rms)
        
        # Enerji değişkenliği
        energy_variability = std_energy / mean_energy if mean_energy > 0 else 0.0
        
        return {
            'mean_energy': float(mean_energy),
            'std_energy': float(std_energy),
            'max_energy': float(max_energy),
            'min_energy': float(min_energy),
            'energy_variability': float(energy_variability)
        }
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict:
        """
        MFCC (Mel-Frequency Cepstral Coefficients) özelliklerini çıkarır.
        Ses kalitesi ve ton özelliklerini temsil eder.
        
        Args:
            audio: Ses sinyali
            n_mfcc: MFCC katsayı sayısı
            
        Returns:
            MFCC özellikleri dictionary'si
        """
        # MFCC çıkarma
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length
        )
        
        # Her MFCC katsayısı için ortalama ve standart sapma
        mfcc_features = {}
        for i in range(n_mfcc):
            mfcc_features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
            mfcc_features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
        
        return mfcc_features
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict:
        """
        Prosodic (prosodik) özelliklerini çıkarır.
        Konuşma ritmi, tempo ve duraklamalar.
        
        Args:
            audio: Ses sinyali
            
        Returns:
            Prosodic özellikleri dictionary'si
        """
        # Zero crossing rate (ses-sessizlik geçişleri)
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # Tempo (BPM - beats per minute)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        
        # Duraklama tespiti (düşük enerji bölgeleri)
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        energy_threshold = np.percentile(rms, 20)  # En düşük %20'lik dilim
        pauses = rms < energy_threshold
        pause_ratio = np.sum(pauses) / len(pauses)
        
        # Konuşma hızı (saniyede kelime tahmini - basit hesaplama)
        # Yüksek ZCR = daha hızlı konuşma (yaklaşık)
        speech_rate = np.mean(zcr) * 100  # Normalize edilmiş hız
        
        return {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'tempo_bpm': float(tempo),
            'pause_ratio': float(pause_ratio),
            'speech_rate': float(speech_rate)
        }
    
    def detect_voice_activity(self, audio: np.ndarray, threshold: float = 0.01) -> Dict:
        """
        Voice Activity Detection (VAD) - Ses aktivitesi tespiti.
        Konuşma ve sessizlik bölgelerini ayırır.
        
        Args:
            audio: Ses sinyali
            threshold: Enerji eşiği
            
        Returns:
            VAD sonuçları
        """
        # RMS enerji
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        
        # Ses aktivitesi (threshold üzeri)
        voice_frames = rms > threshold
        voice_ratio = np.sum(voice_frames) / len(voice_frames)
        
        # Ses segmentleri (sürekli konuşma bölgeleri)
        voice_segments = []
        in_voice = False
        segment_start = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_voice:
                segment_start = i
                in_voice = True
            elif not is_voice and in_voice:
                segment_length = i - segment_start
                voice_segments.append(segment_length)
                in_voice = False
        
        if in_voice:
            segment_length = len(voice_frames) - segment_start
            voice_segments.append(segment_length)
        
        avg_segment_length = np.mean(voice_segments) if voice_segments else 0.0
        
        return {
            'voice_ratio': float(voice_ratio),
            'num_voice_segments': len(voice_segments),
            'avg_segment_length': float(avg_segment_length),
            'total_duration_seconds': len(audio) / self.sample_rate
        }
    
    def calculate_stress_score(self, features: Dict) -> float:
        """
        Stres skoru hesaplar (0-1 arası).
        Yüksek pitch değişkenliği, yüksek enerji değişkenliği = stres göstergesi.
        
        Args:
            features: Tüm ses özellikleri
            
        Returns:
            Stres skoru (0-1, yüksek = daha stresli)
        """
        # Pitch değişkenliği (yüksek = stres)
        pitch_var = features.get('pitch', {}).get('pitch_variability', 0.0)
        pitch_stress = min(1.0, pitch_var * 2.0)  # Normalize et
        
        # Enerji değişkenliği (yüksek = stres)
        energy_var = features.get('energy', {}).get('energy_variability', 0.0)
        energy_stress = min(1.0, energy_var * 3.0)  # Normalize et
        
        # Duraklama oranı (yüksek = stres/tereddüt)
        pause_ratio = features.get('prosodic', {}).get('pause_ratio', 0.0)
        pause_stress = pause_ratio * 1.5  # Normalize et
        
        # Ağırlıklı ortalama
        stress_score = (
            pitch_stress * 0.4 +
            energy_stress * 0.3 +
            pause_stress * 0.3
        )
        
        return float(np.clip(stress_score, 0.0, 1.0))
    
    def calculate_confidence_score(self, features: Dict) -> float:
        """
        Güven skoru hesaplar (0-1 arası).
        Düşük pitch değişkenliği, düşük duraklama = güven göstergesi.
        
        Args:
            features: Tüm ses özellikleri
            
        Returns:
            Güven skoru (0-1, yüksek = daha güvenli)
        """
        # Pitch stabilitesi (düşük değişkenlik = güven)
        pitch_var = features.get('pitch', {}).get('pitch_variability', 0.0)
        pitch_confidence = max(0.0, 1.0 - (pitch_var * 2.0))
        
        # Enerji stabilitesi
        energy_var = features.get('energy', {}).get('energy_variability', 0.0)
        energy_confidence = max(0.0, 1.0 - (energy_var * 2.0))
        
        # Düşük duraklama = güven
        pause_ratio = features.get('prosodic', {}).get('pause_ratio', 0.0)
        pause_confidence = max(0.0, 1.0 - (pause_ratio * 2.0))
        
        # Konuşma hızı (orta hız = güven, çok hızlı/yavaş = güvensizlik)
        speech_rate = features.get('prosodic', {}).get('speech_rate', 0.0)
        # Optimal hız: 50-80 arası (normalize edilmiş)
        if 50 <= speech_rate <= 80:
            rate_confidence = 1.0
        else:
            rate_confidence = max(0.0, 1.0 - abs(speech_rate - 65) / 65)
        
        # Ağırlıklı ortalama
        confidence_score = (
            pitch_confidence * 0.3 +
            energy_confidence * 0.2 +
            pause_confidence * 0.3 +
            rate_confidence * 0.2
        )
        
        return float(np.clip(confidence_score, 0.0, 1.0))
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Ses dosyasını tam olarak analiz eder.
        
        Args:
            audio_path: Ses dosyasının yolu
            
        Returns:
            Tüm analiz sonuçları
        """
        # Ses dosyasını yükle
        audio, sr = self.load_audio(audio_path)
        
        # Tüm özellikleri çıkar
        pitch_features = self.extract_pitch(audio)
        energy_features = self.extract_energy(audio)
        mfcc_features = self.extract_mfcc(audio)
        prosodic_features = self.extract_prosodic_features(audio)
        vad_features = self.detect_voice_activity(audio)
        
        # Özellikleri birleştir
        all_features = {
            'pitch': pitch_features,
            'energy': energy_features,
            'mfcc': mfcc_features,
            'prosodic': prosodic_features,
            'vad': vad_features
        }
        
        # Stres ve güven skorlarını hesapla
        stress_score = self.calculate_stress_score(all_features)
        confidence_score = self.calculate_confidence_score(all_features)
        
        # Özet rapor
        summary = {
            'stress_level': stress_score,
            'confidence_score': confidence_score,
            'speech_rate': prosodic_features.get('speech_rate', 0.0),
            'pause_ratio': prosodic_features.get('pause_ratio', 0.0),
            'mean_pitch': pitch_features.get('mean_pitch', 0.0),
            'pitch_variability': pitch_features.get('pitch_variability', 0.0),
            'voice_activity_ratio': vad_features.get('voice_ratio', 0.0),
            'duration_seconds': vad_features.get('total_duration_seconds', 0.0)
        }
        
        return {
            'summary': summary,
            'detailed_features': all_features
        }


if __name__ == "__main__":
    # Test kodu
    analyzer = VoiceAnalyzer()
    print("VoiceAnalyzer modülü hazır.")
    print("\nKullanım:")
    print("  analyzer = VoiceAnalyzer()")
    print("  result = analyzer.analyze_audio('audio.wav')")
    print("  print(result['summary'])")
