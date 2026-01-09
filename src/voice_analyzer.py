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
    Librosa tabanlı özellikleri çıkarır ve sadece ham (raw) ses özelliklerini döner.
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
                'min_pitch': 0.0,
                'max_pitch': 0.0,
                'pitch_range': 0.0,
                'pitch_variability': 0.0
            }
        
        pitch_values = np.array(pitch_values)
        
        # Pitch özellikleri
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)
        min_pitch = np.min(pitch_values)
        max_pitch = np.max(pitch_values)
        pitch_range = max_pitch - min_pitch
        
        # Pitch değişkenliği (coefficient of variation)
        pitch_variability = std_pitch / mean_pitch if mean_pitch > 0 else 0.0
        
        return {
            'mean_pitch': float(mean_pitch),
            'std_pitch': float(std_pitch),
            'min_pitch': float(min_pitch),
            'max_pitch': float(max_pitch),
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
        (Şu an rapora eklenmiyor; gerekirse ileride kullanılabilir.)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length
        )
        
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
        # Duraklama sürelerini (saniye) hesapla
        pause_durations = []
        in_pause = False
        start_idx = 0
        for i, is_pause in enumerate(pauses):
            if is_pause and not in_pause:
                in_pause = True
                start_idx = i
            elif not is_pause and in_pause:
                length = i - start_idx
                duration_sec = (length * self.hop_length) / float(self.sample_rate)
                pause_durations.append(float(duration_sec))
                in_pause = False
        if in_pause:
            length = len(pauses) - start_idx
            duration_sec = (length * self.hop_length) / float(self.sample_rate)
            pause_durations.append(float(duration_sec))
        
        # Konuşma hızı (göreli indeks, yaklaşık konuşma hızı göstergesi)
        # Yüksek ZCR = daha hızlı konuşma (yaklaşık)
        speech_rate = np.mean(zcr) * 100  # Göreli ölçek (0-100 civarı)
        
        return {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'tempo_bpm': float(tempo),
            'pause_ratio': float(pause_ratio),
            'pause_durations': pause_durations,
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
        prosodic_features = self.extract_prosodic_features(audio)
        vad_features = self.detect_voice_activity(audio)

        # Spektral centroid (enerjinin frekans eksenindeki ağırlık merkezi)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate
        )[0]
        spectral_centroid_mean = float(np.mean(spectral_centroid))
        spectral_centroid_std = float(np.std(spectral_centroid))

        # RAW VOICE FEATURES (yorum içermeyen, ham özellikler)
        raw_voice_features = {
            "speech_rate": {
                "value": float(prosodic_features.get("speech_rate", 0.0)),
                "unit": "relative_index_0_100"
            },
            "rms_energy": {
                "mean": float(energy_features.get("mean_energy", 0.0)),
                "std": float(energy_features.get("std_energy", 0.0)),
                "min": float(energy_features.get("min_energy", 0.0)),
                "max": float(energy_features.get("max_energy", 0.0))
            },
            "pitch_f0": {
                "mean": float(pitch_features.get("mean_pitch", 0.0)),
                "std": float(pitch_features.get("std_pitch", 0.0)),
                "min": float(pitch_features.get("min_pitch", 0.0)),
                "max": float(pitch_features.get("max_pitch", 0.0))
            },
            "pause_durations": prosodic_features.get("pause_durations", []),
            "silence_ratio": float(prosodic_features.get("pause_ratio", 0.0)),
            "spectral_centroid": {
                "mean": spectral_centroid_mean,
                "std": spectral_centroid_std
            },
            "zero_crossing_rate": {
                "mean": float(prosodic_features.get("zero_crossing_rate_mean", 0.0)),
                "std": float(prosodic_features.get("zero_crossing_rate_std", 0.0))
            },
            "duration_seconds": float(vad_features.get("total_duration_seconds", 0.0))
        }

        return {
            "raw_voice_features": raw_voice_features
        }


if __name__ == "__main__":
    # Test kodu
    analyzer = VoiceAnalyzer()
    print("VoiceAnalyzer modülü hazır.")
    print("\nKullanım:")
    print("  analyzer = VoiceAnalyzer()")
    print("  result = analyzer.analyze_audio('audio.wav')")
    print("  print(result['summary'])")
