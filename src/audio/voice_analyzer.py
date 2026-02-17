"""
Ses Analizi Modülü (Phase-2)
DistilHuBERT yaklaşımına uygun şekilde ham ses özelliklerini çıkarır.
"""

import numpy as np
import librosa
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")


class VoiceAnalyzer:
    """
    Ses analizi yapan sınıf.
    Librosa tabanlı ham özellikleri çıkarır ve sadece yorumlanabilir raw çıktılar döner.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 2048,
        hop_length: int = 512,
        top_db: int = 20,
        pre_emphasis: float = 0.97,
        max_norm: float = 0.95,
        window_seconds: int = 8,
        overlap_seconds: int = 0,
    ):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.top_db = top_db
        self.pre_emphasis = pre_emphasis
        self.max_norm = max_norm
        self.window_seconds = window_seconds
        self.overlap_seconds = overlap_seconds

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Ses dosyasını 16 kHz olarak yükler."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Ses dosyası yüklenemedi: {str(e)}")

    def _trim_silence(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Sessiz giriş/çıkışları temizler, trim bilgilerini döndürür."""
        if audio.size == 0:
            return audio, {"trim_start_sec": 0.0, "trim_end_sec": 0.0}
        trimmed, (start, end) = librosa.effects.trim(audio, top_db=self.top_db)
        trim_start = float(start) / float(self.sample_rate)
        trim_end = float(len(audio) - end) / float(self.sample_rate)
        return trimmed, {"trim_start_sec": trim_start, "trim_end_sec": trim_end}

    def _pre_emphasize(self, audio: np.ndarray) -> np.ndarray:
        """Pre-emphasis filtre uygular: H(z) = 1 - 0.97 z^-1."""
        if audio.size == 0:
            return audio
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Genliği 0.95 tepe değere normalize eder."""
        if audio.size == 0:
            return audio
        max_val = float(np.max(np.abs(audio)))
        if max_val <= 0:
            return audio
        return audio * (self.max_norm / max_val)

    def _downsample_series(self, values: np.ndarray, max_points: int = 4000) -> np.ndarray:
        # Removed: function inlined in _extract_waveform. Kept for backward compatibility.
        if values.size <= max_points:
            return values
        idx = np.linspace(0, values.size - 1, max_points).astype(int)
        return values[idx]

    def _extract_waveform(self, audio: np.ndarray) -> Dict:
        """Ham waveform serisini (downsample edilmiş) çıkarır."""
        if audio.size == 0:
            return {"times": [], "values": []}
        # inline downsample logic (avoid extra helper indirection)
        values = audio
        max_points = 4000
        if values.size > max_points:
            idx = np.linspace(0, values.size - 1, max_points).astype(int)
            values = values[idx]
        times = np.linspace(0, len(audio) / float(self.sample_rate), num=values.size, endpoint=False)
        return {"times": times.tolist(), "values": values.tolist()}

    def _compute_vad_metrics(self, audio: np.ndarray) -> Dict:
        """VAD metriklerini (sessizlik/speech oranları) hesaplar."""
        total_duration = float(len(audio)) / float(self.sample_rate) if audio.size else 0.0
        if audio.size == 0:
            return {
                "total_speech_seconds": 0.0,
                "total_silence_seconds": 0.0,
                "average_silence_seconds": 0.0,
                "speech_silence_ratio": 0.0,
                "response_pre_silence_seconds": [],
                "speech_segments": [],
            }

        intervals = librosa.effects.split(audio, top_db=self.top_db)
        if intervals.size == 0:
            return {
                "total_speech_seconds": 0.0,
                "total_silence_seconds": total_duration,
                "average_silence_seconds": total_duration,
                "speech_silence_ratio": 0.0,
                "response_pre_silence_seconds": [total_duration] if total_duration > 0 else [],
                "speech_segments": [],
            }

        speech_segments = [(int(s), int(e)) for s, e in intervals]
        speech_seconds = sum((e - s) for s, e in speech_segments) / float(self.sample_rate)
        silence_seconds = max(0.0, total_duration - speech_seconds)

        silence_durations = []
        response_pre_silences = []
        prev_end = 0
        for start, end in speech_segments:
            silence = max(0.0, (start - prev_end) / float(self.sample_rate))
            response_pre_silences.append(silence)
            silence_durations.append(silence)
            prev_end = end
        tail_silence = max(0.0, (len(audio) - prev_end) / float(self.sample_rate))
        silence_durations.append(tail_silence)

        avg_silence = float(np.mean(silence_durations)) if silence_durations else 0.0
        ratio = speech_seconds / silence_seconds if silence_seconds > 0 else 0.0

        return {
            "total_speech_seconds": float(speech_seconds),
            "total_silence_seconds": float(silence_seconds),
            "average_silence_seconds": float(avg_silence),
            "speech_silence_ratio": float(ratio),
            "response_pre_silence_seconds": [float(v) for v in response_pre_silences],
            "speech_segments": [
                {"start": float(s) / self.sample_rate, "end": float(e) / self.sample_rate}
                for s, e in speech_segments
            ],
        }

    def _extract_rms(self, audio: np.ndarray) -> Dict:
        """RMS enerji serisi ve özet istatistiklerini çıkarır."""
        if audio.size == 0:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "series": {"times": [], "values": []},
            }
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
        times = librosa.frames_to_time(
            np.arange(len(rms)),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        return {
            "mean": float(np.mean(rms)),
            "variance": float(np.var(rms)),
            "series": {"times": times.tolist(), "values": rms.tolist()},
        }

    def _extract_pitch(self, audio: np.ndarray) -> Dict:
        """Pitch (F0) serisi ve istatistiklerini çıkarır."""
        if audio.size == 0:
            return {
                "mean": 0.0,
                "variability": 0.0,
                "jump_count": 0,
                "series": {"times": [], "values": []},
            }

        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=400,
            sr=self.sample_rate,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )
        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        f0_series = np.where(np.isnan(f0), 0.0, f0)
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[np.isfinite(f0)]
        voiced_f0 = voiced_f0[np.isfinite(voiced_f0)] if voiced_f0 is not None else np.array([])

        pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
        pitch_std = float(np.std(voiced_f0)) if voiced_f0.size else 0.0

        # Pitch sıçramaları: ardışık voiced frame'lerde 50 Hz üzeri değişim
        jump_threshold = 50.0
        jump_count = 0
        if voiced_flag is not None and len(f0) > 1:
            prev = None
            for val, voiced in zip(f0, voiced_flag):
                if not voiced or not np.isfinite(val):
                    prev = None
                    continue
                if prev is not None and abs(val - prev) >= jump_threshold:
                    jump_count += 1
                prev = float(val)

        return {
            "mean": pitch_mean,
            "variability": pitch_std,
            "jump_count": int(jump_count),
            "series": {"times": times.tolist(), "values": f0_series.tolist()},
        }

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> Dict:
        """Tek bir Mel spectrogram (ısı haritası) üretir."""
        if audio.size == 0:
            return {"times": [], "frequencies": [], "values": []}
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            hop_length=self.hop_length,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        times = librosa.frames_to_time(
            np.arange(mel_db.shape[1]),
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )
        freqs = librosa.mel_frequencies(n_mels=mel_db.shape[0], fmin=0, fmax=self.sample_rate / 2.0)

        # Downsample time axis for plotting
        max_time_points = 400
        if mel_db.shape[1] > max_time_points:
            idx = np.linspace(0, mel_db.shape[1] - 1, max_time_points).astype(int)
            mel_db = mel_db[:, idx]
            times = times[idx]

        return {
            "times": times.tolist(),
            "frequencies": freqs.tolist(),
            "values": mel_db.tolist(),
        }

    def _window_audio(self, audio: np.ndarray) -> List[Dict]:
        """8 saniyelik pencereler üzerinden ham özellikleri çıkarır."""
        if audio.size == 0:
            return []

        step = self.window_seconds - self.overlap_seconds
        step = step if step > 0 else self.window_seconds
        window_samples = int(self.window_seconds * self.sample_rate)
        step_samples = int(step * self.sample_rate)
        total_samples = len(audio)

        windows = []
        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            segment = audio[start:end]
            if segment.size == 0:
                break
            start_sec = float(start) / float(self.sample_rate)
            end_sec = float(end) / float(self.sample_rate)
            windows.append(
                {
                    "window_start": start_sec,
                    "window_end": end_sec,
                    "energy_rms": self._extract_rms(segment),
                    "pitch_f0": self._extract_pitch(segment),
                    "speech_silence": self._compute_vad_metrics(segment),
                }
            )
            start += step_samples
        return windows

    def analyze_audio(self, audio_path: str) -> Dict:
        """Ses dosyasını DistilHuBERT yaklaşımına uygun şekilde analiz eder."""
        audio, _ = self.load_audio(audio_path)

        # 1) Zorunlu yeniden örnekleme load_audio ile yapılır
        original_duration = float(len(audio)) / float(self.sample_rate) if audio.size else 0.0

        # 2) Sessizlik temizleme (trim) + sessizlik metrikleri
        vad_metrics = self._compute_vad_metrics(audio)
        audio_trimmed, trim_info = self._trim_silence(audio)

        # 3) Pre-emphasis
        audio_emph = self._pre_emphasize(audio_trimmed)

        # 4) Normalizasyon
        audio_norm = self._normalize(audio_emph)

        # Ham özellikler
        waveform_series = self._extract_waveform(audio)
        rms_features = self._extract_rms(audio_norm)
        pitch_features = self._extract_pitch(audio_norm)
        mel_spectrogram = self._extract_mel_spectrogram(audio_norm)
        windowed = self._window_audio(audio_norm)

        raw_voice_features = {
            "preprocessing": {
                "sample_rate": int(self.sample_rate),
                "top_db": int(self.top_db),
                "pre_emphasis": float(self.pre_emphasis),
                "max_norm": float(self.max_norm),
                "original_duration_seconds": float(original_duration),
                "trim_start_seconds": float(trim_info.get("trim_start_sec", 0.0)),
                "trim_end_seconds": float(trim_info.get("trim_end_sec", 0.0)),
                "processed_duration_seconds": float(len(audio_norm)) / float(self.sample_rate) if audio_norm.size else 0.0,
            },
            "waveform": waveform_series,
            "energy_rms": rms_features,
            "pitch_f0": pitch_features,
            "speech_silence": vad_metrics,
            "mel_spectrogram": mel_spectrogram,
            "windowed_features": windowed,
            "duration_seconds": float(len(audio_norm)) / float(self.sample_rate) if audio_norm.size else 0.0,
        }

        return {"raw_voice_features": raw_voice_features}


if __name__ == "__main__":
    analyzer = VoiceAnalyzer()
    print("VoiceAnalyzer modülü hazır.")
    print("\nKullanım:")
    print("  analyzer = VoiceAnalyzer()")
    print("  result = analyzer.analyze_audio('audio.wav')")
