"""
Ana Pipeline Modülü
Tüm analiz adımlarını koordine eder:
  1) Metin Analizi (Whisper + Türkçe BERT)
  2) Ses Duygu Analizi (HuBERT SER)
  3) Yüz Analizi (MediaPipe blendshape)
  4) Ham Ses Özellikleri (Librosa)
  5) Tutarsızlık Analizi
"""

import os
import time
import uuid
from typing import Dict, Optional
from collections import Counter

from .text_analyzer import TextAnalyzer
from .audio_analyzer import AudioAnalyzer
from .face_analyzer import FaceAnalyzer
from .voice_analyzer import VoiceAnalyzer
from .video_processor import VideoProcessor
from .audio_signal_fusion import AudioSignalFusion
from .contextual_aggregator import build_segment_signal_packages
from .thought_unit_merger import merge_into_thought_units


class InterviewAnalysisPipeline:
    """
    Mülakat analiz pipeline'ı.
    Video dosyası alır → 4 modül ile analiz eder → birleşik rapor döndürür.
    """

    def __init__(self, phase3_enabled: bool = True):
        print("=" * 60)
        print("[Pipeline] SensifyHR Mülakat Analiz Sistemi Başlatılıyor...")
        print("=" * 60)

        # v3 default (ürün modu)
        self.phase3_enabled = bool(phase3_enabled)

        self.video_processor = VideoProcessor()
        self.text_analyzer = TextAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.face_analyzer = FaceAnalyzer()
        self.voice_analyzer = VoiceAnalyzer()
        # FAZ-3 audio signal (lazy init değil; model init süresi yüksek olabilir)
        self._audio_signal_fusion: Optional[AudioSignalFusion] = None

        print("[Pipeline] Tüm modüller hazır!\n")

    def _get_audio_signal_fusion(self) -> AudioSignalFusion:
        if self._audio_signal_fusion is None:
            self._audio_signal_fusion = AudioSignalFusion()
        return self._audio_signal_fusion

    # ------------------------------------------------------------------
    # Tutarsızlık Analizi
    # ------------------------------------------------------------------
    @staticmethod
    def analyze_consistency(text_sentiment, face_emotion):
        """
        Metin duygusunu yüz ifadesiyle karşılaştırarak tutarsızlık tespit eder.

        Args:
            text_sentiment: "positive" veya "negative"
            face_emotion: Yüz analizi duygu etiketi (Türkçe)

        Returns:
            str: "Tutarli", "ŞÜPHELİ (...)" veya "Nötr/Belirsiz"
        """
        positive_face = {"Mutlu", "Saskin"}
        negative_face = {"Uzgun", "Korku", "Tiksinti", "Ofkeli", "Stresli"}

        if text_sentiment == "positive":
            if face_emotion in positive_face:
                return "Tutarli"
            if face_emotion in negative_face:
                return "ŞÜPHELİ (Pozitif Söz / Negatif Yüz)"

        elif text_sentiment == "negative":
            if face_emotion in negative_face:
                return "Tutarli"
            if face_emotion == "Mutlu":
                return "ŞÜPHELİ (Negatif Söz / Gülen Yüz - Sarkazm?)"

        return "Nötr/Belirsiz"

    def _find_anomalies(self, text_data, face_timeline):
        """
        Metin ve yüz verilerini zaman bazında eşleştirip tutarsızlıkları bulur.

        Returns:
            list[dict]: Bulunan anomaliler
        """
        anomalies = []
        if not text_data or not face_timeline:
            return anomalies

        for segment in text_data:
            seg_start = segment["start"]
            seg_end = segment["end"]

            # Bu zaman aralığındaki yüz verisini bul
            face_in_range = [
                f for f in face_timeline
                if seg_start <= f["timestamp"] <= seg_end
            ]
            if not face_in_range:
                continue

            # En sık yüz duygusunu bul
            face_emotions = [f["emotion"] for f in face_in_range]
            dominant_face = Counter(face_emotions).most_common(1)[0][0]

            consistency = self.analyze_consistency(
                segment["sentiment"], dominant_face
            )

            if "ŞÜPHELİ" in consistency:
                anomalies.append({
                    "time_range": f"{seg_start:.1f}s - {seg_end:.1f}s",
                    "text": segment["text"],
                    "text_sentiment": segment["sentiment"],
                    "face_emotion": dominant_face,
                    "result": consistency,
                })

        return anomalies

    # ------------------------------------------------------------------
    # Ana Analiz
    # ------------------------------------------------------------------
    def process_interview(
        self,
        video_path: str,
        interview_id: Optional[str] = None,
        phase3_enabled: Optional[bool] = None,
    ) -> Dict:
        """
        Tam mülakat analizi yapar.

        Args:
            video_path: Video dosya yolu
            interview_id: Opsiyonel mülakat ID

        Returns:
            dict: Tüm analiz sonuçlarını içeren rapor
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

        if interview_id is None:
            interview_id = str(uuid.uuid4())

        # Çağrı bazlı override (None ise pipeline init flag'i kullan)
        if phase3_enabled is None:
            phase3_enabled = self.phase3_enabled

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[Pipeline] Analiz Başlıyor: {interview_id}")
        print(f"[Pipeline] Video: {video_path}")
        print(f"[Pipeline] Phase3 Enabled: {bool(phase3_enabled)}")
        print(f"{'='*60}\n")

        # 0) Video bilgileri
        print("[Adım 0/5] Video bilgileri alınıyor...")
        video_info = self.video_processor.get_video_info(video_path)

        # 1) Ses çıkarma + Ham ses özellikleri (librosa)
        print("[Adım 1/5] Ses çıkarılıyor ve ham özellikler hesaplanıyor...")
        audio_path = self.video_processor.extract_audio(video_path)
        voice_analysis = self.voice_analyzer.analyze_audio(audio_path)

        # 2) Metin analizi (STT + (v2) sentiment)
        # Not: MP4 decode bağımlılıklarını azaltmak için STT'yi çıkarılmış WAV üzerinden çalıştırıyoruz.
        print("[Adım 2/5] Metin analizi yapılıyor (STT)...")
        text_data = self.text_analyzer.process_video(audio_path, phase3_enabled=bool(phase3_enabled))
        text_summary = self.text_analyzer.get_summary(text_data)

        thought_units = []
        if phase3_enabled:
            thought_units = merge_into_thought_units(text_data)

        # 3) Ses analizi
        audio_emotion_data = []
        audio_emotion_summary = {}
        audio_signal_data = []
        audio_signal_summary = {}

        if phase3_enabled:
            print("[Adım 3/5] Ses sinyal analizi yapılıyor (HuBERT SER projection + librosa)...")
            fusion = self._get_audio_signal_fusion()
            audio_signal_data = fusion.process_audio(audio_path)
            audio_signal_summary = fusion.get_summary(audio_signal_data)
        else:
            print("[Adım 3/5] Ses duygu analizi yapılıyor (HuBERT SER)...")
            audio_emotion_data = self.audio_analyzer.process_video(video_path)
            audio_emotion_summary = self.audio_analyzer.get_summary(audio_emotion_data)

        # 4) Görsel analiz (MediaPipe)
        if phase3_enabled:
            print("[Adım 4/5] Görsel sinyal analizi yapılıyor (MediaPipe)...")
        else:
            print("[Adım 4/5] Yüz analizi yapılıyor (MediaPipe)...")
        face_timeline, face_summary = self.face_analyzer.process_video(
            video_path, phase3_enabled=bool(phase3_enabled)
        )

        # FAZ-3: Okuma şüphesi olaylarında "konuşma ile eşzamanlı" flag'i ekle (VAD speech segments)
        if phase3_enabled:
            try:
                ss = (voice_analysis or {}).get("raw_voice_features", {}).get("speech_silence", {}) or {}
                speech_segments = ss.get("speech_segments", []) or []
                events = (face_summary or {}).get("reading_suspicion_events", []) or []
                if isinstance(events, list) and isinstance(speech_segments, list):
                    for ev in events:
                        try:
                            s0 = float(ev.get("start", 0.0) or 0.0)
                            e0 = float(ev.get("end", s0) or s0)
                        except Exception:
                            continue
                        overlap = 0.0
                        for seg in speech_segments:
                            try:
                                s1 = float((seg or {}).get("start", 0.0) or 0.0)
                                e1 = float((seg or {}).get("end", s1) or s1)
                            except Exception:
                                continue
                            inter = max(0.0, min(e0, e1) - max(s0, s1))
                            overlap += inter
                        dur = max(1e-6, e0 - s0)
                        ev["speech_overlap_ratio"] = round(float(overlap / dur), 2)
                        ev["speaking"] = bool((overlap / dur) >= 0.3)
            except Exception:
                pass

        # 5) Tutarsızlık analizi
        anomalies = []
        if phase3_enabled:
            print("[Adım 5/5] Tutarsızlık analizi (FAZ-3) kapalı: sentiment/emotion label kullanılmıyor.")
        else:
            print("[Adım 5/5] Tutarsızlık analizi yapılıyor...")
            anomalies = self._find_anomalies(text_data, face_timeline)

        # FAZ-3 Contextual Aggregator: Segment Signal Packages (LLM input)
        segment_signal_packages = []
        if phase3_enabled:
            print("[FAZ-3] Contextual Aggregator: segment paketleri oluşturuluyor...")
            segment_signal_packages = build_segment_signal_packages(
                text_segments=thought_units or text_data,
                audio_signal_timeline=audio_signal_data,
                visual_signal_timeline=face_timeline,
                voice_analysis=voice_analysis,
            )

        # Geçici ses dosyasını temizle
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        duration = time.time() - start_time

        # Birleşik rapor
        report = {
            "interview_id": interview_id,
            "phase": "v3" if phase3_enabled else "v2",
            "duration_seconds": round(duration, 2),
            "video_info": video_info,
            # Metin Analizi
            "text_analysis": {
                "segments": text_data,
                "thought_units": thought_units if phase3_enabled else [],
                "summary": text_summary,
            },
            # Ses Duygu Analizi
            "audio_emotion_analysis": {
                "timeline": audio_emotion_data,
                "summary": audio_emotion_summary,
            },
            # FAZ-3 Audio Signal Analizi (emotion yok)
            "audio_signal_analysis": {
                "timeline": audio_signal_data,
                "summary": audio_signal_summary,
            },
            # Yüz Analizi
            "face_analysis": {
                "timeline": face_timeline,
                "summary": face_summary,
            },
            # FAZ-3 görsel sinyal alias (aynı veri; isim değişimi için)
            "visual_signal_analysis": {
                "timeline": face_timeline if phase3_enabled else [],
                "summary": face_summary if phase3_enabled else {},
            },
            # Ham Ses Özellikleri
            "voice_analysis": voice_analysis,
            # Tutarsızlık
            "anomalies": anomalies,
            # FAZ-3 Segment Signal Package (LLM'ye giden tek veri)
            "segment_signal_packages": segment_signal_packages if phase3_enabled else [],
        }

        print(f"\n{'='*60}")
        print(f"[Pipeline] Analiz Tamamlandı! Süre: {duration:.1f} saniye")
        print(f"  - Metin: {text_summary.get('total_sentences', 0)} cümle")
        if phase3_enabled:
            print(f"  - Ses Sinyali: {audio_signal_summary.get('total_chunks', 0)} parça")
        else:
            print(f"  - Ses Duygusu: {audio_emotion_summary.get('total_chunks', 0)} parça")
        print(f"  - Yüz: {face_summary.get('data_count', 0)} kayıt")
        print(f"  - Anomali: {len(anomalies)} tutarsızlık")
        print(f"{'='*60}\n")

        return report


if __name__ == "__main__":
    p = InterviewAnalysisPipeline()
    print("Pipeline hazır. Kullanım: p.process_interview('video.mp4')")
