"""
Ana Pipeline Modülü
Tüm analiz adımlarını koordine eder:
  1) Metin Analizi (Whisper STT + ThoughtUnitMerger)
  2) Ses Sinyal Analizi (HuBERT SER + torchaudio)
  3) Ham Ses Özellikleri (VoiceAnalyzer — torchaudio)
  4) Yüz Analizi (UniFace: RetinaFace + DDAMFN + MobileGaze)
  5) Contextual Aggregator → LLM Segment Paketleri
"""

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
from collections import Counter

from .audio.text_analyzer import TextAnalyzer
from .audio.audio_analyzer import AudioAnalyzer
from .vision.face_analyzer import FaceAnalyzer
from .audio.voice_analyzer import VoiceAnalyzer
from .vision.video_processor import VideoProcessor
from .audio.audio_signal_fusion import AudioSignalFusion
from .nlp.contextual_aggregator import build_segment_signal_packages, build_time_blocks
from .audio.thought_unit_merger import merge_into_thought_units
def analyze_consistency(text_sentiment, face_emotion) -> str:
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


def find_anomalies(text_data, face_timeline) -> list:
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
        face_in_range = [f for f in face_timeline if seg_start <= f.get("timestamp", 0) <= seg_end]
        if not face_in_range:
            continue

        # En sık yüz duygusunu bul
        face_emotions = [f.get("emotion") for f in face_in_range]
        dominant_face = Counter(face_emotions).most_common(1)[0][0]

        consistency = analyze_consistency(segment.get("sentiment"), dominant_face)

        if "ŞÜPHELİ" in consistency:
            anomalies.append(
                {
                    "time_range": f"{seg_start:.1f}s - {seg_end:.1f}s",
                    "text": segment.get("text", ""),
                    "text_sentiment": segment.get("sentiment"),
                    "face_emotion": dominant_face,
                    "result": consistency,
                }
            )

    return anomalies


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
    # Pipeline sınıfı sadece orkestrasyon yapar; tutarsızlık tespiti ve
    # kıyaslama mantığı modül seviyesinde fonksiyonlara taşındı. Bu sayede
    # `process_interview` daha lineer ve okunaklı kaldı.
    # ------------------------------------------------------------------

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
        _t0 = time.time()
        video_info = self.video_processor.get_video_info(video_path)
        print(f"[TIMING] VideoProcessor.get_video_info: {time.time() - _t0:.1f}s")

        # 1) Ses çıkarma (sequential — audio_path diğer adımlar için gerekli)
        print("[Adım 1/5] Ses çıkarılıyor...")
        _t0 = time.time()
        audio_path = self.video_processor.extract_audio(video_path)
        print(f"[TIMING] VideoProcessor.extract_audio: {time.time() - _t0:.1f}s")

        # 2) Metin analizi — sequential (Whisper tam GPU gerektirir)
        print("[Adım 2/5] Metin analizi yapılıyor (STT)...")
        _t0 = time.time()
        text_data = self.text_analyzer.process_video(audio_path, phase3_enabled=bool(phase3_enabled))
        text_summary = self.text_analyzer.get_summary(text_data)
        print(f"[TIMING] TextAnalyzer.process_video: {time.time() - _t0:.1f}s")

        thought_units = []
        if phase3_enabled:
            thought_units = merge_into_thought_units(text_data)

        # 3+4) Paralel: VoiceAnalyzer (CPU) + AudioSignalFusion (HuBERT GPU) + FaceAnalyzer (ONNX GPU)
        audio_emotion_data = []
        audio_emotion_summary = {}
        audio_signal_data = []
        audio_signal_summary = {}
        voice_analysis = []
        face_timeline, face_summary = [], {}

        if phase3_enabled:
            print("[Adım 3-4/5] Paralel analiz basliyor: VoiceAnalyzer + AudioSignalFusion + FaceAnalyzer...")
            _t_parallel = time.time()

            def _run_voice():
                _t = time.time()
                result = self.voice_analyzer.analyze_audio(audio_path)
                print(f"[TIMING] VoiceAnalyzer.analyze_audio: {time.time() - _t:.1f}s")
                return result

            def _run_audio_signal():
                _t = time.time()
                fusion = self._get_audio_signal_fusion()
                data = fusion.process_audio(audio_path)
                summary = fusion.get_summary(data)
                print(f"[TIMING] AudioSignalFusion.process_audio: {time.time() - _t:.1f}s")
                return data, summary

            def _run_face():
                _t = time.time()
                timeline, summary = self.face_analyzer.process_video(
                    video_path, phase3_enabled=True
                )
                print(f"[TIMING] FaceAnalyzer.process_video: {time.time() - _t:.1f}s")
                return timeline, summary

            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_voice  = executor.submit(_run_voice)
                fut_audio  = executor.submit(_run_audio_signal)
                fut_face   = executor.submit(_run_face)

                voice_analysis              = fut_voice.result()
                audio_signal_data, audio_signal_summary = fut_audio.result()
                face_timeline, face_summary = fut_face.result()

            print(f"[TIMING] Paralel blok toplam: {time.time() - _t_parallel:.1f}s")
        else:
            print("[Adım 3/5] Ses duygu analizi yapılıyor (HuBERT SER)...")
            _t0 = time.time()
            audio_emotion_data = self.audio_analyzer.process_video(video_path)
            audio_emotion_summary = self.audio_analyzer.get_summary(audio_emotion_data)
            print(f"[TIMING] AudioAnalyzer.process_video: {time.time() - _t0:.1f}s")

            print("[Adım 4/5] Yüz analizi yapılıyor (UniFace)...")
            _t0 = time.time()
            face_timeline, face_summary = self.face_analyzer.process_video(
                video_path, phase3_enabled=False
            )
            print(f"[TIMING] FaceAnalyzer.process_video: {time.time() - _t0:.1f}s")

            _t0 = time.time()
            voice_analysis = self.voice_analyzer.analyze_audio(audio_path)
            print(f"[TIMING] VoiceAnalyzer.analyze_audio: {time.time() - _t0:.1f}s")

        # 5) Tutarsızlık analizi
        anomalies = []
        if phase3_enabled:
            print("[Adım 5/5] Tutarsızlık analizi (FAZ-3) kapalı: sentiment/emotion label kullanılmıyor.")
        else:
            print("[Adım 5/5] Tutarsızlık analizi yapılıyor...")
            anomalies = find_anomalies(text_data, face_timeline)

        # FAZ-4 Contextual Aggregator: Segment Signal Packages (LLM input)
        segment_signal_packages = []
        if phase3_enabled:
            print("[FAZ-4] Contextual Aggregator: segment paketleri oluşturuluyor...")
            _t0 = time.time()
            segment_signal_packages = build_segment_signal_packages(
                text_segments=thought_units or text_data,
                audio_signal_timeline=audio_signal_data,
                face_timeline=face_timeline,
                voice_timeline=(
                    voice_analysis.get("per_second_timeline", [])
                    if isinstance(voice_analysis, dict)
                    else (voice_analysis if isinstance(voice_analysis, list) else [])
                ),
            )
            print(f"[TIMING] ContextualAggregator.build: {time.time() - _t0:.1f}s")

        # Zaman bloğu paragrafları (konuşma yapısı)
        time_blocks = []
        if phase3_enabled:
            _video_duration = float(video_info.get("duration_seconds") or 0.0)
            _voice_tl = (
                voice_analysis.get("per_second_timeline", [])
                if isinstance(voice_analysis, dict)
                else (voice_analysis if isinstance(voice_analysis, list) else [])
            )
            time_blocks = build_time_blocks(
                text_segments=text_data,
                face_timeline=face_timeline,
                audio_signal_timeline=audio_signal_data,
                voice_timeline=_voice_tl,
                duration=_video_duration,
            )
            print(f"[Pipeline] Zaman blokları: {len(time_blocks)} blok")

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
            # Zaman bloğu paragrafları
            "time_blocks": time_blocks if phase3_enabled else [],
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
