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
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from .audio.audio_analyzer import AudioAnalyzer
from .audio.audio_signal_fusion import AudioSignalFusion
from .audio.role_mapper import RoleMapper, refine_question_ownership
from .audio.speaker_diarizer import SpeakerDiarizer, assign_speakers_to_segments
from .audio.text_analyzer import TextAnalyzer
from .audio.thought_unit_merger import merge_into_thought_units
from .audio.voice_analyzer import VoiceAnalyzer
from .logging_config import get_logger
from .nlp.contextual_aggregator import build_segment_signal_packages, build_smart_blocks
from .vision.face_analyzer import FaceAnalyzer
from .vision.video_processor import VideoProcessor

logger = get_logger(__name__)


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

# Amaç:
# Pipeline baştan sona tek satırda yüzde bazlı ilerleme göstermek.
class _PipelineProgressBar:
    def __init__(self, label: str = "[Pipeline] İlerleme"):
        self.label = label
        self.last_percent = -1
        self.last_stage = ""

    def update(self, percent: float, stage: str = "") -> None:
        percent = int(max(0, min(100, float(percent))))
        stage = stage or ""

        if percent == self.last_percent and stage == self.last_stage:
            return

        self.last_percent = percent
        self.last_stage = stage

        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = "#" * filled + "-" * (bar_width - filled)

        sys.stdout.write(f"\r{self.label}: [{bar}] {percent:3d}% | {stage:<32}")
        sys.stdout.flush()

    def finish(self, stage: str = "Tamamlandı") -> None:
        self.update(100, stage)
        sys.stdout.write("\n")
        sys.stdout.flush()


# Amaç:
# Mapping confidence değerini okunabilir seviyeye çevirmek.
def interpret_mapping_confidence(score: float) -> str:
    if score is None:
        return "unknown"
    if score >= 6.0:
        return "high"
    if score >= 3.0:
        return "medium"
    return "low"


# Amaç:
# Diarization cluster quality score'unu okunabilir seviyeye çevirmek.
# Silhouette score tipik olarak -1 ile 1 arasındadır; burada pratik yorum katmanı ekliyoruz.
def interpret_cluster_quality(score) -> str:
    if score is None:
        return "unknown"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


# Amaç:
# Speaker bazlı özet üretmek.
# Frontend/debug için role_scores'tan daha okunabilir ve doğrudan gösterilebilir yapı sağlar.
def build_speaker_summary(role_map: Dict, role_scores: Dict) -> Dict:
    summary = {}

    total_duration_all = sum(
        float(stats.get("total_duration", 0.0)) for stats in (role_scores or {}).values()
    ) or 1.0

    for speaker_id, stats in (role_scores or {}).items():
        total_duration = float(stats.get("total_duration", 0.0))
        question_count = int(stats.get("question_count", 0))
        segment_count = int(stats.get("segment_count", 0))
        question_ratio = float(stats.get("question_ratio", 0.0))
        first_start = float(stats.get("first_start", 0.0))
        candidate_advantage = float(stats.get("candidate_advantage", 0.0))

        talk_ratio = total_duration / total_duration_all
        role_confidence = abs(candidate_advantage)

        summary[speaker_id] = {
            "role": role_map.get(speaker_id, "Bilinmiyor"),
            "segment_count": segment_count,
            "total_duration": round(total_duration, 3),
            "talk_ratio": round(talk_ratio, 3),
            "question_count": question_count,
            "question_ratio": round(question_ratio, 3),
            "question_density": round(question_count / max(total_duration, 1e-6), 3),
            "first_start": round(first_start, 3),
            "candidate_advantage": round(candidate_advantage, 3),
            "role_confidence": round(role_confidence, 3),
        }

    return summary

class InterviewAnalysisPipeline:
    """
    Mülakat analiz pipeline'ı.
    Video dosyası alır → 4 modül ile analiz eder → birleşik rapor döndürür.
    """

    def __init__(self, phase3_enabled: bool = True):
        logger.info("=" * 60)
        logger.info("[Pipeline] SensifyHR Mülakat Analiz Sistemi Başlatılıyor...")
        logger.info("=" * 60)

        # Amaç:Pipeline içindeki PyTorch thread kullanımını sunucuya göre sabitlemek.
        # Burada varsayılanı 8 yapıyoruz.
        import torch as _torch

        _n_threads = int(os.getenv("SENSIFYHR_CPU_THREADS", "8"))

        # Ana CPU kütüphaneleri ile aynı thread sayısını kullan
        os.environ["OMP_NUM_THREADS"] = str(_n_threads)
        os.environ["MKL_NUM_THREADS"] = str(_n_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(_n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(_n_threads)

        _torch.set_num_threads(_n_threads)

        # Inter-op thread sayısını düşük tutarak oversubscription riskini azaltıyoruz.
        try:
            _torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        logger.info("[Pipeline] CPU thread sayısı sabitlendi: %s", _n_threads)

        # v3 default (ürün modu)
        self.phase3_enabled = bool(phase3_enabled)

        self.video_processor = VideoProcessor()
        self.text_analyzer = TextAnalyzer() #> v2 için
        self.speaker_diarizer = SpeakerDiarizer() #> v2 için
        self.role_mapper = RoleMapper() #> v2 için
        self.audio_analyzer = AudioAnalyzer() #> v2 için
        self.face_analyzer = FaceAnalyzer()
        self.voice_analyzer = VoiceAnalyzer()
        # FAZ-3 audio signal (lazy init değil; model init süresi yüksek olabilir)
        self._audio_signal_fusion: Optional[AudioSignalFusion] = None

        logger.info("[Pipeline] Tüm modüller hazır!\n")

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
        progress = _PipelineProgressBar()
        progress.update(1, "Başlatılıyor")
        logger.info("\n%s", "=" * 60)
        logger.info("[Pipeline] Analiz Başlıyor: %s", interview_id)
        logger.info("[Pipeline] Video: %s", video_path)
        logger.info("[Pipeline] Phase3 Enabled: %s", bool(phase3_enabled))
        logger.info("%s\n", "=" * 60)

        # 0) Video bilgileri
        logger.info("[Adım 0/5] Video bilgileri alınıyor...")
        _t0 = time.time()
        video_info = self.video_processor.get_video_info(video_path)
        progress.update(5, "Video bilgileri alındı")
        logger.debug("[TIMING] VideoProcessor.get_video_info: %.1fs", time.time() - _t0)

        # 1) Ses çıkarma (sequential — audio_path diğer adımlar için gerekli)
        logger.info("[Adım 1/5] Ses çıkarılıyor...")
        _t0 = time.time()
        audio_path = self.video_processor.extract_audio(video_path)
        progress.update(10, "Ses çıkarıldı")
        logger.debug("[TIMING] VideoProcessor.extract_audio: %.1fs", time.time() - _t0)



        def _pipeline_stt_progress(local_ratio: float) -> None:
            # STT aşamasını toplam pipeline'ın %10 - %55 aralığına yay
            progress.update(10 + (float(local_ratio) * 45), "STT / Transkripsiyon")

        # 2) Metin analizi — sequential (Whisper tam GPU gerektirir)
        logger.info("[Adım 2/5] Metin analizi yapılıyor (STT)...")
        _t0 = time.time()
        text_data = self.text_analyzer.process_video(audio_path,phase3_enabled=bool(phase3_enabled),progress_callback=_pipeline_stt_progress,)
        progress.update(60, "STT / Speaker / Role tamam")
        #> v2 için: Speaker Diarization + Role Mapping (sequential; GPU olmayan sunucular için optimize edildi)
        if phase3_enabled:
            diarized_windows = self.speaker_diarizer.diarize(audio_path)
            diarization_diagnostics = getattr(self.speaker_diarizer, "last_diarization_diagnostics", {})

            diarization_diagnostics = {
                **diarization_diagnostics,
                "quality_rating": interpret_cluster_quality(
                    diarization_diagnostics.get("cluster_quality_score")
                ),
            }

            text_data = assign_speakers_to_segments(text_data, diarized_windows)

            # İlk role mapping
            text_data, role_map, role_scores, role_diagnostics = self.role_mapper.assign_roles(text_data)

            # Interaction-driven ownership düzeltmesi
            text_data = refine_question_ownership(text_data, role_map)

            # Düzeltme sonrası tekrar role mapping
            text_data, role_map, role_scores, role_diagnostics = self.role_mapper.assign_roles(text_data)

            role_diagnostics = {
                **role_diagnostics,
                "rating": interpret_mapping_confidence(
                    float(role_diagnostics.get("mapping_confidence", 0.0) or 0.0)
                ),
            }

            speaker_summary = build_speaker_summary(role_map, role_scores)

        else:
            diarized_windows = []
            role_map = {}
            role_scores = {}
            role_diagnostics = {
                "mapping_confidence": 0.0,
                "rating": "unknown",
                "candidate_id": None,
                "interviewer_count": 0,
                "reason": "phase3 disabled",
            }
            diarization_diagnostics = {
                "selected_cluster_k": 0,
                "cluster_quality_score": None,
                "quality_rating": "unknown",
                "k_search_scores": {},
                "vad_region_count": 0,
                "embedding_window_count": 0,
                "speaker_window_distribution": {},
            }
            speaker_summary = {}

        text_summary = self.text_analyzer.get_summary(text_data)
        progress.update(60, "STT / Speaker / Role tamam")
        logger.debug("[TIMING] TextAnalyzer.process_video: %.1fs", time.time() - _t0)

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
            logger.info("[Adım 3-4/5] Paralel analiz basliyor: VoiceAnalyzer + AudioSignalFusion + FaceAnalyzer...")
            _t_parallel = time.time()

            def _run_voice():
                _t = time.time()
                result = self.voice_analyzer.analyze_audio(audio_path)
                logger.debug("[TIMING] VoiceAnalyzer.analyze_audio: %.1fs", time.time() - _t)
                return result

            def _run_audio_signal():
                _t = time.time()
                fusion = self._get_audio_signal_fusion()
                data = fusion.process_audio(audio_path)
                summary = fusion.get_summary(data)
                logger.debug("[TIMING] AudioSignalFusion.process_audio: %.1fs", time.time() - _t)
                return data, summary

            def _run_face():
                _t = time.time()
                timeline, summary = self.face_analyzer.process_video(
                    video_path, phase3_enabled=True
                )
                logger.debug("[TIMING] FaceAnalyzer.process_video: %.1fs", time.time() - _t)
                return timeline, summary

            progress.update(65, "Ses / Yüz analizleri çalışıyor")

            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_voice  = executor.submit(_run_voice)
                fut_audio  = executor.submit(_run_audio_signal)
                fut_face   = executor.submit(_run_face)

                voice_analysis              = fut_voice.result()
                audio_signal_data, audio_signal_summary = fut_audio.result()
                face_timeline, face_summary = fut_face.result()

            logger.debug("[TIMING] Paralel blok toplam: %.1fs", time.time() - _t_parallel)
            progress.update(90, "Ses / Yüz analizleri tamam")
        else:
            logger.info("[Adım 3/5] Ses duygu analizi yapılıyor (HuBERT SER)...")
            _t0 = time.time()
            audio_emotion_data = self.audio_analyzer.process_video(video_path)
            audio_emotion_summary = self.audio_analyzer.get_summary(audio_emotion_data)
            logger.info("[TIMING] AudioAnalyzer.process_video: %.1fs", time.time() - _t0)

            logger.info("[Adım 4/5] Yüz analizi yapılıyor (UniFace)...")
            _t0 = time.time()
            face_timeline, face_summary = self.face_analyzer.process_video(
                video_path, phase3_enabled=False
            )
            logger.debug("[TIMING] FaceAnalyzer.process_video: %.1fs", time.time() - _t0)

            _t0 = time.time()
            voice_analysis = self.voice_analyzer.analyze_audio(audio_path)
            logger.debug("[TIMING] VoiceAnalyzer.analyze_audio: %.1fs", time.time() - _t0)

        # 5) Tutarsızlık analizi
        anomalies = []
        if phase3_enabled:
            logger.info("[Adım 5/5] Tutarsızlık analizi (FAZ-3) kapalı: sentiment/emotion label kullanılmıyor.")
        else:
            logger.info("[Adım 5/5] Tutarsızlık analizi yapılıyor...")
            anomalies = find_anomalies(text_data, face_timeline)

        # FAZ-4 Contextual Aggregator: Segment Signal Packages (LLM input)
        segment_signal_packages = []
        if phase3_enabled:
            logger.info("[FAZ-4] Contextual Aggregator: segment paketleri oluşturuluyor...")
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
            
            logger.debug("[TIMING] ContextualAggregator.build: %.1fs", time.time() - _t0)
            progress.update(96, "Contextual aggregation tamam")
        # Zaman bloğu paragrafları (konuşma yapısı)
        time_blocks = []
        if phase3_enabled:
            _video_duration = float(video_info.get("duration_seconds") or 0.0)
            _voice_tl = (
                voice_analysis.get("per_second_timeline", [])
                if isinstance(voice_analysis, dict)
                else (voice_analysis if isinstance(voice_analysis, list) else [])
            )
            time_blocks = build_smart_blocks(
                text_segments=text_data,
                face_timeline=face_timeline,
                audio_signal_timeline=audio_signal_data,
                voice_timeline=_voice_tl,
                duration=_video_duration,
            )
            logger.info("[Pipeline] Akıllı bloklar: %s blok", len(time_blocks))

        # Geçici ses dosyasını temizle
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        duration = time.time() - start_time

        # Amaç:
        # Input flag yanlış/kapalı gelse bile, phase3 çıktısı gerçekten oluşmuşsa
        # raporda bunu aktif kabul etmek.
        phase3_report_enabled = bool(
            phase3_enabled
            or diarized_windows
            or role_map
            or role_scores
            or speaker_summary
            or any(seg.get("speaker_id") for seg in (text_data or []))
        )

        progress.update(99, "Rapor oluşturuluyor")
        # Birleşik rapor
        report = {
            "interview_id": interview_id,
            "phase": "v3" if phase3_enabled else "v2",
            "duration_seconds": round(duration, 2),
            "video_info": video_info,
            # Metin Analizi, konuşmacı pencereleri ve rol atamaları dahil
            "text_analysis": {
                "segments": text_data,
                "thought_units": thought_units if phase3_report_enabled else [],
                "speaker_windows": diarized_windows if phase3_report_enabled else [],
                "speaker_count": len(role_map) if phase3_report_enabled else 0,
                "speaker_summary": speaker_summary if phase3_report_enabled else {},
                "diarization_diagnostics": diarization_diagnostics if phase3_report_enabled else {
                    "selected_cluster_k": 0,
                    "cluster_quality_score": None,
                    "quality_rating": "unknown",
                    "k_search_scores": {},
                    "vad_region_count": 0,
                    "embedding_window_count": 0,
                    "speaker_window_distribution": {},
                },
                "role_map": role_map if phase3_report_enabled else {},
                "role_scores": role_scores if phase3_report_enabled else {},
                "role_diagnostics": role_diagnostics if phase3_report_enabled else {
                    "mapping_confidence": 0.0,
                    "rating": "unknown",
                    "candidate_id": None,
                    "interviewer_count": 0,
                    "reason": "phase3 disabled",
                },
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

        logger.info("\n%s", "=" * 60)
        logger.info("[Pipeline] Analiz Tamamlandı! Süre: %.1f saniye", duration)
        logger.info(" - Metin: %s cümle", text_summary.get("total_sentences", 0))
        if phase3_enabled:
            logger.info(" - Ses Sinyali: %s parça", audio_signal_summary.get("total_chunks", 0))
        else:
            logger.info(" - Ses Duygusu: %s parça", audio_emotion_summary.get("total_chunks", 0))
        logger.info(" - Yüz: %s kayıt", face_summary.get("data_count", 0))
        logger.info(" - Anomali: %s tutarsızlık", len(anomalies))
        logger.info("%s\n", "=" * 60)

        progress.finish("Tamamlandı")
        return report


if __name__ == "__main__":
    p = InterviewAnalysisPipeline()
    logger.info("Pipeline hazır. Kullanım: p.process_interview('video.mp4')")
