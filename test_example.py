"""
Hızlı Test Scripti
Pipeline'ı doğrudan çalıştırır ve sonuçları konsola yazdırır.

Kullanım:
    python test_example.py video.mp4
    python test_example.py video.mp4 --ollama   # Ollama + Gemma ile test
    python test_example.py video.mp4 --phase3   # FAZ-3 opt-in
"""

import os
import sys
import json

from src.pipeline import InterviewAnalysisPipeline


def test_pipeline(video_path: str, use_ollama: bool = False):
    """Pipeline'ı çalıştırır ve özet yazdırır."""

    if not os.path.exists(video_path):
        print(f"HATA: '{video_path}' bulunamadı!")
        return

    # 1) Pipeline çalıştır
    pipeline = InterviewAnalysisPipeline()
    report = pipeline.process_interview(video_path, phase3_enabled=("--phase3" in sys.argv))

    # 2) Özet yazdır
    print("\n" + "=" * 60)
    print("📊 ANALİZ ÖZETİ")
    print("=" * 60)

    # Metin
    text_summary = report.get("text_analysis", {}).get("summary", {})
    print(f"\n📝 Metin Analizi:")
    print(f"   Toplam Cümle: {text_summary.get('total_sentences', 0)}")
    print(f"   Baskın Duygu: {text_summary.get('dominant_sentiment', '?')}")
    print(f"   Dağılım: {text_summary.get('sentiment_percentages', {})}")

    # Ses duygu
    audio_summary = report.get("audio_emotion_analysis", {}).get("summary", {})
    print(f"\n🎤 Ses Duygu Analizi:")
    print(f"   Toplam Parça: {audio_summary.get('total_chunks', 0)}")
    print(f"   Baskın Duygu: {audio_summary.get('dominant_emotion', '?')}")
    print(f"   Ort. Güven: {audio_summary.get('avg_confidence', 0):.3f}")

    # Yüz
    face_summary = report.get("face_analysis", {}).get("summary", {})
    print(f"\n😊 Yüz Analizi:")
    print(f"   Baskın Duygu: {face_summary.get('dominant_emotion', '?')}")
    print(f"   Odak Skoru: %{face_summary.get('focus_score', 0)}")
    print(f"   Göz Kırpma/dk: {face_summary.get('blink_rate_per_min', 0)}")

    # Ses özellikleri
    voice = report.get("voice_analysis", {}).get("raw_voice_features", {})
    ss = voice.get("speech_silence", {})
    print(f"\n🔊 Ses Özellikleri:")
    print(f"   Konuşma: {ss.get('total_speech_seconds', 0):.1f}s")
    print(f"   Sessizlik: {ss.get('total_silence_seconds', 0):.1f}s")
    print(f"   RMS Ort: {voice.get('energy_rms', {}).get('mean', 0):.4f}")
    print(f"   Pitch Ort: {voice.get('pitch_f0', {}).get('mean', 0):.1f} Hz")

    # Anomaliler
    anomalies = report.get("anomalies", [])
    print(f"\n⚠️ Tutarsızlıklar: {len(anomalies)}")
    for a in anomalies[:5]:
        print(f"   [{a['time_range']}] {a['result']}")

    # 3) JSON kaydet
    os.makedirs("reports", exist_ok=True)
    output_file = os.path.join("reports", f"report_{report['interview_id']}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ JSON rapor: {output_file}")

    # 4) Opsiyonel: Ollama AI değerlendirmesi
    if use_ollama:
        print("\n" + "=" * 60)
        print("🤖 OLLAMA + GEMMA DEĞERLENDİRMESİ")
        print("=" * 60)
        try:
            from src.ollama_ai import OllamaAI
            ai = OllamaAI()
            result = ai.evaluate_candidate(
                text_data=report.get("text_analysis", {}).get("segments", []),
                audio_data=report.get("audio_emotion_analysis", {}).get("timeline", []),
                face_summary=face_summary,
                anomalies=anomalies,
            )
            print(result)
        except Exception as e:
            print(f"Ollama hatası: {e}")
            print("Ollama kurulumu: https://ollama.com/download")
            print("Model indirme: ollama pull gemma3:12b")
            print("Servis başlatma: ollama serve")

    print("\n✅ Analiz tamamlandı!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python test_example.py video.mp4")
        print("  python test_example.py video.mp4 --ollama")
        print("  python test_example.py video.mp4 --phase3")
        sys.exit(1)

    video = sys.argv[1]
    ollama_flag = "--ollama" in sys.argv
    test_pipeline(video, use_ollama=ollama_flag)
