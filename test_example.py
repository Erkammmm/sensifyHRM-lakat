"""
Test Script - Örnek Kullanım
Bu script, sistemin nasıl kullanılacağını gösterir.
"""

import sys
import os
import json

# Proje root'unu path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import InterviewAnalysisPipeline


def test_pipeline(video_path: str):
    """
    Pipeline'ı test eder.
    
    Args:
        video_path: Test edilecek video dosyasının yolu
    """
    print("=" * 60)
    print("SensifyHR Mülakat Analiz Sistemi - Test")
    print("=" * 60)
    
    # Pipeline oluştur
    print("\n[1/4] Pipeline başlatılıyor...")
    pipeline = InterviewAnalysisPipeline()
    print("✓ Pipeline hazır")
    
    # Video kontrolü
    if not os.path.exists(video_path):
        print(f"\n❌ Hata: Video dosyası bulunamadı: {video_path}")
        return
    
    print(f"\n[2/4] Video analizi başlatılıyor: {video_path}")
    
    try:
        # Analizi çalıştır
        report = pipeline.process_interview(video_path)
        
        print("\n[3/4] Analiz tamamlandı!")
        print("\n[4/4] Rapor özeti:")
        print("-" * 60)
        print(f"Interview ID: {report['interview_id']}")
        print(f"Süre: {report['duration_seconds']:.2f} saniye")
        print(f"\nDuygu Analizi:")
        print(f"  - Dominant Duygu: {report['emotion_analysis']['dominant_emotion_tr']}")
        print(f"  - Stabilite Skoru: {report['emotion_analysis']['stability_score']:.2f}")
        print(f"\nGöz Teması Analizi:")
        print(f"  - Ortalama Göz Teması: {report['eye_contact_analysis']['average_eye_contact_percentage']:.1f}%")
        print(f"  - Tutarlılık Skoru: {report['eye_contact_analysis']['consistency_score']:.2f}")
        print(f"\nGenel Değerlendirme:")
        print(f"  - Katılım Skoru: {report['overall_assessment']['engagement_score']:.2f}")
        print(f"  - Öneriler:")
        for rec in report['overall_assessment']['recommendations']:
            print(f"    • {rec}")
        print("-" * 60)
        
        # JSON çıktısı (raporlar klasörüne kaydet)
        # Raporlar klasörü oluştur
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        output_file = os.path.join(reports_dir, f"report_{report['interview_id']}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Detaylı rapor kaydedildi: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test için video yolu
    # Kullanım: python test_example.py <video_path>
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        test_pipeline(video_path)
    else:
        print("Kullanım: python test_example.py <video_dosyası_yolu>")
        print("\nÖrnek:")
        print("  python test_example.py test_video.mp4")
