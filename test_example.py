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
        print(f"\nFrame Bazlı Mediapipe Analizi (örnek 5 frame):")
        frame_analysis = report.get("frame_analysis", [])
        if frame_analysis:
            for fa in frame_analysis[:5]:
                ts = fa.get("timestamp", 0.0)
                rf = fa.get("raw_features", {})
                gaze = fa.get("gaze", {})
                print(
                    f"  - t={ts:.2f}s | "
                    f"mouth_w={rf.get('mouth_width_norm', 0.0):.3f}, "
                    f"mouth_h={rf.get('mouth_height_norm', 0.0):.3f}, "
                    f"eye_open={rf.get('eye_opening_norm', 0.0):.3f}, "
                    f"brow_dist={rf.get('brow_distance_norm', 0.0):.3f}, "
                    f"jaw_open={rf.get('jaw_open_norm', 0.0):.3f} | "
                    f"gaze_dir={gaze.get('direction', 'center')}, "
                    f"gx={gaze.get('x', 0.0):.2f}, gy={gaze.get('y', 0.0):.2f}"
                )
        else:
            print("  - Frame analizi mevcut değil")

        # Frame Özeti (Gemini API için sadeleştirilmiş)
        frame_summary = report.get("frame_summary", {})
        print(f"\nFrame Özeti (Gemini API için):")
        if frame_summary:
            # Genel İstatistikler
            gen_stats = frame_summary.get("general_statistics", {})
            if gen_stats:
                avg_feat = gen_stats.get("average_features", {})
                print(f"  Genel İstatistikler:")
                print(f"    - Ortalama Mouth Width: {avg_feat.get('mouth_width_norm', 0.0):.3f}")
                print(f"    - Ortalama Mouth Height: {avg_feat.get('mouth_height_norm', 0.0):.3f}")
                print(f"    - Ortalama Eye Opening: {avg_feat.get('eye_opening_norm', 0.0):.3f}")
                print(f"    - Ortalama Brow Distance: {avg_feat.get('brow_distance_norm', 0.0):.3f}")
                print(f"    - Ortalama Jaw Open: {avg_feat.get('jaw_open_norm', 0.0):.3f}")
                print(f"    - Gaze Center Yüzdesi: {gen_stats.get('gaze_center_percentage', 0.0):.1f}%")
                print(f"    - Toplam Frame: {gen_stats.get('total_frames_analyzed', 0)}")
            
            # Göz Analizi Özeti
            gaze_sum = frame_summary.get("gaze_summary", {})
            if gaze_sum:
                print(f"  Göz Analizi Özeti:")
                dir_perc = gaze_sum.get("direction_percentages", {})
                for direction, percentage in dir_perc.items():
                    count = gaze_sum.get("direction_counts", {}).get(direction, 0)
                    print(f"    - {direction}: {percentage:.1f}% ({count} frame)")
            
            # Bilişsel Yük Skoru
            cog_load = frame_summary.get("cognitive_load_score", {})
            if cog_load:
                print(f"  Bilişsel Yük Skoru:")
                print(f"    - Thinking/Reading: {cog_load.get('thinking_reading_count', 0)} frame ({cog_load.get('thinking_reading_percentage', 0.0):.1f}%)")
                print(f"    - Konuşma: {cog_load.get('total_speaking_frames', 0)} frame ({cog_load.get('speaking_percentage', 0.0):.1f}%)")
            
            # Duygu Değişim Noktaları
            change_points = frame_summary.get("emotion_change_points", [])
            if change_points:
                print(f"  Duygu Değişim Noktaları ({len(change_points)} adet):")
                for cp in change_points[:5]:  # İlk 5'ini göster
                    print(f"    - t={cp.get('timestamp', 0.0):.2f}s: {cp.get('feature', '')} {cp.get('change_percentage', 0.0):.1f}% değişti ({cp.get('previous_value', 0.0):.3f} → {cp.get('current_value', 0.0):.3f})")
        else:
            print("  - Frame özeti mevcut değil")

        # Ses Analizi (ham özellikler)
        voice_analysis = report.get('voice_analysis', {})
        print(f"\nSes Analizi (raw özellikler):")
        if voice_analysis:
            # Pipeline'da voice_analysis doğrudan raw_voice_features dict'i olarak yazılıyor.
            rv = voice_analysis.get('raw_voice_features', voice_analysis)
            sr = rv.get('speech_rate', {})
            print(f"  - Speech rate: {sr.get('value', 0.0):.2f} ({sr.get('unit', '')})")
            print(f"  - RMS Energy (mean): {rv.get('rms_energy', {}).get('mean', 0.0):.6f}")
            print(f"  - Pitch f0 mean: {rv.get('pitch_f0', {}).get('mean', 0.0):.2f} Hz")
            print(f"  - Silence ratio: {rv.get('silence_ratio', 0.0):.3f}")
            print(f"  - Duration: {rv.get('duration_seconds', 0.0):.2f} s")
        else:
            print("  - Ses analizi mevcut değil")
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
