"""
Davranışsal Analiz Modülü
-------------------------

Bu modül, adayın davranışını sadece tekil frame'lere bakarak değil,
zaman içindeki değişimlere (temporal patterns) göre analiz eder.

Modern yaklaşımlara daha yakın bir yapı kurmak için:
- Frame bazlı statik eşikler yerine 2–5 saniyelik sliding window'lar,
- İlk 60–90 saniyeyi "baseline" (adayın normal davranışı) olarak öğrenme,
- Tüm metrikleri (göz teması, baş pozu vb.) bu baseline'dan sapma (deviation)
  üzerinden değerlendirme
mantığı eklenmiştir.

NOT: Burada hala basitleştirilmiş, kural tabanlı bir yaklaşım kullanıyoruz.
Bi-LSTM / Temporal Transformer gibi modelleri entegre etmek için bu yapı
temel bir zemin sağlar.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


class BehavioralAnalyzer:
    """
    Davranışsal analiz yapan sınıf.
    
    Eski sürümde tamamen frame bazlı ve statik eşiklere dayalı bir analiz
    yapılırken, bu sürümde:
    - Zaman pencereleri (sliding window) üzerinden özet özellikler üretilir,
    - İlk X saniye adayın baseline'ı olarak kabul edilir,
    - Sonraki pencereler bu baseline'dan sapma üzerinden değerlendirilir.
    
    Amaç, "okuma/kopya çekme" gibi kırılgan iddialar yerine,
    **behavioral deviation & regulation** analizi sunmaktır.
    """
    
    def __init__(self, 
                 reading_suspicion_threshold: float = 0.3,
                 anomaly_threshold: float = 2.0,
                 window_seconds: float = 3.0,
                 step_seconds: float = 1.0,
                 baseline_seconds: float = 60.0):
        """
        Davranışsal analizcisini başlatır.
        
        Args:
            reading_suspicion_threshold: Okuma şüphesi eşiği (göz teması yüzdesi)
            anomaly_threshold: Anomali tespiti için z-score eşiği
            window_seconds: Zaman penceresi (sliding window) süresi (saniye)
            step_seconds: Pencereler arası kayma miktarı (saniye)
            baseline_seconds: İlk X saniyeyi baseline (normal davranış) olarak al
        """
        self.reading_suspicion_threshold = reading_suspicion_threshold
        self.anomaly_threshold = anomaly_threshold
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.baseline_seconds = baseline_seconds
    
    # ---------------------------------------------------------------------
    # 1. Sliding window + baseline yardımcı fonksiyonları
    # ---------------------------------------------------------------------

    def _create_time_windows(
        self,
        total_frames: int,
        duration_seconds: float
    ) -> List[Tuple[int, int, float, float]]:
        """
        Frame sayısı ve video süresine göre sliding window indeksleri üretir.
        
        Args:
            total_frames: Toplam frame sayısı (eye_contact_results uzunluğu)
            duration_seconds: Videonun toplam süresi (saniye)
        
        Returns:
            (start_idx, end_idx, start_time, end_time) tuple'larından oluşan liste
        """
        if total_frames == 0 or duration_seconds <= 0:
            return []
        
        # Efektif FPS: elimizdeki analiz edilen frame sayısına göre
        fps_eff = total_frames / duration_seconds
        
        window_size = max(1, int(self.window_seconds * fps_eff))
        step_size = max(1, int(self.step_seconds * fps_eff))
        
        windows: List[Tuple[int, int, float, float]] = []
        
        for start in range(0, max(1, total_frames - window_size + 1), step_size):
            end = min(total_frames, start + window_size)
            start_time = start / fps_eff
            end_time = end / fps_eff
            windows.append((start, end, start_time, end_time))
        
        return windows

    def _extract_window_features(
        self,
        eye_contact_results: List[Optional[Dict]],
        windows: List[Tuple[int, int, float, float]]
    ) -> List[Dict]:
        """
        Her zaman penceresi (window) için özet özellikler çıkarır.
        
        Şimdilik sadece göz teması ve baş pozu üzerinden basit metrikler
        hesaplıyoruz. İleride ses ve yüz embedding'leri de buraya eklenebilir.
        
        Returns:
            Her window için:
            {
              'start_frame', 'end_frame', 'start_time', 'end_time',
              'eye_contact_mean', 'eye_contact_std',
              'gaze_angle_mean', 'gaze_angle_std',
              'downward_ratio', 'saccade_rate'
            }
        """
        window_features: List[Dict] = []
        
        for (start, end, start_time, end_time) in windows:
            segment = eye_contact_results[start:end]
            
            # Geçerli (yüz bulunan) sonuçlar
            valid = [
                r for r in segment
                if r is not None and r.get('has_face', False)
            ]
            
            if not valid:
                window_features.append({
                    'start_frame': start,
                    'end_frame': end,
                    'start_time': start_time,
                    'end_time': end_time,
                    'eye_contact_mean': 0.0,
                    'eye_contact_std': 0.0,
                    'gaze_angle_mean': 90.0,
                    'gaze_angle_std': 0.0,
                    'downward_ratio': 0.0,
                    'saccade_rate': 0.0,
                    'coverage': 0.0
                })
                continue
            
            # Göz teması skorları ve bakış açıları
            eye_scores = []
            gaze_angles = []
            pitch_values = []
            
            for r in valid:
                gaze = r.get('gaze', {})
                head_pose = r.get('head_pose', {})
                
                eye_scores.append(gaze.get('eye_contact_score', 0.0))
                gaze_angles.append(gaze.get('angle_degrees', 90.0))
                
                if head_pose.get('success', False):
                    pitch_values.append(head_pose.get('pitch', 0.0))
            
            # Eye contact istatistikleri
            eye_mean = float(np.mean(eye_scores)) if eye_scores else 0.0
            eye_std = float(np.std(eye_scores)) if len(eye_scores) > 1 else 0.0
            
            # Gaze açı istatistikleri
            angle_mean = float(np.mean(gaze_angles)) if gaze_angles else 90.0
            angle_std = float(np.std(gaze_angles)) if len(gaze_angles) > 1 else 0.0
            
            # Aşağı bakma oranı (pitch > 15 derece)
            if pitch_values:
                downward_frames = sum(1 for p in pitch_values if p > 15.0)
                downward_ratio = float(downward_frames / len(pitch_values))
            else:
                downward_ratio = 0.0
            
            # Saccade (bakış sıçraması) oranı: açı farkı > 10 derece ise bir saccade say
            saccades = 0
            if len(gaze_angles) > 1:
                for i in range(1, len(gaze_angles)):
                    if abs(gaze_angles[i] - gaze_angles[i-1]) > 10.0:
                        saccades += 1
                saccade_rate = float(saccades / max(1, len(gaze_angles) - 1))
            else:
                saccade_rate = 0.0
            
            coverage = float(len(valid) / max(1, len(segment)))
            
            window_features.append({
                'start_frame': start,
                'end_frame': end,
                'start_time': start_time,
                'end_time': end_time,
                'eye_contact_mean': eye_mean,
                'eye_contact_std': eye_std,
                'gaze_angle_mean': angle_mean,
                'gaze_angle_std': angle_std,
                'downward_ratio': downward_ratio,
                'saccade_rate': saccade_rate,
                'coverage': coverage
            })
        
        return window_features

    def _compute_baseline_stats(
        self,
        window_features: List[Dict]
    ) -> Dict:
        """
        İlk X saniyelik pencereleri kullanarak baseline istatistiklerini hesaplar.
        
        Baseline:
        - eye_contact_mean
        - gaze_angle_mean
        - downward_ratio
        - saccade_rate
        için ortalama ve standart sapma değerlerini içerir.
        """
        if not window_features:
            return {}
        
        baseline_windows = [
            w for w in window_features
            if w['end_time'] <= self.baseline_seconds
        ]
        
        # Eğer video çok kısaysa, tüm pencereleri baseline olarak kullan
        if len(baseline_windows) < 3:
            baseline_windows = window_features
        
        def _collect(key: str) -> List[float]:
            return [w[key] for w in baseline_windows]
        
        stats_dict: Dict[str, Dict[str, float]] = {}
        
        for key in ['eye_contact_mean', 'gaze_angle_mean',
                    'downward_ratio', 'saccade_rate']:
            values = _collect(key)
            if not values:
                stats_dict[key] = {'mean': 0.0, 'std': 0.0}
            else:
                mean = float(np.mean(values))
                std = float(np.std(values)) if len(values) > 1 else 0.0
                stats_dict[key] = {'mean': mean, 'std': std}
        
        return {
            'baseline_windows': len(baseline_windows),
            'stats': stats_dict
        }

    def _analyze_deviations(
        self,
        window_features: List[Dict],
        baseline: Dict
    ) -> Dict:
        """
        Baseline'a göre her window'un sapmasını (deviation) analiz eder.
        
        Burada henüz gerçek bir Bi-LSTM / Transformer yok; ancak:
        - Hangi pencerelerde ciddi sapma var,
        - Bu sapmaların oranı nedir,
        - Özellikle downward gaze + düşük eye contact kombinasyonu
        için basit skorlar üretir.
        """
        if not window_features or not baseline:
            return {
                'high_deviation_windows': 0,
                'total_windows': len(window_features),
                'high_deviation_ratio': 0.0,
                'reading_suspicion_score': 0.0,
                'average_deviation': 0.0,
                'window_deviations': []
            }
        
        stats_dict = baseline.get('stats', {})
        
        window_deviations = []
        high_dev_count = 0
        deviation_sums = []
        
        for w in window_features:
            dev_entry = {
                'start_time': w['start_time'],
                'end_time': w['end_time']
            }
            total_dev = 0.0
            dev_components = 0
            
            # Her metrik için basit z-score benzeri sapma
            for key in ['eye_contact_mean', 'gaze_angle_mean',
                        'downward_ratio', 'saccade_rate']:
                if key not in stats_dict:
                    continue
                mean = stats_dict[key]['mean']
                std = stats_dict[key]['std']
                value = w[key]
                
                if std > 1e-6:
                    z = abs(value - mean) / std
                else:
                    # Std yoksa (sabit baseline), farkı mutlak sapma olarak al
                    z = abs(value - mean)
                
                dev_entry[f'{key}_z'] = float(z)
                total_dev += z
                dev_components += 1
            
            avg_dev = total_dev / dev_components if dev_components > 0 else 0.0
            dev_entry['avg_deviation'] = float(avg_dev)
            deviation_sums.append(avg_dev)
            
            # "Yüksek sapma" eşiği: z > 2.0 benzeri
            if avg_dev > 2.0:
                high_dev_count += 1
                dev_entry['is_high_deviation'] = True
            else:
                dev_entry['is_high_deviation'] = False
            
            window_deviations.append(dev_entry)
        
        total_windows = len(window_features)
        high_dev_ratio = high_dev_count / total_windows if total_windows > 0 else 0.0
        avg_deviation_overall = float(np.mean(deviation_sums)) if deviation_sums else 0.0
        
        # Okuma / yoğun odaklanma şüphesini,
        # özellikle downward_ratio ve eye_contact_mean sapmaları üzerinden yaklaşıkla.
        reading_suspicion = 0.0
        if stats_dict:
            # Downward ve eye contact sapmalarının yüksek olduğu pencereleri say
            suspicious_windows = 0
            for d in window_deviations:
                dw_z = d.get('downward_ratio_z', 0.0)
                eye_z = d.get('eye_contact_mean_z', 0.0)
                if dw_z > 1.5 and eye_z > 1.0:
                    suspicious_windows += 1
            if total_windows > 0:
                reading_suspicion = min(1.0, suspicious_windows / total_windows)
        
        return {
            'high_deviation_windows': high_dev_count,
            'total_windows': total_windows,
            'high_deviation_ratio': float(high_dev_ratio),
            'reading_suspicion_score': float(reading_suspicion),
            'average_deviation': float(avg_deviation_overall),
            'window_deviations': window_deviations
        }
    
    def analyze_gaze_patterns(self, 
                              eye_contact_results: List[Optional[Dict]]) -> Dict:
        """
        Göz hareketi desenlerini analiz eder (frame bazlı hızlı özet).
        
        NOT:
        - Eski versiyon tamamen bu fonksiyona dayanıyordu.
        - Yeni versiyonda asıl karar mekanizması sliding window + baseline
          analizi (_analyze_deviations) olmakla birlikte, bu fonksiyon
          hala hızlı bir özet üretmek ve explainability sağlamak için
          kullanılmaktadır.
        """
        # Geçerli sonuçları filtrele
        valid_results = [r for r in eye_contact_results if r is not None and r.get('has_face', False)]
        
        if len(valid_results) < 10:  # Minimum frame sayısı
            return {
                'reading_suspicion_score': 0.0,
                'downward_gaze_ratio': 0.0,
                'gaze_pattern_anomalies': []
            }
        
        # Baş pozisyonu ve bakış açıları
        pitch_values = []
        gaze_angles = []
        eye_contact_scores = []
        
        for result in valid_results:
            head_pose = result.get('head_pose', {})
            gaze = result.get('gaze', {})
            
            if head_pose.get('success', False):
                pitch = head_pose.get('pitch', 0.0)
                pitch_values.append(pitch)
            
            if gaze:
                angle = gaze.get('angle_degrees', 90.0)
                eye_contact_score = gaze.get('eye_contact_score', 0.0)
                gaze_angles.append(angle)
                eye_contact_scores.append(eye_contact_score)
        
        if not pitch_values or not gaze_angles:
            return {
                'reading_suspicion_score': 0.0,
                'downward_gaze_ratio': 0.0,
                'gaze_pattern_anomalies': []
            }
        
        # Aşağı bakma tespiti (pitch pozitif = aşağı bakma)
        # Pitch > 15 derece = aşağı bakma
        downward_frames = sum(1 for p in pitch_values if p > 15)
        downward_ratio = downward_frames / len(pitch_values)
        
        # Yüksek açılı bakış (göz teması dışı)
        high_angle_frames = sum(1 for a in gaze_angles if a > 45)
        high_angle_ratio = high_angle_frames / len(gaze_angles)
        
        # Düşük göz teması + aşağı bakma = okuma şüphesi
        avg_eye_contact = np.mean(eye_contact_scores) if eye_contact_scores else 0.0
        low_eye_contact = avg_eye_contact < self.reading_suspicion_threshold
        
        # Okuma şüphesi skoru (eski basit skor, explainability için tutuluyor)
        reading_suspicion = 0.0
        if low_eye_contact and downward_ratio > 0.3:
            # Yüksek aşağı bakma + düşük göz teması
            reading_suspicion = min(1.0, (downward_ratio * 0.6 + (1 - avg_eye_contact) * 0.4))
        elif downward_ratio > 0.5:
            # Çok yüksek aşağı bakma
            reading_suspicion = downward_ratio * 0.8
        
        # Sürekli aşağı bakma pattern'i (arka arkaya frame'ler)
        consecutive_downward = self._detect_consecutive_pattern(
            [p > 15 for p in pitch_values],
            min_consecutive=5
        )
        
        # Anomali tespiti (z-score)
        anomalies = []
        if len(pitch_values) > 10:
            pitch_z_scores = np.abs(stats.zscore(pitch_values))
            anomaly_frames = np.where(pitch_z_scores > self.anomaly_threshold)[0]
            if len(anomaly_frames) > 0:
                anomalies.append({
                    'type': 'head_pose_anomaly',
                    'count': len(anomaly_frames),
                    'ratio': len(anomaly_frames) / len(pitch_values)
                })
        
        return {
            'reading_suspicion_score': float(reading_suspicion),  # Düzeltildi: _legacy kaldırıldı
            'downward_gaze_ratio': float(downward_ratio),
            'high_angle_gaze_ratio': float(high_angle_ratio),
            'consecutive_downward_patterns': consecutive_downward,
            'gaze_pattern_anomalies': anomalies,
            'average_eye_contact_during_analysis': float(avg_eye_contact)
        }
    
    def analyze_emotion_eye_correlation(self,
                                        emotion_results: List[Optional[Dict]],
                                        eye_contact_results: List[Optional[Dict]]) -> Dict:
        """
        Duygu ve göz teması korelasyonunu analiz eder.
        Stresli duygular + düşük göz teması = şüpheli davranış.
        
        Args:
            emotion_results: Duygu analiz sonuçları
            eye_contact_results: Göz teması analiz sonuçları
            
        Returns:
            Korelasyon analizi
        """
        # Frame sayısını eşitle (minimum)
        min_frames = min(len(emotion_results), len(eye_contact_results))
        if min_frames < 10:
            return {
                'correlation_score': 0.0,
                'stress_eye_correlation': 0.0,
                'anomalies': []
            }
        
        # Her frame için duygu ve göz teması skorları
        emotion_scores = []
        eye_contact_scores = []
        stress_emotions = ['angry', 'fear', 'sad', 'disgust']
        
        for i in range(min_frames):
            emotion_result = emotion_results[i]
            eye_result = eye_contact_results[i]
            
            if emotion_result and eye_result and eye_result.get('has_face', False):
                # Dominant duyguyu bul
                if emotion_result:
                    dominant_emotion = max(emotion_result.items(), key=lambda x: x[1])[0]
                    emotion_value = 1.0 if dominant_emotion in stress_emotions else 0.0
                    emotion_scores.append(emotion_value)
                else:
                    emotion_scores.append(0.0)
                
                # Göz teması skoru
                gaze = eye_result.get('gaze', {})
                eye_contact = gaze.get('eye_contact_score', 0.0) if gaze else 0.0
                eye_contact_scores.append(eye_contact)
        
        if len(emotion_scores) < 10:
            return {
                'correlation_score': 0.0,
                'stress_eye_correlation': 0.0,
                'anomalies': []
            }
        
        # Korelasyon hesapla
        if len(emotion_scores) > 1:
            correlation = np.corrcoef(emotion_scores, eye_contact_scores)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Stres + düşük göz teması pattern'i
        stress_low_eye_pattern = 0
        for i in range(len(emotion_scores)):
            if emotion_scores[i] > 0.5 and eye_contact_scores[i] < 0.3:
                stress_low_eye_pattern += 1
        
        stress_eye_ratio = stress_low_eye_pattern / len(emotion_scores) if emotion_scores else 0.0
        
        # Anomali: Yüksek stres + çok düşük göz teması
        anomalies = []
        if stress_eye_ratio > 0.2:
            anomalies.append({
                'type': 'stress_low_eye_contact',
                'ratio': stress_eye_ratio,
                'description': 'Stresli duygular ile düşük göz teması birlikte görüldü'
            })
        
        return {
            'correlation_score': float(correlation),
            'stress_eye_correlation': float(stress_eye_ratio),
            'stress_emotion_frames': int(sum(emotion_scores)),
            'anomalies': anomalies
        }
    
    def analyze_voice_behavior_correlation(self,
                                           voice_summary: Optional[Dict],
                                           emotion_summary: Dict,
                                           eye_contact_metrics: Dict) -> Dict:
        """
        Ses, duygu ve göz teması korelasyonunu analiz eder.
        Yüksek stres (ses) + negatif duygular + düşük göz teması = şüpheli.
        
        Args:
            voice_summary: Ses analizi özeti
            emotion_summary: Duygu analizi özeti
            eye_contact_metrics: Göz teması metrikleri
            
        Returns:
            Çoklu korelasyon analizi
        """
        if not voice_summary or 'error' in voice_summary:
            return {
                'multi_modal_suspicion_score': 0.0,
                'anomalies': []
            }
        
        # Skorları normalize et
        stress_level = voice_summary.get('stress_level', 0.0)
        confidence_score = voice_summary.get('confidence_score', 0.0)
        
        # Negatif duygu kontrolü
        dominant_emotion = emotion_summary.get('dominant_emotion', 'neutral')
        negative_emotions = ['angry', 'fear', 'sad', 'disgust']
        is_negative_emotion = dominant_emotion in negative_emotions
        
        # Göz teması
        eye_contact_pct = eye_contact_metrics.get('average_eye_contact_percentage', 0.0) / 100.0
        
        # Şüphe skoru hesapla
        suspicion_factors = []
        
        # Yüksek stres
        if stress_level > 0.6:
            suspicion_factors.append(('high_stress', stress_level))
        
        # Düşük güven
        if confidence_score < 0.4:
            suspicion_factors.append(('low_confidence', 1.0 - confidence_score))
        
        # Negatif duygu
        if is_negative_emotion:
            suspicion_factors.append(('negative_emotion', 0.7))
        
        # Düşük göz teması
        if eye_contact_pct < 0.4:
            suspicion_factors.append(('low_eye_contact', 1.0 - eye_contact_pct))
        
        # Çoklu faktör şüphesi
        multi_modal_suspicion = 0.0
        if len(suspicion_factors) >= 2:
            # Birden fazla şüpheli faktör varsa skor artar
            base_score = sum(factor[1] for factor in suspicion_factors) / len(suspicion_factors)
            multiplier = 1.0 + (len(suspicion_factors) - 1) * 0.2  # Her ek faktör %20 artırır
            multi_modal_suspicion = min(1.0, base_score * multiplier)
        elif len(suspicion_factors) == 1:
            multi_modal_suspicion = suspicion_factors[0][1] * 0.6  # Tek faktör daha düşük skor
        
        # Anomaliler
        anomalies = []
        if len(suspicion_factors) >= 3:
            anomalies.append({
                'type': 'multi_modal_suspicion',
                'severity': 'high',
                'factors': [f[0] for f in suspicion_factors],
                'description': 'Birden fazla şüpheli davranış birlikte tespit edildi'
            })
        
        return {
            'multi_modal_suspicion_score': float(multi_modal_suspicion),
            'suspicion_factors': [f[0] for f in suspicion_factors],
            'anomalies': anomalies
        }
    
    def calculate_overall_suspicion_score(self,
                                         gaze_patterns: Dict,
                                         emotion_eye_correlation: Dict,
                                         voice_behavior_correlation: Dict) -> Dict:
        """
        Genel şüphe skorunu hesaplar.
        
        Args:
            gaze_patterns: Göz hareketi desen analizi
            emotion_eye_correlation: Duygu-göz korelasyonu
            voice_behavior_correlation: Ses-davranış korelasyonu
            
        Returns:
            Genel şüphe analizi
        """
        # Bireysel skorlar
        reading_suspicion = gaze_patterns.get('reading_suspicion_score', 0.0)
        stress_eye_correlation = emotion_eye_correlation.get('stress_eye_correlation', 0.0)
        multi_modal_suspicion = voice_behavior_correlation.get('multi_modal_suspicion_score', 0.0)
        
        # Ağırlıklı ortalama
        overall_suspicion = (
            reading_suspicion * 0.4 +  # Okuma şüphesi en önemli
            stress_eye_correlation * 0.3 +  # Stres-göz korelasyonu
            multi_modal_suspicion * 0.3  # Çoklu modal şüphe
        )
        
        # Tüm anomalileri birleştir
        all_anomalies = []
        all_anomalies.extend(gaze_patterns.get('gaze_pattern_anomalies', []))
        all_anomalies.extend(emotion_eye_correlation.get('anomalies', []))
        all_anomalies.extend(voice_behavior_correlation.get('anomalies', []))
        
        # Risk seviyesi
        if overall_suspicion > 0.7:
            risk_level = 'high'
        elif overall_suspicion > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'overall_suspicion_score': float(overall_suspicion),
            'reading_suspicion_score': float(reading_suspicion),
            'stress_behavior_score': float(stress_eye_correlation),
            'multi_modal_score': float(multi_modal_suspicion),
            'risk_level': risk_level,
            'total_anomalies': len(all_anomalies),
            'anomalies': all_anomalies
        }
    
    def _detect_consecutive_pattern(self, pattern: List[bool], min_consecutive: int = 5) -> int:
        """
        Arka arkaya gelen pattern'leri tespit eder.
        
        Args:
            pattern: Boolean pattern listesi
            min_consecutive: Minimum arka arkaya sayısı
            
        Returns:
            Tespit edilen pattern sayısı
        """
        if not pattern:
            return 0
        
        consecutive_count = 0
        current_streak = 0
        
        for value in pattern:
            if value:
                current_streak += 1
                if current_streak >= min_consecutive:
                    consecutive_count += 1
            else:
                current_streak = 0
        
        return consecutive_count
    
    def analyze_behavior(self,
                        emotion_results: List[Optional[Dict]],
                        eye_contact_results: List[Optional[Dict]],
                        voice_summary: Optional[Dict],
                        emotion_summary: Dict,
                        eye_contact_metrics: Dict,
                        video_info: Optional[Dict] = None) -> Dict:
        """
        Tüm davranışsal analizleri yapar.
        
        Bu fonksiyon artık iki katmanlı çalışır:
        1) Sliding window + baseline deviation analizi (temel modern yaklaşım)
        2) Eski frame bazlı analiz fonksiyonlarının "explainability" için
           ürettiği ek metrikler (legacy ama faydalı özetler)
        
        Args:
            emotion_results: Frame bazlı duygu sonuçları
            eye_contact_results: Frame bazlı göz teması sonuçları
            voice_summary: Ses analizi özeti
            emotion_summary: Duygu analizi özeti
            eye_contact_metrics: Göz teması metrikleri
            video_info: Video bilgileri (fps, duration_seconds vb.) -
                        sliding window için kullanılır.
            
        Returns:
            Davranışsal analiz sonuçları
        """
        # ------------------------------------------------------------------
        # 1) Sliding window + baseline deviation analizi (ANA KATMAN)
        # ------------------------------------------------------------------
        duration_seconds = 0.0
        if video_info:
            duration_seconds = float(video_info.get('duration_seconds', 0.0))
        
        total_frames = len(eye_contact_results)
        
        windows = self._create_time_windows(
            total_frames=total_frames,
            duration_seconds=duration_seconds if duration_seconds > 0 else max(1.0, total_frames / 30.0)
        )
        
        window_features = self._extract_window_features(
            eye_contact_results=eye_contact_results,
            windows=windows
        )
        
        baseline_stats = self._compute_baseline_stats(window_features)
        deviation_analysis = self._analyze_deviations(
            window_features=window_features,
            baseline=baseline_stats
        )
        
        # ------------------------------------------------------------------
        # 2) Eski fonksiyonlar üzerinden özet / açıklama (LEGACY KATMAN)
        # ------------------------------------------------------------------
        gaze_patterns_legacy = self.analyze_gaze_patterns(eye_contact_results)
        
        emotion_eye_correlation = self.analyze_emotion_eye_correlation(
            emotion_results,
            eye_contact_results
        )
        
        voice_behavior_correlation = self.analyze_voice_behavior_correlation(
            voice_summary,
            emotion_summary,
            eye_contact_metrics
        )
        
        # Legacy skorları hesapla (eski rule-based)
        overall_suspicion = self.calculate_overall_suspicion_score(
            gaze_patterns_legacy,
            emotion_eye_correlation,
            voice_behavior_correlation
        )
        
        # ------------------------------------------------------------------
        # 3) Özet
        # ------------------------------------------------------------------
        # Ana skor: baseline deviation + legacy skorların birleşimi
        reading_suspicion = deviation_analysis.get('reading_suspicion_score', 0.0)
        legacy_suspicion = overall_suspicion.get('overall_suspicion_score', 0.0)
        
        # Güvenli tip kontrolü
        if not isinstance(reading_suspicion, (int, float)):
            reading_suspicion = 0.0
        if not isinstance(legacy_suspicion, (int, float)):
            legacy_suspicion = 0.0
        
        combined_suspicion = float(
            0.6 * float(reading_suspicion) +
            0.4 * float(legacy_suspicion)
        )
        
        # Risk seviyesi
        if combined_suspicion > 0.7:
            risk_level = 'high'
        elif combined_suspicion > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        summary = {
            'suspicion_score': combined_suspicion,
            'risk_level': risk_level,
            'reading_suspicion': float(reading_suspicion),
            'high_deviation_ratio': deviation_analysis.get('high_deviation_ratio', 0.0),
            'average_deviation': deviation_analysis.get('average_deviation', 0.0),
            'total_anomalies': overall_suspicion.get('total_anomalies', 0)
        }
        
        return {
            'summary': summary,
            'temporal_baseline_analysis': {
                'window_seconds': self.window_seconds,
                'step_seconds': self.step_seconds,
                'baseline_seconds': self.baseline_seconds,
                'baseline_stats': baseline_stats,
                'deviation_analysis': deviation_analysis
            },
            'gaze_patterns_legacy': gaze_patterns_legacy,
            'emotion_eye_correlation': emotion_eye_correlation,
            'voice_behavior_correlation': voice_behavior_correlation,
            'overall_suspicion': overall_suspicion
        }


if __name__ == "__main__":
    # Test kodu
    analyzer = BehavioralAnalyzer()
    print("BehavioralAnalyzer modülü hazır.")
    print("\nÖzellikler:")
    print("  - Okuma/cheating şüphesi tespiti")
    print("  - Göz hareketi desen analizi")
    print("  - Duygu-göz teması korelasyonu")
    print("  - Çoklu modal şüphe analizi")
