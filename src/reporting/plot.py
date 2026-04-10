"""
Grafik Modülü — CPU branch (behaviralanalizcpu)

Matplotlib tabanlı zaman çizelgesi grafikleri kaldırıldı.
HR odaklı raporda bu grafikler yerine kritik anlar bölümü kullanılıyor.
Fonksiyon imzaları korundu (mevcut importlar bozulmasın).
"""

from typing import Dict, List, Any


def generate_report_charts(report: Dict, charts_dir: str) -> List[Dict]:
    """Grafik üretimi devre dışı (CPU branch). Boş liste döner."""
    return []


def plot_face_emotion_timeline_b64(segment_packages: List[Dict]) -> str:
    return ""


def plot_voice_valence_timeline_b64(segment_packages: List[Dict]) -> str:
    return ""


def plot_gaze_timeline_b64(segment_packages: List[Dict]) -> str:
    return ""


def plot_speech_confidence_timeline_b64(segment_packages: List[Dict]) -> str:
    return ""


def plot_voice_energy_timeline_b64(segment_packages: List[Dict]) -> str:
    return ""
