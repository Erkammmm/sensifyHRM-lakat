"""
Gemini AI İstemcisi
Google Gemini API ile İK değerlendirmesi yapar.
Prompt dosyasını okuyup analiz verisiyle birleştirir.
"""

import os
import json
import sys
import warnings


def _load_prompt(phase: str = "v2") -> str:
    """Faz'a göre prompt dosyasını okur."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if (phase or "").lower() == "v3":
        prompt_path = os.path.join(base_dir, "prompt_phase3.txt")
    else:
        prompt_path = os.path.join(base_dir, "prompt.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_env_from_file() -> None:
    """Opsiyonel .env dosyasını okuyup ortam değişkenlerine yükler."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.abspath(os.path.join(base_dir, "..", ".env"))
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


def build_summary_payload(report: dict) -> dict:
    """
    Gemini'ye gidecek minimal özet payload'ı oluşturur.
    Yeni pipeline yapısına uygun.
    """
    phase = report.get("phase", "v2")
    payload = {
        "interview_id": report.get("interview_id"),
        "phase": phase,
        "duration_seconds": report.get("duration_seconds"),
        "video_info": report.get("video_info", {}),
        # Metin analizi özeti
        "text_analysis_summary": report.get("text_analysis", {}).get("summary", {}),
        # Ses duygu özeti
        "audio_emotion_summary": report.get("audio_emotion_analysis", {}).get("summary", {}),
        # FAZ-3 audio signal özeti
        "audio_signal_summary": report.get("audio_signal_analysis", {}).get("summary", {}),
        # Yüz analizi özeti
        "face_analysis_summary": report.get("face_analysis", {}).get("summary", {}),
        # FAZ-3 visual signal özeti
        "visual_signal_summary": report.get("visual_signal_analysis", {}).get("summary", {}),
        # Ham ses özellikleri (serileri kırpılmış)
        "voice_analysis": _strip_voice_series(report.get("voice_analysis", {})),
        # Tutarsızlıklar
        "anomalies": report.get("anomalies", []),
    }

    # FAZ-3: LLM'ye giden ana veri (segment packages) - chunk'lanmış şekilde
    if (phase or "").lower() == "v3":
        packages = report.get("segment_signal_packages", []) or []
        chunks = [packages[i : i + 10] for i in range(0, len(packages), 10)]
        payload["segment_signal_packages_chunks"] = chunks[:30]  # safety cap

    return payload


def _strip_voice_series(data: dict) -> dict:
    """Ham zaman serilerini kaldırarak küçük payload oluşturur."""
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    raw = cleaned.get("raw_voice_features", cleaned)
    if isinstance(raw, dict):
        raw = dict(raw)
        raw.pop("waveform", None)
        raw.pop("mel_spectrogram", None)
        raw.pop("windowed_features", None)
        if isinstance(raw.get("energy_rms"), dict):
            energy = dict(raw["energy_rms"])
            energy.pop("series", None)
            raw["energy_rms"] = energy
        if isinstance(raw.get("pitch_f0"), dict):
            pitch = dict(raw["pitch_f0"])
            pitch.pop("series", None)
            raw["pitch_f0"] = pitch
        if isinstance(raw.get("speech_silence"), dict):
            ss = dict(raw["speech_silence"])
            ss.pop("speech_segments", None)
            raw["speech_silence"] = ss
        if "raw_voice_features" in cleaned:
            cleaned["raw_voice_features"] = raw
        else:
            cleaned = raw
    return cleaned


def generate_gemini_text(summary_text: str, phase: str = "v2") -> str:
    """Prompt + özet metinle Gemini yanıtı üretir."""
    _load_env_from_file()
    warnings.simplefilter("ignore", FutureWarning)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı (.env veya ortam değişkeni).")

    prompt = _load_prompt(phase=phase)
    if not prompt:
        raise RuntimeError("prompt.txt bulunamadı veya boş.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip()
    full_prompt = f"{prompt}\n\n{summary_text}"

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(full_prompt)
    return getattr(response, "text", "")


def _run_cli() -> int:
    """CLI giriş noktası: JSON rapordan özet çıkarıp Gemini'ye yollar."""
    if len(sys.argv) < 2:
        print("Kullanım: python src/gemini.py <report_json_path>", file=sys.stderr)
        return 2

    report_path = sys.argv[1]
    if not os.path.exists(report_path):
        print("report_path bulunamadı", file=sys.stderr)
        return 2

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    summary_payload = build_summary_payload(report)
    summary_text = json.dumps(summary_payload, ensure_ascii=False)
    phase = report.get("phase", "v2")

    try:
        text = generate_gemini_text(summary_text, phase=phase)
        print(text)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
