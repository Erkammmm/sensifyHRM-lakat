"""
Gemini AI İstemcisi — CPU Branch
google-genai (yeni SDK) kullanır; gemini-2.5-flash öncelikli.
"""

import os
import json
import sys
import warnings
from typing import Optional


# ─── Prompt yükleyici ──────────────────────────────────────────────────────

def _load_prompt(phase: str = "v2") -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fname = "prompt_phase3.txt" if (phase or "").lower() == "v3" else "prompt.txt"
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_env_from_file() -> None:
    """Proje kökündeki .env dosyasını ortam değişkenlerine yükler."""
    # src/nlp/ → src/ → proje kökü
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for levels_up in (2, 3):
        env_path = os.path.abspath(
            os.path.join(base_dir, *[".."] * levels_up, ".env")
        )
        if os.path.exists(env_path):
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
                pass
            break


# ─── Payload oluşturucu ────────────────────────────────────────────────────

def build_summary_payload(report: dict) -> dict:
    """Gemini'ye gidecek minimal özet payload'ı oluşturur."""
    phase = report.get("phase", "v2")
    payload = {
        "interview_id": report.get("interview_id"),
        "phase": phase,
        "duration_seconds": report.get("duration_seconds"),
        "video_info": report.get("video_info", {}),
        "text_analysis_summary": report.get("text_analysis", {}).get("summary", {}),
        "audio_emotion_summary": report.get("audio_emotion_analysis", {}).get("summary", {}),
        "audio_signal_summary": report.get("audio_signal_analysis", {}).get("summary", {}),
        "face_analysis_summary": report.get("face_analysis", {}).get("summary", {}),
        "visual_signal_summary": report.get("visual_signal_analysis", {}).get("summary", {}),
        "voice_analysis": _strip_voice_series(report.get("voice_analysis", {})),
        "anomalies": report.get("anomalies", []),
    }

    if (phase or "").lower() == "v3":
        packages = report.get("segment_signal_packages", []) or []
        chunks = [packages[i : i + 10] for i in range(0, len(packages), 10)]
        payload["segment_signal_packages_chunks"] = chunks[:30]
        payload["time_blocks"] = report.get("time_blocks", []) or []

    return payload


def _strip_voice_series(data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    raw = cleaned.get("raw_voice_features", cleaned)
    if isinstance(raw, dict):
        raw = dict(raw)
        for key in ("waveform", "mel_spectrogram", "windowed_features"):
            raw.pop(key, None)
        for k in ("energy_rms", "pitch_f0"):
            if isinstance(raw.get(k), dict):
                sub = dict(raw[k])
                sub.pop("series", None)
                raw[k] = sub
        if isinstance(raw.get("speech_silence"), dict):
            ss = dict(raw["speech_silence"])
            ss.pop("speech_segments", None)
            raw["speech_silence"] = ss
        if "raw_voice_features" in cleaned:
            cleaned["raw_voice_features"] = raw
        else:
            cleaned = raw
    return cleaned


# ─── Gemini çağrısı (yeni SDK) ─────────────────────────────────────────────

def _try_gemini_model(model_name: str, full_prompt: str, api_key: str,
                      retries: int = 3, retry_delay: float = 8.0) -> str:
    """
    google-genai SDK ile tek model denemesi.
    503 (geçici yoğunluk) için retry uygular; diğer hatalarda anında çıkar.
    """
    import time
    from google import genai as google_genai

    client = google_genai.Client(api_key=api_key)
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise RuntimeError(f"{model_name}: boş yanıt")
            return text
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # 503 geçici → retry; 404 / 400 → kalıcı hata, tekrar deneme
            if "503" in err_str and attempt < retries:
                print(f"[LLM] {model_name} 503 geçici yoğunluk, {retry_delay}s sonra tekrar ({attempt}/{retries})...")
                time.sleep(retry_delay)
                continue
            raise

    raise last_exc  # type: ignore


def generate_gemini_text(summary_text: str, phase: str = "v2") -> str:
    """
    Gemini fallback zinciri: 2.5-flash → 2.5-flash-8b → 1.5-flash → hata.
    """
    _load_env_from_file()
    warnings.simplefilter("ignore", FutureWarning)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı (.env veya ortam değişkeni).")

    prompt = _load_prompt(phase=phase)
    if not prompt:
        raise RuntimeError("Prompt dosyası bulunamadı veya boş.")

    full_prompt = f"{prompt}\n\n{summary_text}"

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Env override + fallback zinciri (404 olan gemini-2.0-flash kaldırıldı)
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    models = []
    if env_model:
        models.append(env_model)
    models.extend([
        "gemini-2.5-flash-lite",     # birincil (hızlı + ekonomik)
        "gemini-2.5-flash",          # yedek
        "gemini-2.5-flash-8b",       # daha küçük
        "gemini-3.1-flash-lite",     # son yedek
    ])
    seen: set = set()
    candidates = [m for m in models if m and not (m in seen or seen.add(m))]

    last_error: Optional[Exception] = None
    for model_name in candidates:
        try:
            print(f"[LLM] Gemini deneniyor: {model_name}...")
            text = _try_gemini_model(model_name, full_prompt, api_key)
            print(f"[LLM] Gemini başarılı: {model_name} ({len(text)} karakter)")
            return text
        except Exception as exc:
            last_error = exc
            print(f"[LLM] Gemini başarısız ({model_name}): {exc}")
            continue

    raise RuntimeError(f"Tüm Gemini modelleri başarısız: {last_error}")


# ─── Public entry point ────────────────────────────────────────────────────

def generate_analysis(report: dict) -> str:
    """
    api/main.py tarafından çağrılan ana fonksiyon.
    Pipeline rapor dict'ini alır, özet payload oluşturur, Gemini'ye gönderir.
    """
    summary_payload = build_summary_payload(report)
    summary_text = json.dumps(summary_payload, ensure_ascii=False)
    phase = report.get("phase", "v2")
    return generate_gemini_text(summary_text, phase=phase)


# ─── CLI ──────────────────────────────────────────────────────────────────

def _run_cli() -> int:
    if len(sys.argv) < 2:
        print("Kullanım: python src/nlp/gemini.py <report_json_path>", file=sys.stderr)
        return 2
    report_path = sys.argv[1]
    if not os.path.exists(report_path):
        print("report_path bulunamadı", file=sys.stderr)
        return 2
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    try:
        print(generate_analysis(report))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
