"""
Gemini istemcisi.
Prompt dosyasını okuyup summary metni ile birleştirir ve metin üretir.
"""

import os
import json
import sys


def _load_prompt() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, "prompt.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_env_from_file() -> None:
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


def _build_summary_payload(report: dict) -> dict:
    return {
        "interview_id": report.get("interview_id"),
        "duration_seconds": report.get("duration_seconds"),
        "video_info": report.get("video_info", {}),
        "frame_summary": report.get("frame_summary", {}),
        "voice_analysis": report.get("voice_analysis", {}),
        "pyfeat_summary": report.get("pyfeat_summary", {}),
    }


def generate_gemini_text(summary_text: str) -> str:
    _load_env_from_file()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")

    prompt = _load_prompt()
    if not prompt:
        raise RuntimeError("prompt.txt bulunamadı veya boş.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip()
    full_prompt = f"{prompt}\n\n{summary_text}"

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(full_prompt)
    return getattr(response, "text", "")


def _run_cli() -> int:
    if len(sys.argv) < 2:
        print("Kullanım: python src/gemini.py <report_json_path>", file=sys.stderr)
        return 2
    report_path = sys.argv[1]
    if not os.path.exists(report_path):
        print("report_path bulunamadı", file=sys.stderr)
        return 2
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    summary_payload = _build_summary_payload(report)
    summary_text = json.dumps(summary_payload, ensure_ascii=False)
    try:
        text = generate_gemini_text(summary_text)
        print(text)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
