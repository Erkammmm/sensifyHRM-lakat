"""
FastAPI Ana Uygulama
SensifyHR Mülakat Analiz API endpoint'leri.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import subprocess
import math

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from typing import Optional, Dict, Any
import uuid
import aiofiles
from datetime import datetime

# Proje root'unu path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import InterviewAnalysisPipeline
from src.report_generator import ReportGenerator

# FastAPI uygulaması
app = FastAPI(
    title="SensifyHR Mülakat Analiz API",
    description="Multimodal AI Mülakat Değerlendirme Sistemi (Metin + Ses + Yüz)",
    version="2.0.0",
)

# Global instance'lar
pipeline = None
report_generator = None
analysis_status = {}


# =====================================================================
# Yardımcılar
# =====================================================================
def _save_upload(file: UploadFile, interview_id: str) -> str:
    """Yüklenen videoyu uploads klasörüne kaydeder."""
    allowed_ext = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen format. İzin: {', '.join(allowed_ext)}",
        )
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    return os.path.join(uploads_dir, f"{interview_id}{ext}")


def _update_status(interview_id: str, status: str, **extra: Any) -> None:
    """Analiz durumunu günceller."""
    analysis_status[interview_id] = {"status": status, **extra}


def _sanitize_for_json(obj: Any) -> Any:
    """NumPy/NaN temizleyici."""
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [_sanitize_for_json(v) for v in obj.tolist()]
        if isinstance(obj, np.generic):
            obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _strip_voice_series(data: Dict) -> Dict:
    """Gemini ve API yanıtı için büyük zaman serilerini kaldırır."""
    if not isinstance(data, dict):
        return data
    cleaned = dict(data)
    raw = cleaned.get("raw_voice_features", cleaned)
    if isinstance(raw, dict):
        raw = dict(raw)
        raw.pop("waveform", None)
        raw.pop("mel_spectrogram", None)
        raw.pop("windowed_features", None)
        for key in ("energy_rms", "pitch_f0"):
            sub = raw.get(key)
            if isinstance(sub, dict):
                sub = dict(sub)
                sub.pop("series", None)
                raw[key] = sub
        ss = raw.get("speech_silence")
        if isinstance(ss, dict):
            ss = dict(ss)
            ss.pop("speech_segments", None)
            raw["speech_silence"] = ss
        if "raw_voice_features" in cleaned:
            cleaned["raw_voice_features"] = raw
        else:
            cleaned = raw
    return cleaned


def _run_gemini(report_paths: Dict[str, str]) -> Dict[str, Any]:
    """Gemini'yi ayrı process ile çalıştırır (protobuf izolasyonu)."""
    if not report_paths.get("json"):
        return {}

    gemini_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "gemini.py")
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [sys.executable, gemini_script, report_paths["json"]],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    if result.returncode == 0:
        gemini_text = result.stdout.strip()
        return {"analysis": gemini_text, "provider": "gemini"} if gemini_text else {}
    return {"status": "error", "message": result.stderr.strip(), "provider": "gemini"}


def _run_ollama(full_report: Dict[str, Any]) -> Dict[str, Any]:
    """Yerel Ollama/Gemma ile değerlendirme yapar (Gemini alternatifi/fallback)."""
    try:
        from src.ollama_ai import OllamaAI

        ai = OllamaAI()  # default: gemma3:12b

        # FAZ-3: segment signal packages üzerinden reasoning
        if (full_report.get("phase") or "").lower() == "v3":
            packages = full_report.get("segment_signal_packages", [])
            result_text = ai.evaluate_phase3(segment_signal_packages=packages)
        else:
            text_segments = full_report.get("text_analysis", {}).get("segments", [])
            audio_timeline = full_report.get("audio_emotion_analysis", {}).get("timeline", [])
            face_summary = full_report.get("face_analysis", {}).get("summary", {})
            anomalies = full_report.get("anomalies", [])
            result_text = ai.evaluate_candidate(
                text_data=text_segments,
                audio_data=audio_timeline,
                face_summary=face_summary,
                anomalies=anomalies,
            )
        result_text = (result_text or "").strip()
        return {"analysis": result_text, "provider": "ollama"} if result_text else {}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "provider": "ollama"}


def _write_ai_to_reports(report_paths: Dict[str, str], ai_analysis: Dict) -> list:
    """AI metnini JSON ve HTML raporlara ekler."""
    warns = []
    gemini_text = ai_analysis.get("analysis", "") if isinstance(ai_analysis, dict) else ""

    # JSON'a ekle
    if report_paths.get("json"):
        try:
            with open(report_paths["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            data["ai_analysis"] = ai_analysis
            with open(report_paths["json"], "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            warns.append("gemini_json_write_failed")

    # HTML'e ekle
    if report_paths.get("html") and gemini_text:
        try:
            with open(report_paths["html"], "r", encoding="utf-8") as f:
                html = f.read()
            escaped = gemini_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            paragraphs = [p.strip() for p in escaped.replace("\r\n", "\n").split("\n\n") if p.strip()]
            formatted = "</p><p>".join(p.replace("\n", "<br>") for p in paragraphs)
            block = (
                '<div class="section"><h2>🤖 AI Destekli İK Değerlendirmesi</h2>'
                f'<div class="ai-analysis"><p>{formatted}</p></div></div>'
            )
            if '<div class="footer">' in html:
                html = html.replace('<div class="footer">', f'{block}\n<div class="footer">')
            elif "</body>" in html:
                html = html.replace("</body>", f"{block}\n</body>")
            else:
                html += block
            with open(report_paths["html"], "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            warns.append("gemini_html_write_failed")

    return warns


# =====================================================================
# Startup
# =====================================================================
@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında modülleri yükle."""
    global pipeline, report_generator

    try:
        print("[API] Pipeline yükleniyor...")
        pipeline = InterviewAnalysisPipeline(phase3_enabled=True)
        print("[API] Pipeline hazır!")

        # Opsiyonel: model prewarm (ilk analizde beklemeyi azaltır)
        if os.getenv("SENSIFYHR_PREWARM_MODELS", "0").strip() == "1":
            try:
                print("[API] Prewarm: STT (faster-whisper) yükleniyor...")
                pipeline.text_analyzer._ensure_fw_model()
            except Exception as exc:
                print(f"[API] Prewarm uyarı (STT): {exc}")
            try:
                print("[API] Prewarm: HuBERT SER (audio signal) yükleniyor...")
                pipeline._get_audio_signal_fusion()
            except Exception as exc:
                print(f"[API] Prewarm uyarı (SER): {exc}")
    except Exception as e:
        print(f"[API] KRİTİK: Pipeline yüklenemedi: {e}")
        import traceback
        traceback.print_exc()
        pipeline = None

    try:
        report_generator = ReportGenerator()
        print("[API] Report Generator hazır.")
    except Exception as e:
        print(f"[API] Hata: Report Generator: {e}")
        report_generator = None


# =====================================================================
# Endpoints
# =====================================================================
@app.get("/")
async def root():
    """Ana endpoint."""
    return {
        "service": "SensifyHR Mülakat Analiz API",
        "version": "2.0.0",
        "status": "running",
        "modules": ["TextAnalyzer (Whisper+BERT)", "AudioAnalyzer (wav2vec2)",
                     "FaceAnalyzer (MediaPipe)", "VoiceAnalyzer (Librosa)"],
        "endpoints": {
            "analyze": "POST /analyze",
            "status": "GET /status/{interview_id}",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_ready": pipeline is not None,
    }

@app.get("/favicon.ico")
async def favicon():
    # Browser'lar otomatik ister; logda 404 görünmesin diye 204 döndürüyoruz.
    return Response(status_code=204)


@app.post("/analyze")
async def analyze_interview(
    file: UploadFile = File(..., description="Video dosyası (MP4, AVI, MOV, MKV, WEBM)"),
    interview_id: Optional[str] = None,
    phase3: bool = True,
    llm_provider: str = "ollama",
):
    """
    Video yükle → analiz et → JSON + HTML rapor döndür.

    1. Swagger UI'da "Try it out" ile video seçin
    2. "Execute" ile gönderin
    3. JSON yanıtı + HTML/JSON rapor dosyaları oluşturulur
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline henüz hazır değil")

    if interview_id is None:
        interview_id = str(uuid.uuid4())

    video_path = _save_upload(file, interview_id)

    try:
        # 1) Dosyayı kaydet
        async with aiofiles.open(video_path, "wb") as out:
            content = await file.read()
            await out.write(content)

        _update_status(interview_id, "processing",
                       started_at=datetime.utcnow().isoformat() + "Z")

        # 2) Pipeline çalıştır (metin + ses + yüz + voice)
        full_report = pipeline.process_interview(
            video_path, interview_id, phase3_enabled=bool(phase3)
        )

        # 3) Rapor oluştur (JSON + HTML + grafikler)
        warnings_list = []
        report_paths = {}
        if report_generator:
            try:
                report_paths = report_generator.generate_report(full_report)
            except Exception as e:
                warnings_list.append(f"report_generation_failed: {e}")

        # 4) LLM değerlendirmesi (varsayılan: Gemini; kota/hatada Ollama fallback)
        provider = (llm_provider or "gemini").strip().lower()
        ai_analysis: Dict[str, Any] = {}
        if report_paths.get("json"):
            if provider == "ollama":
                ai_analysis = _run_ollama(full_report)
            elif provider == "none":
                ai_analysis = {}
            else:
                ai_analysis = _run_gemini(report_paths)
                # Gemini kota/hatada otomatik fallback
                if isinstance(ai_analysis, dict) and ai_analysis.get("status") == "error":
                    warnings_list.append(f"gemini_failed_fallback_to_ollama: {ai_analysis.get('message', '')}")
                    ai_analysis = _run_ollama(full_report)
        warnings_list.extend(_write_ai_to_reports(report_paths, ai_analysis))

        # 5) API yanıtı oluştur
        voice_clean = _strip_voice_series(full_report.get("voice_analysis", {}))

        response = {
            "interview_id": interview_id,
            "duration_seconds": full_report.get("duration_seconds", 0),
            "phase": full_report.get("phase", "v2"),
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "video_info": full_report.get("video_info", {}),
            # Özetler
            "text_analysis": full_report.get("text_analysis", {}),
            "audio_emotion_analysis": full_report.get("audio_emotion_analysis", {}),
            "audio_signal_analysis": full_report.get("audio_signal_analysis", {}),
            "visual_signal_analysis": full_report.get("visual_signal_analysis", {}),
            "face_analysis": full_report.get("face_analysis", {}),
            "voice_analysis": voice_clean,
            "anomalies": full_report.get("anomalies", []),
            "segment_signal_packages": full_report.get("segment_signal_packages", []),
            # AI
            "ai_analysis": ai_analysis,
            "llm_provider": (ai_analysis.get("provider") if isinstance(ai_analysis, dict) else provider),
            # Rapor dosyaları
            "report_files": report_paths,
            "warnings": warnings_list,
        }

        _update_status(
            interview_id, "completed",
            started_at=analysis_status[interview_id]["started_at"],
            completed_at=datetime.utcnow().isoformat() + "Z",
            report_files=report_paths,
        )

        return JSONResponse(content=_sanitize_for_json(response))

    except Exception as e:
        _update_status(
            interview_id, "failed",
            error=str(e),
            started_at=analysis_status.get(interview_id, {}).get("started_at"),
            completed_at=datetime.utcnow().isoformat() + "Z",
        )

        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

        return JSONResponse(
            content=_sanitize_for_json({
                "interview_id": interview_id,
                "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
                "partial_success": True,
                "warnings": [f"analysis_error: {e}"],
            }),
            status_code=500,
        )


@app.get("/status/{interview_id}")
async def get_analysis_status(interview_id: str):
    """Analiz durumunu sorgular."""
    status = analysis_status.get(interview_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Mülakat bulunamadı")
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
