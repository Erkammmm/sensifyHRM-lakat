"""
FastAPI Ana Uygulama
Mülakat analiz API endpoint'leri.
"""

import os
import sys
import json
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import uuid
import aiofiles
from datetime import datetime

# Proje root'unu path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MediaPipe'u temiz bir ortamda import et
from src.pipeline import InterviewAnalysisPipeline
from src.report_generator import ReportGenerator

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="SensifyHR Mülakat Analiz API",
    description="Online iş görüşmelerinde aday davranışlarını analiz eden AI motoru",
    version="1.0.0"
)

# Pipeline instance (global, tek seferlik yükleme)
pipeline = None
report_generator = None

# Analiz durumlarını saklamak için (production'da database kullanılmalı)
analysis_status = {}


@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında modülleri yükle."""
    global pipeline, report_generator
    
    # Pipeline (Mediapipe içerir) - ÖNCE YÜKLE
    try:
        print("[API] Pipeline yükleniyor (MediaPipe başlatılıyor)...")
        pipeline = InterviewAnalysisPipeline()
        print("[API] Pipeline hazır (MediaPipe başarıyla başlatıldı).")
    except Exception as e:
        print(f"[API] KRİTİK HATA: Pipeline yüklenemedi: {str(e)}")
        import traceback
        traceback.print_exc()
        pipeline = None
    
    # Report Generator (MediaPipe'dan sonra, bağımlılık yok)
    try:
        report_generator = ReportGenerator()
        print("[API] Report Generator hazır.")
    except Exception as e:
        print(f"[API] Hata: Report Generator yüklenemedi: {str(e)}")
        import traceback
        traceback.print_exc()
        report_generator = None


@app.get("/")
async def root():
    """Ana endpoint - API bilgileri."""
    return {
        "service": "SensifyHR Mülakat Analiz API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze - Video yükleme ve analiz + Gemini yorumu",
            "status": "GET /status/{interview_id} - Analiz durumu",
            "health": "GET /health - Sistem sağlık kontrolü"
        }
    }


@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_ready": pipeline is not None
    }


@app.post("/analyze")
async def analyze_interview(
    file: UploadFile = File(..., description="Video dosyası (MP4, AVI, MOV, MKV, WEBM)"),
    interview_id: Optional[str] = None
):
    """
    Video dosyası yükler ve analiz başlatır.
    
    **Kullanım:**
    1. Swagger UI'da "Try it out" butonuna tıkla
    2. "Choose File" ile video dosyasını seç
    3. "Execute" ile gönder
    4. Analiz tamamlandığında JSON, HTML ve PDF raporları oluşturulur
    
    **Parametreler:**
    - **file**: Yüklenecek video dosyası (MP4, AVI, MOV, MKV, WEBM formatları desteklenir)
    - **interview_id**: Opsiyonel mülakat ID (yoksa otomatik oluşturulur)
    
    **Dönen Değerler:**
    - Analiz sonuçları (frame_summary, voice_analysis)
    - Rapor dosya yolları (JSON, HTML, PDF)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline henüz hazır değil")
    
    # Dosya formatı kontrolü
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı. İzin verilen formatlar: {', '.join(allowed_extensions)}"
        )
    
    # Interview ID oluştur
    if interview_id is None:
        interview_id = str(uuid.uuid4())

    # Uploads klasörüne kaydet
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    video_file_path = os.path.join(uploads_dir, f"{interview_id}{file_extension}")
    
    try:
        # Dosyayı kaydet
        async with aiofiles.open(video_file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Analiz durumunu güncelle
        analysis_status[interview_id] = {
            "status": "processing",
            "started_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # 1. Analizi başlat (senkron, production'da async task queue kullanılmalı)
        print(f"[API] Analiz başlatılıyor: {interview_id}")
        full_report = pipeline.process_interview(video_file_path, interview_id)
        
        # 2. Rafine verileri çıkar (terminaldeki özet gibi)
        frame_summary = full_report.get("frame_summary", {})
        voice_analysis = full_report.get("voice_analysis", {})
        pyfeat_analysis = full_report.get("pyfeat_analysis", {})
        pyfeat_summary = pyfeat_analysis.get("summary", {})
        video_info = full_report.get("video_info", {})
        duration_seconds = full_report.get("duration_seconds", 0.0)
        
        # 3. AI analizi (Gemini - summary only)
        def _strip_voice_series(data: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(data, dict):
                return data
            cleaned = dict(data)
            raw = cleaned.get("raw_voice_features", cleaned)
            if isinstance(raw, dict):
                raw = dict(raw)
                raw.pop("rms_energy_series", None)
                raw.pop("pitch_series", None)
                raw.pop("pitch_histogram", None)
                if "raw_voice_features" in cleaned:
                    cleaned["raw_voice_features"] = raw
                else:
                    cleaned = raw
            return cleaned

        voice_analysis_clean = _strip_voice_series(voice_analysis)

        summary_payload = _build_summary_payload({
            "interview_id": interview_id,
            "duration_seconds": duration_seconds,
            "video_info": video_info,
            "frame_summary": frame_summary,
            "voice_analysis": voice_analysis_clean,
            "pyfeat_summary": pyfeat_summary,
        })
        ai_analysis = {}
        
        # 4. Rapor oluştur (JSON, HTML, PDF)
        if report_generator:
            frame_analysis = full_report.get("frame_analysis", [])
            report_paths = report_generator.generate_report(
                interview_id=interview_id,
                frame_analysis=frame_analysis,
                frame_summary=frame_summary,
                voice_analysis=voice_analysis,
                ai_analysis=ai_analysis,
                pyfeat_summary=pyfeat_summary,
                pyfeat_frame_analysis=pyfeat_analysis.get("frame_analysis", []),
                video_info=video_info,
                duration_seconds=duration_seconds,
            )
        else:
            report_paths = {}

        # 4.5 Gemini (ayrı process - protobuf çakışması için izolasyon)
        gemini_text = ""
        if report_paths.get("json"):
            gemini_script = os.path.join(os.path.dirname(__file__), "..", "src", "gemini.py")
            gemini_script = os.path.abspath(gemini_script)
            result = subprocess.run(
                [sys.executable, gemini_script, report_paths["json"]],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                gemini_text = result.stdout.strip()
            else:
                gemini_text = ""
                ai_analysis = {"status": "error", "message": result.stderr.strip()}

        if gemini_text:
            ai_analysis = {"analysis": gemini_text}

        # Gemini metnini HTML ve JSON rapora ekle
        if report_paths.get("json"):
            try:
                with open(report_paths["json"], "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                report_data["ai_analysis"] = ai_analysis
                with open(report_paths["json"], "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        if report_paths.get("html") and gemini_text:
            try:
                with open(report_paths["html"], "r", encoding="utf-8") as f:
                    html_content = f.read()
                escaped = (
                    gemini_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                insertion = f"<h2>AI Destekli İK Değerlendirmesi</h2><p>{escaped}</p>"
                if "<div class=\"footer\">" in html_content:
                    html_content = html_content.replace("<div class=\"footer\">", f"{insertion}\n<div class=\"footer\">")
                elif "</body>" in html_content:
                    html_content = html_content.replace("</body>", f"{insertion}\n</body>")
                else:
                    html_content += insertion
                with open(report_paths["html"], "w", encoding="utf-8") as f:
                    f.write(html_content)
            except Exception:
                pass
        
        # 5. Rafine JSON yanıtı oluştur (ham frame_analysis olmadan)
        refined_response = {
            "interview_id": interview_id,
            "duration_seconds": duration_seconds,
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "video_info": video_info,
            "frame_summary": frame_summary,
            "voice_analysis": voice_analysis_clean,
            "pyfeat_summary": pyfeat_summary,
        }
        
        # Analiz durumunu güncelle
        analysis_status[interview_id] = {
            "status": "completed",
            "started_at": analysis_status[interview_id]["started_at"],
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "report_files": report_paths,
        }
        
        return JSONResponse(content=refined_response)
        
    except Exception as e:
        # Hata durumunda durumu güncelle
        analysis_status[interview_id] = {
            "status": "failed",
            "error": str(e),
            "started_at": analysis_status.get(interview_id, {}).get("started_at"),
            "failed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Video dosyasını temizle (hata durumunda)
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
        
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


@app.get("/status/{interview_id}")
async def get_analysis_status(interview_id: str):
    """
    Analiz durumunu sorgular.
    
    Args:
        interview_id: Mülakat ID
        
    Returns:
        Analiz durumu
    """
    status = analysis_status.get(interview_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Mülakat bulunamadı")
    
    return status


def _build_summary_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "interview_id": report.get("interview_id"),
        "duration_seconds": report.get("duration_seconds"),
        "video_info": report.get("video_info", {}),
        "frame_summary": report.get("frame_summary", {}),
        "voice_analysis": report.get("voice_analysis", {}),
        "pyfeat_summary": report.get("pyfeat_summary", {}),
    }






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
