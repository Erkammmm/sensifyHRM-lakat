"""
FastAPI Ana Uygulama
Mülakat analiz API endpoint'leri.
"""

import os
import sys
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
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
        ai_analysis = {"note": "Gemini analizi yapılamadı."}
        try:
            summary_payload = _build_summary_payload({
                "interview_id": interview_id,
                "duration_seconds": duration_seconds,
                "video_info": video_info,
                "frame_summary": frame_summary,
                "voice_analysis": voice_analysis,
                "pyfeat_summary": pyfeat_summary,
            })
            prompt_text = _load_prompt()
            if not prompt_text:
                raise RuntimeError("Prompt dosyası boş veya bulunamadı.")
            full_prompt = f"{prompt_text}\n\nVERI:\n{json.dumps(summary_payload, ensure_ascii=False)}"
            result = _call_gemini(full_prompt)
            ai_analysis = {
                "analysis": result.get("text", ""),
                "model": result.get("model"),
            }
        except Exception as e:
            ai_analysis = {
                "status": "error",
                "message": str(e),
            }
        
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
        
        # 5. Rafine JSON yanıtı oluştur (ham frame_analysis olmadan)
        refined_response = {
            "interview_id": interview_id,
            "duration_seconds": duration_seconds,
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "video_info": video_info,
            "frame_summary": frame_summary,
            "voice_analysis": voice_analysis,
            "pyfeat_summary": pyfeat_summary,
            "ai_analysis": ai_analysis,
            "report_files": report_paths,
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


def _load_prompt() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "src", "prompt.txt"),
        os.path.join(base_dir, "src", "prompt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    return ""


def _build_summary_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "interview_id": report.get("interview_id"),
        "duration_seconds": report.get("duration_seconds"),
        "video_info": report.get("video_info", {}),
        "frame_summary": report.get("frame_summary", {}),
        "voice_analysis": report.get("voice_analysis", {}),
        "pyfeat_summary": report.get("pyfeat_summary", {}),
    }


def _call_gemini(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro").strip()
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai kütüphanesi bulunamadı.") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return {"model": model, "text": getattr(response, "text", "")}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
