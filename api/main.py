"""
FastAPI Ana Uygulama
Mülakat analiz API endpoint'leri.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import os
import uuid
import aiofiles
from datetime import datetime
import sys

# Proje root'unu path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import InterviewAnalysisPipeline

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="SensifyHR Mülakat Analiz API",
    description="Online iş görüşmelerinde aday davranışlarını analiz eden AI motoru",
    version="1.0.0"
)

# Pipeline instance (global, tek seferlik yükleme)
pipeline = None

# Analiz durumlarını saklamak için (production'da database kullanılmalı)
analysis_status = {}


@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında pipeline'ı yükle."""
    global pipeline
    print("[API] Pipeline yükleniyor...")
    pipeline = InterviewAnalysisPipeline()
    print("[API] Pipeline hazır.")


@app.get("/")
async def root():
    """Ana endpoint - API bilgileri."""
    return {
        "service": "SensifyHR Mülakat Analiz API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze - Video yükleme ve analiz başlatma",
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
    file: UploadFile = File(...),
    interview_id: Optional[str] = None
):
    """
    Video dosyası yükler ve analiz başlatır.
    
    Args:
        file: Yüklenecek video dosyası
        interview_id: Opsiyonel mülakat ID (yoksa otomatik oluşturulur)
        
    Returns:
        Analiz sonuçları
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
    
    # Geçici dosya yolu
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{interview_id}{file_extension}")
    
    try:
        # Dosyayı kaydet
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Analiz durumunu güncelle
        analysis_status[interview_id] = {
            "status": "processing",
            "started_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Analizi başlat (senkron, production'da async task queue kullanılmalı)
        print(f"[API] Analiz başlatılıyor: {interview_id}")
        report = pipeline.process_interview(temp_file_path, interview_id)
        
        # Analiz durumunu güncelle
        analysis_status[interview_id] = {
            "status": "completed",
            "started_at": analysis_status[interview_id]["started_at"],
            "completed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Geçici dosyayı sil (gizlilik için)
        os.remove(temp_file_path)
        
        return JSONResponse(content=report)
        
    except Exception as e:
        # Hata durumunda durumu güncelle
        analysis_status[interview_id] = {
            "status": "failed",
            "error": str(e),
            "started_at": analysis_status.get(interview_id, {}).get("started_at"),
            "failed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Geçici dosyayı temizle
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
