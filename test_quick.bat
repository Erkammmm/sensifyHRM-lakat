@echo off
REM Hızlı Test Scripti - Windows Batch
REM Kullanım: test_quick.bat test_video.mp4

echo ========================================
echo SensifyHR Mülakat Analiz Sistemi
echo Hızlı Test Scripti
echo ========================================
echo.

REM Anaconda environment'ı aktif et
call conda activate torch_env

REM Video dosyası kontrolü
if "%1"=="" (
    echo HATA: Video dosyası belirtilmedi!
    echo Kullanım: test_quick.bat test_video.mp4
    pause
    exit /b 1
)

if not exist "%1" (
    echo HATA: Video dosyasi bulunamadi: %1
    pause
    exit /b 1
)

echo Test videosu: %1
echo.
echo Test baslatiliyor...
echo.

REM Test scriptini çalıştır
python test_example.py %1

echo.
echo ========================================
echo Test tamamlandi!
echo ========================================
pause
