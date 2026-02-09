# SensifyHR Mülakat Analiz Sistemi v2.0
# Modüller:
#   - TextAnalyzer: Whisper STT + Türkçe BERT sentiment
#   - AudioAnalyzer: HuBERT SER ses duygu analizi
#   - FaceAnalyzer: MediaPipe blendshape yüz/bakış/göz kırpma
#   - VoiceAnalyzer: Librosa ham ses özellikleri (RMS, Pitch, VAD)
#   - InterviewAnalysisPipeline: Ana pipeline
#   - ReportGenerator: HTML + JSON rapor
#   - Gemini/Ollama AI: İK değerlendirmesi
