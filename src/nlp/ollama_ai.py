"""
Ollama + Gemma AI Mülakat Değerlendirme Modülü
Yerel LLM (Ollama/Gemma) ile İK değerlendirmesi yapar.

Kullanım:
  1) Ollama'yı kurun:  https://ollama.com/download
  2) Modeli indirin:   ollama pull gemma3
  3) Servisi başlatın:  ollama serve  (arka planda çalışır)
"""

import json
import concurrent.futures
from collections import Counter
from typing import Dict, List, Optional

_LLM_TIMEOUT_SEC = 150
_LLM_FALLBACK    = "LLM analizi zaman aşımına uğradı — lütfen tekrar deneyin"
_LLM_OPTIONS     = {"temperature": 0.3}


# Varsayılan model (12B parametre - en iyi sonuç)
MODEL_NAME = "gemma3:12b"


class OllamaAI:
    """
    Ollama + Gemma ile mülakat değerlendirmesi yapar.
    Gemini'ye alternatif, tamamen yerel/offline çalışır.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        print(f"[OllamaAI] Mülakat Asistanı Başlatılıyor... Model: {model_name}")

        try:
            import ollama
            self._client = ollama
            # Bağlantı testi
            self._client.list()
            print(f"[OllamaAI] Ollama bağlantısı başarılı.")
        except ImportError:
            raise ImportError(
                "ollama paketi bulunamadı. Yükleyin: pip install ollama"
            )
        except Exception as e:
            raise ConnectionError(
                f"Ollama servisine bağlanılamadı. 'ollama serve' çalışıyor mu?\n"
                f"Hata: {e}"
            )

    def _summarize_data(
        self,
        text_data: List[Dict],
        audio_data: List[Dict],
        face_summary: Dict,
        anomalies: List[Dict],
    ) -> str:
        """Analiz verilerini İK prompt'una uygun metin özetine dönüştürür."""

        # Metin özeti
        if text_data:
            transcript = " ".join([s.get("text", "") for s in text_data])
            sentiments = [s.get("sentiment", "") for s in text_data]
            sent_counter = Counter(sentiments)
            text_section = (
                f'Transkript: "{transcript[:2000]}"\n'
                f"Duygu Dağılımı: {dict(sent_counter)}"
            )
        else:
            text_section = "Metin verisi yok."

        # Ses duygu özeti
        if audio_data:
            audio_emotions = [a.get("emotion", "") for a in audio_data]
            audio_counter = Counter(audio_emotions)
            audio_section = f"Ses Duygu Dağılımı: {dict(audio_counter)}"
        else:
            audio_section = "Ses duygu verisi yok."

        # Yüz özeti
        face_section = (
            f"Baskın Yüz Duygusu: {face_summary.get('dominant_emotion', 'Bilinmiyor')}\n"
            f"Odak Skoru: %{face_summary.get('focus_score', 0)}\n"
            f"Göz Kırpma/dk: {face_summary.get('blink_rate_per_min', 0)}"
        )

        # Anomaliler
        if anomalies:
            anomaly_lines = []
            for a in anomalies[:10]:
                anomaly_lines.append(
                    f"  - [{a['time_range']}] {a['result']} "
                    f"(Söz: {a['text_sentiment']}, Yüz: {a['face_emotion']})"
                )
            anomaly_section = "\n".join(anomaly_lines)
        else:
            anomaly_section = "Tutarsızlık tespit edilmedi."

        return (
            f"1. SÖZEL CEVAPLAR:\n{text_section}\n\n"
            f"2. SES DUYGU ANALİZİ:\n{audio_section}\n\n"
            f"3. YÜZ ANALİZİ:\n{face_section}\n\n"
            f"4. TUTARSIZLIK ANALİZİ:\n{anomaly_section}"
        )

    def evaluate_candidate(
        self,
        text_data: List[Dict],
        audio_data: List[Dict],
        face_summary: Dict,
        anomalies: List[Dict],
        job_description: str = "",
    ) -> str:
        """
        Tüm analiz verilerini AI'ya gönderip İK değerlendirmesi alır.

        Args:
            text_data: Metin analizi segmentleri
            audio_data: Ses duygu timeline'ı
            face_summary: Yüz analizi özeti
            anomalies: Tutarsızlık listesi
            job_description: İş tanımı (opsiyonel)

        Returns:
            str: AI tarafından üretilen İK değerlendirme metni
        """
        data_summary = self._summarize_data(
            text_data, audio_data, face_summary, anomalies
        )

        job_section = f"--- İŞ TANIMI ---\n{job_description}\n" if job_description else ""

        prompt = f"""Sen uzman bir İşe Alım Uzmanı (Recruiter) ve Davranış Bilimcisin.
Aşağıdaki verileri analiz ederek adayın işe uygunluğunu değerlendir.

{job_section}--- ADAY ANALİZ VERİLERİ ---
{data_summary}

--- TALİMATLAR ---
Lütfen aşağıdaki başlıklarda profesyonel ve nesnel bir İK raporu üret:

1. Genel Davranış Profili: Adayın dikkat, odak ve etkileşim seviyesi
2. İletişim Değerlendirmesi: Sözel ifade kalitesi ve tutarlılık
3. Duygusal Stabilite: Ses ve yüz verileri bazında duygusal denge
4. Tutarsızlık Analizi: Sözel-yüz uyumu, risk sinyalleri
5. Genel İK Yorumu: Kısa özet ve tavsiye

NOT: Abartı ve klinik teşhis yapma. Kesinlik ifadelerinden kaçın, "işaret etmektedir", "göstermektedir" kullan.
"""

        print(f"[OllamaAI] Değerlendirme yapılıyor ({self.model_name})...")

        response = self._client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        result = response["message"]["content"]
        print(f"[OllamaAI] Değerlendirme tamamlandı.")
        return result

    # ==================================================================
    # FAZ-3: Signal-based reasoning (emotion/sentiment yok)
    # ==================================================================
    @staticmethod
    def _chunk_list(items: List[Dict], chunk_size: int = 10) -> List[List[Dict]]:
        if not items:
            return []
        chunk_size = max(1, int(chunk_size))
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    @staticmethod
    def _select_representative(packages: List[Dict], n: int = 10) -> List[Dict]:
        """Evenly-spaced segment selection: first + last + evenly distributed middle."""
        if len(packages) <= n:
            return packages
        # Pick n evenly-spaced indices including first and last
        indices = [0]
        step = (len(packages) - 1) / (n - 1) if n > 1 else 1
        for i in range(1, n - 1):
            indices.append(round(i * step))
        indices.append(len(packages) - 1)
        # deduplicate preserving order
        seen, out = set(), []
        for idx in indices:
            pid = packages[idx].get("segment_id", idx)
            if pid not in seen:
                seen.add(pid)
                out.append(packages[idx])
        return out

    def _chat(self, messages: List[Dict]) -> str:
        """Chat call with timeout and fixed options. Returns fallback string on timeout."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._client.chat,
                model=self.model_name,
                messages=messages,
                options=_LLM_OPTIONS,
            )
            try:
                response = future.result(timeout=_LLM_TIMEOUT_SEC)
                return (response.get("message", {}) or {}).get("content", "") or ""
            except concurrent.futures.TimeoutError:
                print(f"[OllamaAI] Zaman aşımı ({_LLM_TIMEOUT_SEC}s) — yanıt bekleniyor.")
                return _LLM_FALLBACK

    def _load_phase3_prompt(self) -> str:
        """src/prompt_phase3.txt dosyasını okur."""
        import os

        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "prompt_phase3.txt")
        if not os.path.exists(path):
            # fallback: minimum prompt
            return (
                "You are NOT classifying emotions. Interpret behavioral signals in interview context. "
                "Use probabilistic language and cite timestamps."
            )
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def evaluate_phase3(
        self,
        segment_signal_packages: List[Dict],
        job_description: str = "",
        chunk_size: int = 10,
    ) -> str:
        """
        FAZ-3 LLM değerlendirmesi:
          - Segment paketlerini 10'arlı chunk'lar halinde analiz ettir
          - Sonra tek bir nihai sentez üret

        Args:
            segment_signal_packages: pipeline report["segment_signal_packages"]
            job_description: opsiyonel iş tanımı
            chunk_size: chunk boyutu (default 10)

        Returns:
            str: LLM değerlendirme metni (ürün dili)
        """
        if not segment_signal_packages:
            return "Yeterli FAZ-3 segment sinyali bulunamadı; değerlendirme üretilemedi."

        base_prompt = self._load_phase3_prompt()
        job_section = f"--- JOB DESCRIPTION ---\n{job_description}\n" if job_description else ""

        # If too many segments, select representative subset before chunking
        packages = segment_signal_packages
        if len(packages) > 15:
            packages = self._select_representative(packages, n=10)
            print(f"[OllamaAI] Segment sayısı {len(segment_signal_packages)} > 15; "
                  f"temsili {len(packages)} segment seçildi.")

        chunks = self._chunk_list(packages, chunk_size=chunk_size)
        chunk_analyses: List[str] = []

        # 1) Chunk bazlı analiz
        for idx, chunk in enumerate(chunks, start=1):
            payload = {
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "segments": chunk,
            }
            prompt = (
                f"{base_prompt}\n\n"
                f"{job_section}--- DATA (JSON) ---\n"
                f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Focus on: consistency, notable segments, and probabilistic interpretations.\n"
                "Return concise bullet points; do not output raw JSON."
            )

            text = self._chat([{"role": "user", "content": prompt}])
            if text == _LLM_FALLBACK:
                return _LLM_FALLBACK
            if text:
                chunk_analyses.append(text.strip())

        if not chunk_analyses:
            return "LLM chunk analizleri üretilemedi."

        # 2) Final sentez
        synthesis_payload = {
            "chunk_summaries": chunk_analyses[:50],  # safety cap
        }
        synthesis_prompt = (
            f"{base_prompt}\n\n"
            f"{job_section}--- CHUNK SUMMARIES ---\n"
            f"{json.dumps(synthesis_payload, ensure_ascii=False)}\n\n"
            "Now synthesize a single product-grade report using the required headings.\n"
            "Do NOT be absolute; cite key timestamps mentioned in summaries."
        )

        result = self._chat([{"role": "user", "content": synthesis_prompt}])
        return result.strip() if result != _LLM_FALLBACK else result


if __name__ == "__main__":
    try:
        ai = OllamaAI()
        print("OllamaAI modülü hazır.")
    except Exception as e:
        print(f"OllamaAI başlatılamadı: {e}")


def generate_analysis(report: Dict, job_description: str = "", chunk_size: int = 10) -> str:
    """
    Public entry: accept full pipeline report and return AI analysis text.
    Detects phase and delegates to proper evaluator.
    """
    ai = OllamaAI()
    phase = (report.get("phase") or "v2").lower()
    if phase == "v3":
        packages = report.get("segment_signal_packages", []) or []
        return ai.evaluate_phase3(packages, job_description=job_description, chunk_size=chunk_size)
    else:
        text_segments = report.get("text_analysis", {}).get("segments", [])
        audio_timeline = report.get("audio_emotion_analysis", {}).get("timeline", [])
        face_summary = report.get("face_analysis", {}).get("summary", {})
        anomalies = report.get("anomalies", [])
        return ai.evaluate_candidate(text_segments, audio_timeline, face_summary, anomalies, job_description=job_description)