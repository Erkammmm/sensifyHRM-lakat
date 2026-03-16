# SensifyHR FAZ-4 — Fix Brief V2
# Üç Kritik Düzeltme + İyileştirmeler

---

## Analiz: Ne Gördük, Ne Yanlış

### Gaze — Neden Hala Yanlış
24 saniyelik test videosundan gaze istatistikleri:
- Pitch: min=-40.7°  max=+4.1°  mean=-10.5°  stdev=13.1°
- Yaw:   min=-6.1°   max=+38.5°  mean=+8.2°   stdev=10.7°

Kişi kameraya bakıyor ama pitch mean=-10.5° (negatif!).
Önceki videoda mean=+11.76° idi — her video farklı kamera açısı.
Normalizasyon doğru ama eşik hala sorunlu:
- pitch threshold=8° → mean=-10.5° videosunda normalizasyon sonrası
  -10.5 - (-10.5) = 0° olmalı ama yine de "up" çıkıyor
- Yaw mean=+8.2° → yaw threshold=12° → 38.5° peak var ama ortalamalar hep 12° altında
- Sonuç: tüm segmentler gaze_away=True, %0 kamera odağı

### HuBERT — Referans Kullanım vs Bizim Kullanım
Referans kod (doğru kullanım):
```python
classifier = pipeline("audio-classification", model=model_id)
prediction = classifier(speech[0].numpy())
# Çıktı: [{'label': 'sadness', 'score': 0.85}, ...]
# Label isimleri: sadness, neutral, happy, angry (sadness — not "sad"!)
```

Bizim şu anki kullanım: custom remap + AutoModelForAudioClassification
Model 4 sınıf: sadness, neutral, happy, angry
Mevcut kodda label mapping muhtemelen yanlış ("sad" vs "sadness")

Bu nedenle: HuBERT'i pipeline() ile baştan yaz — basit, doğru, referansa uygun.

### Ollama Timeout — Neden
max_tokens=800 ayarından ÖNCE Ollama çalışıyordu.
Gemma3:12b ile max_tokens=800 bazı durumlarda modeli yarıda kesiyor
ve model tamamlanmamış JSON/metin üretince timeout gibi davranıyor.
Çözüm: max_tokens kaldır (unlimited bırak), temperature=0.3 kalsın.

---

## FIX A — Gaze: Per-Video Normalizasyon + Doğru Eşikler

### Dosya: src/face/contextual_aggregator.py (veya neredeyse)

Mevcut sorun: pitch_center hesaplıyoruz ama eşikler (8°/12°) hala çok dar.
Ayrıca segment ortalaması yerine per-frame mode kullanıyoruz — bu doğru.

Yapılacak değişiklik:
1. Pitch threshold: 8° → 18°
2. Yaw threshold: 12° → 18°  
3. Non-center frame oranı: 35% → 40% (daha az hassas)
4. Pitch normalizasyon: ZATEN VAR — dokunma

Neden 18°: stdev=13.1° olan bir videoda 8° eşik çok gürültülü.
18° = mean + ~0.6 stdev → gerçek sapmaları yakalar, gürültüyü geçmez.

Test sonrası beklenen: 24s videoda en az %60 center gaze (kişi kameraya bakıyor).

---

## FIX B — HuBERT: pipeline() ile Yeniden Yaz

### Dosya: src/audio/audio_signal_fusion.py

Mevcut kod: AutoModelForAudioClassification + custom remap + valence/arousal mapping
Sorun: label isimleri yanlış match ediyor olabilir ("sad" vs "sadness")

Yeni implementasyon — referans koda uygun, basit:

```python
from transformers import pipeline as hf_pipeline

# Model yüklenirken (init'te):
self._ser_pipeline = hf_pipeline(
    "audio-classification",
    model="SeaBenSea/hubert-large-turkish-speech-emotion-recognition",
    device=0 if torch.cuda.is_available() else -1
)

# Her chunk için:
def _run_ser(self, audio_chunk_numpy: np.ndarray) -> dict:
    """
    audio_chunk_numpy: float32, shape (N,), sample_rate=16000
    Returns: {'top_label': str, 'top_score': float, 'all_scores': dict}
    """
    result = self._ser_pipeline(audio_chunk_numpy)
    # result = [{'label': 'sadness', 'score': 0.85}, {'label': 'neutral', 'score': 0.10}, ...]
    
    top = result[0]
    all_scores = {r['label']: round(r['score'], 3) for r in result}
    
    # Normalize label names for display
    label_map = {
        'sadness': 'sad',
        'neutral': 'neutral', 
        'happy': 'happy',
        'angry': 'angry'
    }
    top_label = label_map.get(top['label'], top['label'])
    
    return {
        'top_label': top_label,
        'top_score': round(top['score'], 3),
        'all_scores': {label_map.get(k, k): v for k, v in all_scores.items()}
    }
```

Valence/arousal mapping (KORU — iyi çalışıyor):
```python
_LABEL_TO_SIGNAL = {
    'sad':     {'valence': -0.8, 'arousal': -0.4},
    'angry':   {'valence': -0.9, 'arousal':  0.9},
    'happy':   {'valence':  1.0, 'arousal':  0.6},
    'neutral': {'valence':  0.0, 'arousal':  0.0},
}
```

ÖNEMLI: Sessiz chunk'ları (rms < -45 dBFS veya chunk tamamen sıfır) SER'den hariç tut.
JSON'da gördük: [24-27s] chunk score=0.0 — bu sessizlik, dağılıma dahil etme.

Debug alanında saklama formatı:
```python
debug = {
    'ser_top_label': top_label,       # 'sad'/'happy'/'neutral'/'angry'
    'ser_top_score': top_score,
    'ser_all_scores': all_scores,     # {'sad': 0.85, 'neutral': 0.10, ...}
    'valence_score_raw': valence,
    'arousal_score_raw': arousal,
    ...
}
```

---

## FIX C — Ollama: max_tokens Kaldır

### Dosya: src/nlp/ollama_ai.py

Değişiklik:
- max_tokens / num_predict: KALDIR (unlimited)
- temperature: 0.3 kalsın
- timeout: 150s yap (90s'den artır)
- Segment truncation (>10 → 5 temsilci): KORU

---

## İYİLEŞTİRME D — Dashboard: ser_all_scores ile Daha Zengin Dağılım

Şu an: her chunk için sadece top_label sayıyoruz
Yeni: ser_all_scores varsa, weighted distribution hesapla

```python
# report_generator.py - _compute_voice_emotion_distribution
# Eğer debug.ser_all_scores varsa: her label'ın score'unu topla
# Eğer yoksa: top_label say (eski yöntem, fallback)

def _compute_voice_emotion_distribution(audio_timeline):
    score_totals = {'sad': 0.0, 'happy': 0.0, 'neutral': 0.0, 'angry': 0.0}
    valid_chunks = 0
    for chunk in audio_timeline:
        dbg = chunk.get('debug', {})
        all_scores = dbg.get('ser_all_scores')
        top_score = dbg.get('ser_top_score', 0)
        if top_score == 0.0:  # sessiz chunk - atla
            continue
        if all_scores:
            for label, score in all_scores.items():
                if label in score_totals:
                    score_totals[label] += score
            valid_chunks += 1
        else:
            # fallback: sadece top_label
            top = dbg.get('ser_top_label', '')
            if top in score_totals:
                score_totals[top] += 1
            valid_chunks += 1
    # normalize to %
    ...
```

---

## İYİLEŞTİRME E — JSON'dan Dashboard'a: Thought Units Göster

JSON'da güzel bir alan var: text_analysis.thought_units
Her thought unit = bir fikir bloğu (başlangıç/bitiş/metin)

Dashboard'a ekle: "Konuşma Yapısı" küçük kartı
- Her thought unit'i timeline'da bir blok olarak göster
- Altında metin snippet (ilk 60 karakter)
- Segment 1, 2, 3... olarak numarala
- Bu LLM'e de daha iyi bağlam sağlar

report_generator.py'de text_analysis.thought_units'i çek,
template'e geç, HTML'de compact liste olarak göster.

---

## Çalışma Sırası

**ADIM 1 — Fix C (Ollama):** En kolay, hemen yap.
ollama_ai.py → max_tokens/num_predict satırını sil, timeout=150s yap.
Test: uvicorn başlat, kısa video gönder, LLM cevap veriyor mu kontrol et.
Dur ve bekle.

**ADIM 2 — Fix B (HuBERT pipeline):**
audio_signal_fusion.py → HuBERT kısmını pipeline() ile yeniden yaz.
MEVCUT valence/arousal mapping ve debug field yapısını koru.
ser_all_scores field'ı ekle.
Sessiz chunk filter ekle (top_score == 0.0 veya rms < -45 dBFS).
Test: print her chunk'un ser_top_label ve ser_all_scores değerlerini.
Dur ve bekle.

**ADIM 3 — Fix A (Gaze eşikleri):**
contextual_aggregator.py → pitch=18°, yaw=18°, non-center oranı=%40.
Test: Test_erkam.mp4 üzerinde per-segment gaze_direction yazdır.
24s videoda en az %50 center bekliyoruz.
Dur ve bekle.

**ADIM 4 — İyileştirme D (Weighted HuBERT dağılımı):**
report_generator.py → ser_all_scores varsa weighted distribution kullan.
Dur ve bekle.

**ADIM 5 — İyileştirme E (Thought Units):**
report_generator.py + report_v3.html → thought_units listesi ekle.
Dur ve bekle.

**ADIM 6 — Final test:**
uvicorn → 3 farklı video test et (kısa/orta/uzun).
Raporları kontrol et. Dur ve bekle.

---

## Kesinlikle Dokunma
- text_analyzer.py
- thought_unit_merger.py
- gemini.py
- api/main.py
- test_example.py
- plot.py (mevcut matplotlib chartlar çalışıyor)