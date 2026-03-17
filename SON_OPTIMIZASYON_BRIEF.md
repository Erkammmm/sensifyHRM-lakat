# SensifyHR FAZ-4 — Son Optimizasyon Brief'i
# Model: claude-opus-4-6
# Bu brief'ten sonra git commit + push yapılacak

---

## Mevcut Durum Analizi

Referans video: 2 dakikalık teknik mülakat (stajyer, DST yazılımı)
- Göz Teması: Orta %64 — makul
- Ses Güveni: Kararlı 0.86 — doğru
- Duygusal Denge: Dengeli %19.4 olumsuz — makul
- Baskın Duygu: Neutral %66.2 — doğru
- LLM çıktısı: 4496 karakter, Gemini Flash — çalışıyor ✅

Sorunlar aşağıda.

---

## SORUN 1: SER Modeli Yanlış Sonuç Veriyor

### Kanıt (JSON'dan)
```
angry: 17 chunk, fearful: 10 chunk, surprised: 5, happy: 1
```
Teknik mülakattaki sakin stajyer için angry %51 — bu tamamen yanlış.
firdhokk/wav2vec2-xlsr-53 prosodi (enerji/pitch) tabanlı — Türkçe için kalibre edilmemiş.
Enerjik konuşma = angry olarak etiketliyor.

### Görevin
1. HuggingFace'de Türkçe SER modellerini araştır:
   - https://huggingface.co/models?language=tr&pipeline_tag=audio-classification&sort=downloads
   - Özellikle ara: "turkish speech emotion", "türkçe duygu", "emotion recognition turkish"
2. GitHub'da araştır: "Turkish speech emotion recognition"
3. Değerlendirme kriterleri:
   - Türkçe üzerinde eğitilmiş veya fine-tuned
   - pipeline() ile çalışıyor
   - GPU destekli
   - Makul sınıf sayısı (4-8 arası)
   - HuggingFace'de indirme sayısı yüksek (güvenilirlik)
4. Eğer iyi bir Türkçe model yoksa:
   - Çok dilli + Türkçe içeren model seç
   - VEYA: dil-bağımsız prosodi modelini değil, içerik-bağımsız spektral model seç
5. Seçtiğin modeli test et — aynı audio üzerinde çalıştır, sonuçları göster
6. Kararını gerekçeyle yaz, uygula
7. Dashboard'da model ismini güncelle (şu an "SeaBenSea/hubert-large-turkish" yazıyor — yanlış)
   report_v3.html'de SER subtitle'ını seçtiğin modelin gerçek adıyla güncelle

---

## SORUN 2: Yüz Duygusu Modeli — Araştır ve Karar Ver

### Mevcut
UniFace DDAMFN AffectNet7 kullanıyoruz — 7 sınıf, iyi çalışıyor.

### Bilgi
UniFace'in EfficientNet-B7 tabanlı modeli daha yüksek doğruluk veriyor.
Bkz: https://github.com/yakhyo/uniface — model seçeneklerine bak.
Mevcut: DDAMFN
Alternatif: EfficientNet-B7 veya benzeri daha güçlü backbone

### Görevin
1. face_analyzer.py'ı oku — hangi UniFace modelini kullanıyor?
2. https://github.com/yakhyo/uniface adresini araştır — mevcut model seçenekleri neler?
3. Daha iyi model varsa ve GPU memory izin veriyorsa değiştir
4. Değiştirmeden önce basit test yap — aynı video üzerinde her iki modeli karşılaştır
5. Kararını gerekçeyle yaz
6. Eğer mevcut model zaten en iyisiyse "mevcut model yeterli" yazıp geç

---

## SORUN 3: Konuşma Yapısı — Paragraf Bazlı Yeniden Tasarım

### Mevcut sorun
Thought units çok küçük parçalar: "Tamam." (0.1s), "Neden bu işe başvurdunuz?" (2s)
Bunlar anlamlı değil. IK uzmanı için değersiz.

### İstenen: Zaman Bloğu Bazlı Paragraflar

Thought units yerine videoyu anlamlı zaman bloklarına böl:
- Video <= 3 dakika → 1'er dakikalık bloklar
- Video 3-10 dakika → 2'şer dakikalık bloklar  
- Video > 10 dakika → 3'er dakikalık bloklar

Her blok için:
```python
{
    "block_id": 1,
    "label": "0:00 - 1:00",
    "text": "Tüm bu bloktaki Whisper segment metinleri birleştirilmiş",
    "segment_count": 3,           # bu bloktaki Whisper segment sayısı
    "dominant_emotion": "Neutral", # bu blokta baskın yüz duygusu
    "avg_speech_confidence": 0.82,
    "avg_gaze_away_pct": 0.15,
    "hubert_valence_mean": -0.35,
    "word_count": 45
}
```

### Dashboard'da Gösterim
Her blok için bir kart:
- Üstte: "0:00 - 1:00" etiketi
- Ortada: metin (max 200 karakter, fazlası "..." ile kes)
- Altta: küçük ikonlarla göstergeler: [Duygu: Neutral] [Güven: 0.82] [Göz: %85 kamera]

### LLM Prompt Güncellemesi
prompt_phase3.txt'e ekle:
Her blok için ayrı behavioral yorum yap.
Format:
```
## [0:00 - 1:00] Blok 1 Analizi
[Bu bloktaki metin: ...]
Davranışsal gözlem: ...
```

Bu şekilde LLM hem genel analiz hem blok bazlı yorum yapacak.

### Implementasyon
1. contextual_aggregator.py'a yeni fonksiyon ekle: build_time_blocks(text_analysis, face_timeline, audio_signal_timeline, duration)
2. report_generator.py'a time_blocks'u al, template'e geç
3. report_v3.html'de "Konuşma Yapısı" bölümünü time_block kartlarıyla güncelle
4. prompt_phase3.txt'e blok bazlı analiz talimatı ekle

---

## SORUN 4: Dashboard'da Model İsmi Güncelle

report_v3.html'de şu an yanlış model isimleri yazıyor:
- SER subtitle: "SeaBenSea/hubert-large-turkish-speech-emotion-recognition" → seçtiğin yeni modelin adı
- Yüz duygu subtitle: zaten doğru (UniFace DDAMFN) — ama eğer model değiştirdiysen güncelle

---

## Çalışma Sırası

1. SER model araştırması + değiştirme (en kritik)
2. Yüz duygu modeli araştırması (karar ver, gerekirse değiştir)
3. Konuşma yapısı yeniden tasarımı
4. LLM prompt güncelleme (blok bazlı analiz)
5. Dashboard model isimleri güncelleme
6. Final test — referans video üzerinde çalıştır, sonuçları özetle

---

## Referans Dosyalar

Proje dizininde:
- report_f1f80cb5-5825-4b8d-807d-523cbbb7e36a.json (mevcut rapor — sorunlu SER çıktısı)
- report_f1f80cb5-5825-4b8d-807d-523cbbb7e36a.html (mevcut dashboard)
- kamera_hafif_sağda__ekrana_bakıyor__mutlu__motive.json (iyi örnek)
- rapor_SORUNLU-kameraya-bakmiyor-uzgun.json

JSON'daki audio_signal_analysis.timeline[].debug.ser_top_label değerlerini karşılaştırarak
SER modelinin doğruluğunu değerlendir.

---

## Kesinlikle Dokunma
- src/audio/text_analyzer.py
- src/audio/thought_unit_merger.py
- test_example.py

