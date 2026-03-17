# SensifyHR FAZ-4 — SER Replacement + Akıllı Paragraf Brief'i
# Model: claude-sonnet-4-6
# İzin almadan araştır, karar ver, uygula

---

## Bağlam

Mevcut sistemde sesten duygu için hazır SER modeli kullanıyoruz.
Denedik: SeaBenSea HuBERT → random çıktı, firdhokk wav2vec2 → angry dominant,
ehcalabres wav2vec2 → sad dominant. Hiçbiri Türkçe için güvenilir değil.

Zaten pipeline'da torchaudio ile üretilen mükemmel ses sinyalleri var:
- f0_mean, f0_std (pitch ve varyasyon)
- rms_dbfs (ses enerjisi)
- speech_confidence (konuşma netliği)
- speech_style, konusma_enerjisi (zaten hesaplanmış)

Bu verilerden kural bazlı ses profili üretmek hem dil bağımsız hem güvenilir.

---

## OPTİMİZASYON 1: SER Modelini Kaldır, torchaudio Sinyalleriyle Değiştir

### Ne Kalacak (Dokunma)
- audio_signal_fusion.py'daki valence/arousal pipeline mantığı → KORU
- voice_analyzer.py → DOKUNMA
- Dashboard yapısı → KORU (sadece içerik değişecek)
- segment_signal_packages'taki hubert_valence/arousal alanları → KORU (isim değişmez)

### Ne Değişecek

**src/audio/audio_signal_fusion.py:**
- ehcalabres pipeline() çağrısını ve SER model yüklemesini tamamen kaldır
- Valence/arousal artık f0 + rms tabanlı kural sisteminden gelecek

Önce voice_analyzer.py çıktılarındaki gerçek f0_mean, f0_std, rms_dbfs aralıklarını kontrol et.
Sonra eşikleri gerçek verilere göre kalibre et.

Kural sistemi mantığı:
```
Enerji (rms_dbfs):
  > -25 dBFS  → yüksek
  > -40 dBFS  → orta
  <= -40 dBFS → düşük

Pitch varyasyonu (f0_std):
  > 50 Hz → yüksek
  > 20 Hz → orta
  <= 20 Hz → düşük

Profil kuralları:
  yüksek enerji + yüksek varyasyon → "Canlı"    (valence=+0.6, arousal=+0.7)
  yüksek enerji + orta/düşük var  → "Kararlı"   (valence=+0.3, arousal=+0.5)
  orta enerji + yüksek varyasyon  → "Gergin"    (valence=-0.4, arousal=+0.6)
  orta enerji + orta varyasyon    → "Dengeli"   (valence=+0.1, arousal=+0.2)
  düşük/orta enerji + düşük var   → "Sakin"     (valence=-0.1, arousal=-0.2)
```

Debug alanında şunları sakla:
- voice_profile: "Canlı"/"Kararlı"/"Gergin"/"Sakin"/"Dengeli"
- voice_energy: "high"/"medium"/"low"
- voice_variation: "high"/"medium"/"low"
- valence_score_raw, arousal_score_raw (hesaplandığı gibi)

**src/reporting/report_generator.py:**
- _compute_voice_emotion_distribution → ses profili dağılımını hesapla
- Renk mapping: Canlı=yeşil, Kararlı=mavi, Dengeli=gri, Sakin=açık mavi, Gergin=turuncu

**src/reporting/templates/report_v3.html:**
- "Ses Duygu Dağılımı (HuBERT SER)" → "Ses Profili Dağılımı"
- Subtitle: "torchaudio · f0 + enerji tabanlı · dil bağımsız"

**src/nlp/prompt_phase3.txt:**
- SER açıklamalarını güncelle
- Yeni alan ekle: voice_profile → "Canlı|Kararlı|Gergin|Sakin|Dengeli (f0+enerji tabanlı)"

---

## OPTİMİZASYON 2: Akıllı Paragraf Bölme

### Mevcut Sorun
Zaman blokları mekanik: her 1-2-3 dakikada kes.
Sonuç: cümle ortasından kesilme, "Tamam." gibi tek kelime blokları.

### İstenen: Sessizlik + Kelime Sayısı Bazlı Bölme

Whisper segmentleri arasındaki boşlukları kullan.

**Bölme kuralları (öncelik sırası):**
1. Ardışık iki segment arası >2 saniyelik boşluk VE blokta >=50 kelime → kes
2. Blokta >=150 kelime → bir sonraki boşlukta kes
3. Blok süresi >=3 dakika → zorla kes
4. Son segment → zorla kes
5. Minimum: blok <20 kelimeyse → bir sonrakiyle birleştir

**contextual_aggregator.py:**
- build_time_blocks() fonksiyonunu build_smart_blocks() ile değiştir
- Yukarıdaki bölme mantığını uygula
- Her blok için: başlangıç/bitiş zamanı, birleşik metin, kelime sayısı, sinyal ortalamaları

**Dashboard gösterimi aynı kalır** — bloklar sadece daha anlamlı olacak.

**prompt_phase3.txt'e ekle:**
"Her konuşma bloğu doğal sessizlik sınırlarında ayrılmıştır — her blok bir düşünce bütününü temsil eder."

---

## Test Kriterleri

**Ses profili (sadievrenhocamız videosu — konferans tonu):**
- "Sakin" veya "Dengeli" dominant olmalı
- "Canlı" veya "Kararlı" az olabilir
- "Gergin" minimum olmalı
- Kesinlikle "sad"/"angry" gibi duygu etiketi olmamalı

**Akıllı paragraflar:**
- Bloklar cümle ortasında kesilmemeli
- Her blok 50-150 kelime arası
- "Tamam." gibi tek kelime blokları olmamalı
- Ortalama blok süresi 30-90 saniye arası

---

## Çalışma Sırası

1. voice_analyzer.py çıktılarındaki gerçek f0/rms aralıklarını kontrol et
2. Optimizasyon 1 uygula — SER kaldır, voice profile sistemi yaz
3. Test: ses profili dağılımını yazdır — mantıklı mı?
4. Optimizasyon 2 uygula — akıllı paragraf bölme
5. Test: blok içerikleri anlamlı mı?
6. sadievrenhocamız videosu üzerinde final test
7. Özet yaz

---

## Kesinlikle Dokunma
- src/audio/text_analyzer.py
- src/audio/thought_unit_merger.py
- test_example.py
- api/main.py (zorunluysa minimal değişiklik, not bırak)