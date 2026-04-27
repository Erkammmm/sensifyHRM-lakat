# SensifyHR FAZ3 — Kapsamlı Güncelleme Talebi

## Bağlam

Sen bu projeyi daha önce gördün. Projenin genel amacı: mülakat videosundan adayın duygusal ve davranışsal profilini çıkarmak ve bunu İK profesyonellerine anlaşılır bir rapor olarak sunmak.

Son çıktıları (`report_test_video_mülakat_said.json` ve `.html`) inceledik. İki ana başlıkta kapsamlı güncelleme yapman gerekiyor:

1. **Rapor ve Gemini prompt mühendisliği** — raporu sadeleştir, IK odaklı yap, teknik terimleri kaldır
2. **Göz (gaze) algoritması optimizasyonu** — videodan anlamlı göz verisi almak için yeni yaklaşım

---

## BÖLÜM 1 — Rapor Sadeleştirme ve Prompt Mühendisliği

### 1A. HTML Rapordan Kaldırılacak Bölümler

Şu bölümleri tamamen kaldır, hiçbir iz bırakma:

- **Hero card / Genel Katılım Değerlendirmesi**: 10 üzerinden puan, ses güveni skoru, yüz algılama yüzdesi, konuşma hızı metrikleri — bunlar doğru hesaplanamıyor, kaldır.
- **Tutarsızlık Sinyalleri** bölümü — tamamen kaldır, güvenilmez.
- **Konuşma Yapısı / Zaman Blokları** bölümü (sessizlik sınırlarında ayrılmış blok listesi) — gereksiz, kaldır.

### 1B. Kalan Bölümlerde Teknik Terim Temizliği

Şu teknik terimleri raporun hiçbir yerinde kullanma (ne HTML'de ne Gemini çıktısında):

- `f0`, `f0_mean`, `f0_std`, `pitch`
- `tension_score`
- `spectral_centroid`, `spectral_flatness`, `mel_energy`
- `rms_dbfs`
- `konuşma güveni` skoru (sayısal olarak — örn. "0.918")
- `hubert_valence`, `hubert_arousal`
- `voice_stress` (teknik terim olarak)
- "Hz aralığında seyretti", "monoton profil" gibi ses frekansı yorumları

Bu veriler backend'de hesaplanmaya devam edebilir ama rapora yansımasın.

### 1C. Gemini Prompt'unu Yeniden Yaz

Şu an Gemini'ye ham teknik veri gönderiyorsun ve teknik cevap alıyorsun. Prompt'u aşağıdaki yapıya göre yeniden yaz.

**Gemini'ye gönderilecek girdi — sadece şunlar:**
```xml
<mulakat_verisi>
  <transcript>
    <!-- Sadece ADAY'ın konuştuğu segmentler, zaman damgasıyla -->
    <segment timestamp="00:04 - 00:06">Merhaba ben Muhammed Said Doğmuş.</segment>
    ...
  </transcript>
  
  <duygu_zaman_cizgisi>
    <!-- Her segment için dominant_emotion ve confidence -->
    <segment timestamp="00:04 - 00:06" duygu="Neutral" guven="0.73" />
    ...
  </duygu_zaman_cizgisi>
  
  <kritik_anlar>
    <!-- Sadece is_critical_moment=true olan segmentler veya gaze_away=true olanlar -->
  </kritik_anlar>
</mulakat_verisi>
```

> **Önemli:** Mülakatçı konuşmalarını Gemini'ye gönderme. Transcript'i filtrele, sadece aday segmentlerini ver. Mülakatçı/aday ayrımını artık "az konuşan = aday" mantığıyla değil, diarization veya mevcut `speaker` etiketiyle yap. Eğer speaker etiketi güvenilmez görünüyorsa, tüm segmentleri gönder ama bunu Gemini'ye belirt.

**Gemini sistem promptu — şu yapıyı kullan:**

```
Sen bir davranışsal mülakat uzmanısın. Sana bir iş mülakatından adayın konuşma metni ve yüz duygu verileri verilecek.

Görevin: İnsan Kaynakları uzmanlarının anlayacağı, sade ve somut bir değerlendirme raporu oluşturmak.

KURALLAR:
- Asla teknik terim kullanma (f0, pitch, Hz, valence, arousal, tension score vb.)
- Her gözlemi mutlaka konuşmadan somut bir örnekle destekle
- Tahmin yürütme, sadece gözlemle ve yorumla
- "Muhtemelen", "görünüyor" gibi belirsiz ifadeler yerine net gözlem cümleleri kur
- Türkçe yaz, resmi ama anlaşılır bir dil kullan

ÇIKTI FORMATI — sadece bu 5 bölümü üret, başka bir şey ekleme:

<rapor>

<genel_izlenim>
2-3 cümle. Adayın mülakat boyunca sergilediği genel tutum ve iletişim tarzı.
</genel_izlenim>

<guclu_yanlar>
Madde madde, her madde somut bir örnekle. Maksimum 4 madde.
Format: [Güçlü yan]: [Konuşmadan somut örnek veya zaman]
</guclu_yanlar>

<gelisim_alanlari>
Madde madde, her madde somut bir örnekle. Maksimum 3 madde.
Format: [Alan]: [Gözlem]
</gelisim_alanlari>

<davranissal_uyarilar>
Eğer dikkat çekici bir davranışsal sinyal varsa yaz. Yoksa: "Bu mülakatta belirgin bir uyarı sinyali gözlemlenmedi."
Sadece yüz duygu verisiyle destekleyebildiğin gözlemleri yaz.
</davranissal_uyarilar>

<ik_icin_sorular>
Bu adaya sorulması önerilen 3 adet takip sorusu. Her soru, yukarıdaki gözlemlerden birine dayansın.
</ik_icin_sorular>

</rapor>
```

### 1D. HTML Rapor Yeniden Yapısı

Rapor şu bölümlerden oluşsun, bu sırayla:

1. **Header** — Aday adı, mülakat tarihi, süre (bunlar kalabilir)
2. **Genel İzlenim** — Gemini çıktısından, büyük ve okunabilir kart
3. **Güçlü Yanlar & Gelişim Alanları** — yan yana iki kolon
4. **Duygusal Seyir** — zaman çizelgesi üzerinde duygu grafiği (bu kısım güzel, koru, sadece teknik label'ları Türkçeleştir: "Neutral" → "Sakin/Nötr", "Happy" → "Pozitif" vb.)
5. **Konuşma İçeriği Analizi (Zaman Bloğu Bazlı)** — bu kısım güzel, koru
6. **Davranışsal Uyarılar** — varsa göster, yoksa gösterme
7. **İK İçin Önerilen Sorular** — her zaman göster

---

## BÖLÜM 2 — Göz (Gaze) Algoritması Optimizasyonu

### Mevcut Sorun

JSON çıktısında tüm segmentlerin `avg_gaze_yaw` değeri -21 ile -23 arasında sabit kalıyor. Bu, gerçek göz hareketini değil, kamera açısı veya ekran boyutu kaynaklı bir offset'i yansıtıyor. Mutlak yaw değeriyle "sağa/sola bakıyor" demek anlamsız.

Ek olarak: video yüklendiğinde algoritma canlı kamerada verdiği kaliteli sonuçları veremiyor. Olası sebepler: telefon videosu (dikey, 480x864), büyük monitör karşısında oturma, farklı FPS.

### Yeni Yaklaşım — Delta / Değişim Tabanlı Gaze Analizi

Mutlak değer yerine **değişim** ve **süre** odaklı bir mantık kur:

```python
# Parametre önerileri — bunları config'e al, sabit kodlama
GAZE_AWAY_YAW_DELTA = 15      # derece — baseline'dan bu kadar sapma = "bakış kaydı"
GAZE_AWAY_PITCH_DELTA = 12    # derece — yukarı/aşağı için
GAZE_AWAY_MIN_DURATION = 6.0  # saniye — bu kadar süre devam ederse kayıt et
GAZE_BASELINE_WINDOW = 30     # saniye — ilk N saniyeden baseline hesapla

# Mantık:
# 1. İlk GAZE_BASELINE_WINDOW saniyedeki yaw/pitch ortalamasını baseline olarak al
# 2. Her frame için: abs(frame_yaw - baseline_yaw) > GAZE_AWAY_YAW_DELTA ise "potansiyel kayma"
# 3. Potansiyel kayma GAZE_AWAY_MIN_DURATION saniye kesintisiz devam ederse → gaze_away=True olarak işaretle
# 4. Bu anı JSON'a timestamp ile kaydet: {"start": X, "end": Y, "direction": "sol/sağ/yukarı/aşağı"}
```

### Telefon Videosu Uyumu

Video dikey (480x864) ve telefon kaynaklı olabilir. Buna göre:

- FPS normalize et: video FPS ile frame sampling arasında uyumsuzluk varsa uyar
- Yüz algılama confidence threshold'unu videodan okurken biraz düşür (canlı kameradaki değerin %80'i)
- Eğer video genişliği < yüksekliği ise (dikey video) bunu logla ve gaze threshold'larını buna göre ayarla (dikey videoda yaw açıları daha belirgin kayar)

### Gaze Çıktısı JSON Formatı

Mevcut segment bazlı `gaze_direction: "center"` yerine şunu ekle:

```json
"gaze_analysis": {
  "baseline_yaw": -22.1,
  "baseline_pitch": -0.6,
  "gaze_away_events": [
    {
      "start": 45.2,
      "end": 52.8,
      "duration_seconds": 7.6,
      "direction": "sağ",
      "interpretation": "Ekran dışına uzun süreli bakış — muhtemelen not okuyor veya dikkat dağıldı"
    }
  ],
  "total_gaze_away_seconds": 7.6,
  "gaze_away_percentage": 5.5,
  "data_quality": "telefon_videosu"  // "yuksek" | "orta" | "telefon_videosu" | "dusuk"
}
```

Eğer hiç gaze_away_event yoksa HTML raporda "Mülakat boyunca kameraya odaklanma sürekliydi" yaz.

---

## BÖLÜM 3 — Mülakatçı/Aday Ayırma Mantığını Kaldır

"Az konuşan = aday" mantığını tamamen sil. Bu hiç çalışmadı.

Şu an `speaker` alanı zaten JSON'da var (bazı segmentlerde "Aday", bazılarında "Mülakatçı"). Bu etiket nereden geliyorsa (diarization, manuel, vs.) onu kullan. Eğer speaker etiketi yoksa veya güvenilmez görünüyorsa, tüm segmentleri "Aday" olarak işle ve bunu raporda belirt: "Konuşmacı ayrımı yapılamadı, tüm konuşma analiz edildi."

Ses analizi de dahil — RMS, f0 hesaplamalarını artık "az konuşan kişi" filtresine göre değil doğrudan tüm timeline'a veya `speaker=Aday` segmentlerine göre yap.

---

## Özet — Ne Değişmeyecek

- Yüz duygu analizi pipeline'ı (uniface) — dokunma, çalışıyor
- Transcript + whisper pipeline — dokunma
- Gemini entegrasyonu — sadece prompt ve girdi değişecek
- JSON'un genel yapısı — `segment_signal_packages` ve `time_blocks` kalacak, sadece içerik sadeleşecek
- HTML'in dark mode tasarımı ve Chart.js grafikleri — korunacak

---

## Test Kriteri

Güncelleme sonrası `report_test_video_mülakat_said` videosuyla yeniden rapor üret. Beklentiler:

1. HTML raporda hiçbir yerde f0, pitch, tension_score, rms, hubert gibi terimler geçmemeli
2. Hero kart (10/10 puanı) görünmemeli
3. Gemini çıktısı `<rapor>` XML tag'leriyle yapılandırılmış gelmeli ve parse edilmeli
4. Gaze analizi delta bazlı çalışmalı, baseline hesaplanmalı
5. "Mülakatçı/aday ayırma" ile ilgili hiçbir log veya kod bloğu kalmamalı