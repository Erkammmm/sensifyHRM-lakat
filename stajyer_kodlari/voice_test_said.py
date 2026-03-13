import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tkinter import Tk, filedialog


# Ana çıktı klasörü --> her analiz için bunun içine Video_1 / Video_2 açılır
kayit_klasoru = Path.home() / "out"

# Ses işleme ayarları --> tüm dosya bu örnekleme hızında analiz edilir
hedef_ornekleme_hizi = 16000
parca_suresi_sn = 1.0

# Spektral analiz ayarları --> spektrogram çözünürlüğünü belirler
fft_boyutu = 1024
kayma_boyutu = 256
mel_bant_sayisi = 64

# Frame ayarları --> RMS ve ZCR daha küçük frame'lerde ölçülür
cerceve_ms = 30
cerceve_kayma_ms = 10

# Pitch ayarları --> f0 tespiti için kullanılır
pitch_cerceve_suresi = 0.01
pitch_pencere_uzunlugu = 11
f0_min = 70.0
f0_max = 420.0

# Sessizlik / konuşma ayarları --> önce aktif ses, sonra konuşma-benzeri frame, sonra gerçek konuşma kararı verilir
min_tam_sessizlik_aktif_oran = 0.06
min_konusma_orani = 0.22
min_gercek_konusma_orani = 0.28
min_sesli_oran = 0.12
konusma_guven_esigi = 0.58

# Uzun konuşmama bölgesi --> konuşmacının en az 10 saniye konuşmadığı yerler ayrıca raporlanır
uzun_konusmama_esigi_sn = 10
gecis_penceresi_sn = 10

# Sayısal güvenlik --> log(0) ve sıfıra bölme patlamasın
eps = 1e-8
mutlak_sessizlik_rms = 1e-4
mutlak_sessizlik_peak = 5e-4


# Komut çalıştırma --> ffmpeg / ffprobe çağırmak için ortak yer
def komut_calistir(komut: List[str], yakala: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(map(str, komut)))
    try:
        return subprocess.run(
            komut,
            check=True,
            stdout=subprocess.PIPE if yakala else None,
            stderr=subprocess.PIPE if yakala else None,
            text=True,
        )
    except FileNotFoundError as hata:
        raise RuntimeError(f"Komut bulunamadı --> {komut[0]}") from hata
    except subprocess.CalledProcessError as hata:
        stderr = hata.stderr.strip() if hata.stderr else "stderr yok"
        raise RuntimeError(f"Komut başarısız --> {' '.join(map(str, komut))}\n{stderr}") from hata


# Metin kaydetme --> txt raporu utf-8 olarak yazar
def metin_kaydet(dosya_yolu: Path, metin: str):
    with open(dosya_yolu, "w", encoding="utf-8") as dosya:
        dosya.write(metin)

# Güvenli yazım --> nan / inf / None değerleri raporda bozuk görünmesin
def guvenli_yaz(deger: Any, basamak: int = 4) -> str:
    if deger is None:
        return "yok"
    try:
        sayi = float(deger)
        if math.isnan(sayi) or math.isinf(sayi):
            return "yok"
        return f"{sayi:.{basamak}f}"
    except Exception:
        return str(deger)

# Json güvenliği --> Path / Tensor / numpy tiplerini json'a uygun hale getirir
def json_guvenli(obj: Any) -> Any:
    if isinstance(obj, dict):               return {str(k): json_guvenli(v) for k, v in obj.items()}
    if isinstance(obj, list):               return [json_guvenli(v) for v in obj]
    if isinstance(obj, tuple):              return [json_guvenli(v) for v in obj]
    if isinstance(obj, Path):               return str(obj)
    if isinstance(obj, torch.Tensor):       return json_guvenli(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):         return json_guvenli(obj.tolist())
    if isinstance(obj, (np.integer,)):      return int(obj)
    if isinstance(obj, (np.floating, float)):
        sayi = float(obj)
        return None if (math.isnan(sayi) or math.isinf(sayi)) else sayi
    return obj

# Süre yazımı --> 65 sn gibi değeri 01:05 formatına çevirir
def saniyeyi_yaz(sure_sn: float) -> str:
    sure_sn = max(0, int(round(sure_sn)))
    return f"{sure_sn // 60:02d}:{sure_sn % 60:02d}"

# Yüzde --> 0-1 oranını yüzdeye çevirir
def yuzde(oran: float) -> float:
    return round(float(oran) * 100.0, 2)

# Sayım --> bir seride hangi etiket kaç kez geçmiş çıkarır
def sayim_sozlugu(seri: pd.Series) -> Dict[str, int]:
    if len(seri) == 0:
        return {}
    return {str(k): int(v) for k, v in seri.value_counts(dropna=False).to_dict().items()}

# Yüzde dağılım --> count sözlüğünü yüzdeye çevirir
def dagilim_yuzdesi(sayimlar: Dict[str, int]) -> Dict[str, float]:
    toplam = sum(sayimlar.values())
    if toplam <= 0:
        return {}
    return {k: round(v * 100.0 / toplam, 2) for k, v in sayimlar.items()}

# Baskın etiket --> en sık görülen sınıfı döndürür
def baskin_etiket(sayimlar: Dict[str, int], varsayilan: str = "belirsiz") -> str:
    return max(sayimlar.items(), key=lambda x: x[1])[0] if sayimlar else varsayilan

# Güvenli ortalama --> sayısal olmayanları atıp ortalama alır
def guvenli_ortalama(seri: pd.Series) -> float:
    degerler = pd.to_numeric(seri, errors="coerce").dropna()
    return float(degerler.mean()) if len(degerler) else float("nan")

# Dağılım metni --> yüzde sözlüğünü okunabilir metne çevirir
def dagilim_metni(yuzdeler: Dict[str, float], sira: List[str]) -> str:
    if not yuzdeler:
        return "veri yok"
    parcalar = [f"{k} %{yuzdeler[k]:.1f}" for k in sira if k in yuzdeler]
    parcalar += [f"{k} %{v:.1f}" for k, v in yuzdeler.items() if k not in sira]
    return ", ".join(parcalar)

# RMS -> dBFS --> ses gücünü log ölçekte yazmak için
def rmsi_dbfs(rms: float) -> float:
    return 20.0 * math.log10(max(float(rms), eps))

# Doğrusal 0-1 skor --> eşik kurallarını daha kademeli yapmak için
def dogrusal_skor(deger: float, alt: float, ust: float) -> float:
    if not np.isfinite(deger) or ust <= alt:
        return 0.0
    return float(np.clip((float(deger) - alt) / (ust - alt), 0.0, 1.0))

# Bool maske yumuşatma --> tek tük kopuk frame'leri toparlar
def maske_duzgunlestir(maske: np.ndarray, pencere: int = 5) -> np.ndarray:
    if maske.size == 0:
        return maske.astype(bool)
    pencere = max(1, int(pencere))
    if pencere % 2 == 0:
        pencere += 1
    cekirdek = np.ones(pencere, dtype=np.float32)
    sayi = np.convolve(maske.astype(np.float32), cekirdek, mode="same")
    return sayi >= int(math.ceil(pencere / 2.0))

# Dosya seçimi --> kullanıcıdan video veya ses dosyası seçtirir
def girdi_dosyasi_sec() -> Path:
    kok = Tk()
    kok.withdraw()
    kok.update()
    dosya_yolu = filedialog.askopenfilename(
        title="Video veya ses dosyası seçiniz..",
        filetypes=[
            ("Desteklenen medya", "*.mp4 *.mkv *.avi *.mov *.webm *.wav *.mp3 *.flac *.m4a *.aac *.ogg"),
            ("Video dosyalari", "*.mp4 *.mkv *.avi *.mov *.webm"),
            ("Ses dosyalari", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg"),
            ("Tum dosyalar", "*.*"),
        ],
    )
    kok.destroy()
    if not dosya_yolu:
        raise RuntimeError("Dosya seçilmedi..")
    return Path(dosya_yolu)

# Çıktı klasörü --> her çalışmada yeni Video_N klasörü açar
def sonraki_video_klasoru(ana_klasor: Path) -> Path:
    ana_klasor.mkdir(parents=True, exist_ok=True)
    indeks = 1
    while True:
        aday = ana_klasor / f"Video_{indeks}"
        if not aday.exists():
            aday.mkdir(parents=True, exist_ok=False)
            return aday
        indeks += 1

# Ses akışları --> ffprobe ile medyadaki audio stream bilgilerini alır
def ffprobe_ses_akislari(medya_yolu: Path) -> List[Dict[str, Any]]:
    sonuc = komut_calistir([
        "ffprobe", "-v", "error",
        "-show_streams", "-select_streams", "a",
        "-of", "json", str(medya_yolu)
    ])
    veri = json.loads(sonuc.stdout)
    return [{"index": s.get("index"),"codec_name": s.get("codec_name"),
        "sample_rate": s.get("sample_rate"),"channels": s.get("channels"),
        "channel_layout": s.get("channel_layout"),"duration": s.get("duration"),
    } for s in veri.get("streams", [])]

# Ses çıkarma --> medyadaki ilk ses akışını wav olarak dışarı alır
def sesi_cikar(medya_yolu: Path, cikti_wav: Path, hedef_hz: int):
    komut_calistir([
        "ffmpeg", "-y","-i", str(medya_yolu),
        "-map", "0:a:0","-vn",
        "-c:a", "pcm_s16le","-ar", str(hedef_hz),
        str(cikti_wav),
    ], yakala=False)


# En iyi kanal --> çok kanallı seste RMS'i en yüksek kanalı seçer
def en_iyi_kanal(dalga: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
    if dalga.ndim != 2:
        raise ValueError("dalga için yapı --> channels x samples")

    # Enerji: E = sum x[n]^2
    # RMS  ---> RMS = sqrt((1/N) * sum x[n]^2)
    kanal_rms = torch.sqrt(torch.mean(dalga ** 2, dim=1) + eps)
    kanal_peak = torch.max(torch.abs(dalga), dim=1).values

    bilgi = {
        f"kanal_{i}": {
            "rms_genlik": float(kanal_rms[i].item()),
            "peak": float(kanal_peak[i].item()),
            "rms_dbfs": float(rmsi_dbfs(float(kanal_rms[i].item()))),
        }
        for i in range(dalga.shape[0])
    }

    secilen = int(torch.argmax(kanal_rms).item())
    bilgi["secilen_kanal"] = secilen
    return secilen, bilgi


# Çerçeveleme --> 1D sinyali küçük frame'lere böler
def cercevele(x_1d: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    if x_1d.ndim != 1:
        raise ValueError("cercevele yalnızca 1D tensor bekler")
    if x_1d.numel() < frame_len:
        x_1d = torch.nn.functional.pad(x_1d, (0, frame_len - x_1d.numel()))
    return x_1d.unfold(0, frame_len, hop_len)


# Frame RMS --> kısa frame'lerde enerji düzeyini çıkarır
def cerceve_rms(x_mono: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    cerceveler = cercevele(x_mono.squeeze(0).cpu(), frame_len, hop_len)
    return torch.sqrt(torch.mean(cerceveler ** 2, dim=1) + eps)


# Frame ZCR --> kısa frame'lerde işaret değişim oranını çıkarır
def cerceve_zcr(x_mono: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    cerceveler = cercevele(x_mono.squeeze(0).cpu(), frame_len, hop_len)
    if cerceveler.shape[1] < 2:
        return torch.zeros(cerceveler.shape[0], dtype=torch.float32)
    return ((cerceveler[:, 1:] * cerceveler[:, :-1]) < 0).float().mean(dim=1)


# Parça ZCR --> 1 saniyelik blok için genel ZCR
def zcr_orani(ses_parcasi: torch.Tensor) -> float:
    ses = ses_parcasi.squeeze(0)
    if ses.numel() < 2:
        return 0.0
    return float(((ses[1:] * ses[:-1]) < 0).float().mean().item())


# Spektral özellikler --> frekans dağılımı ve mel enerji bilgisi çıkarır
def spektral_ozellikler(
    ses_parcasi: torch.Tensor,
    ornekleme_hizi: int,
    spektrogram_donusumu: torchaudio.transforms.Spectrogram,
    mel_donusumu: torchaudio.transforms.MelSpectrogram,
) -> Tuple[float, float, float]:
    # STFT: X_t[k]      --> sum x_t[n] * w[n] * exp(-j * 2*pi*k*n / N)
    # Power spectrogram --> S_t[k] = |X_t[k]|^2
    parca_spektrogrami = spektrogram_donusumu(ses_parcasi).squeeze(0)
    toplam_enerji = float(parca_spektrogrami.sum().item())
    if toplam_enerji <= eps:
        return np.nan, np.nan, 0.0

    frekanslar = torch.linspace(0, ornekleme_hizi / 2, parca_spektrogrami.shape[0], dtype=parca_spektrogrami.dtype).unsqueeze(1)
    frame_enerji = parca_spektrogrami.sum(dim=0, keepdim=True).clamp_min(eps)

    # Spectral centroid - - - >Centroid_t = sum_k f_k * S_t[k] / sum_k S_t[k]
    spektral_merkez = ((frekanslar * parca_spektrogrami).sum(dim=0, keepdim=True) / frame_enerji).mean().item()

    guvenli_spektrogram = parca_spektrogrami.clamp_min(eps)

    # Spectral flatness - -  - > Flatness_t = geometric_mean(S_t[k]) / arithmetic_mean(S_t[k])
    geometrik_ort = torch.exp(torch.log(guvenli_spektrogram).mean(dim=0))
    aritmetik_ort = guvenli_spektrogram.mean(dim=0)
    spektral_duzluk = (geometrik_ort / aritmetik_ort.clamp_min(eps)).mean().item()

    # Mel band enerjileri - - - > E_m(t) = sum_k S_t[k] * H_m[k]
    mel_spektrogram = mel_donusumu(ses_parcasi).squeeze(0)
    mel_enerjisi = float(mel_spektrogram.sum(dim=0).mean().item())

    return float(spektral_merkez), float(spektral_duzluk), float(mel_enerjisi)

# Pitch izi --> tüm kayıt boyunca temel frekansı izler
def tum_pitch_izi(x_mono: torch.Tensor, sr: int) -> torch.Tensor:
    try:
        # f0 = fs / T0
        return torchaudio.functional.detect_pitch_frequency(
            x_mono,  sample_rate=sr,   frame_time=pitch_cerceve_suresi,    win_length=pitch_pencere_uzunlugu,    freq_low=f0_min,    freq_high=f0_max,
        ).squeeze(0).cpu()
    except Exception as hata:
        print("Pitch hesaplanamadı -->", hata)
        return torch.empty(0, dtype=torch.float32)


# Parça pitch --> güvenilir frame'lerden f0 ortalama / std / sesli oranı çıkarır
def parca_pitch_istatistikleri(
    pitch_izi: torch.Tensor,
    cerceve_rms_izi: np.ndarray,cerceve_zcr_izi: np.ndarray,
    baslangic_sn: float,bitis_sn: float,
    sessizlik_rms_esigi: float,
) -> Tuple[float, float, float]:
    ust = min(int(pitch_izi.numel()), len(cerceve_rms_izi), len(cerceve_zcr_izi))
    if ust == 0:
        return np.nan, np.nan, 0.0

    bas_ind = max(0, min(int(baslangic_sn / pitch_cerceve_suresi), ust))
    bit_ind = max(bas_ind, min(int(math.ceil(bitis_sn / pitch_cerceve_suresi)), ust))
    if bit_ind <= bas_ind:
        return np.nan, np.nan, 0.0

    p = pitch_izi[bas_ind:bit_ind].numpy()
    rms_bolumu = cerceve_rms_izi[bas_ind:bit_ind]
    zcr_bolumu = cerceve_zcr_izi[bas_ind:bit_ind]

    ortak = min(len(p), len(rms_bolumu), len(zcr_bolumu))
    if ortak == 0:
        return np.nan, np.nan, 0.0

    p, rms_bolumu, zcr_bolumu = p[:ortak], rms_bolumu[:ortak], zcr_bolumu[:ortak]

    sesli_maske = (
        np.isfinite(p) &  (p >= f0_min) &
        (p <= f0_max)  &  (rms_bolumu >= max(sessizlik_rms_esigi * 1.4, mutlak_sessizlik_rms)) &
        (zcr_bolumu <= 0.18)
    )

    sesli_oran = float(np.mean(sesli_maske)) if len(sesli_maske) else 0.0
    gecerli = p[sesli_maske]

    if gecerli.size == 0:
        return np.nan, np.nan, sesli_oran

    # mean_f0 = (1/M) * sum f0_i
    # F0 standart sapması: std_f0 = sqrt((1/M) * sum (f0_i - mean_f0)^2)
    return float(np.mean(gecerli)), float(np.std(gecerli)), sesli_oran

# Ses seviyesi etiketi --> dBFS'e göre gücü sınıflar
def ses_seviyesi_etiketi(rms_dbfs_degeri: float) -> str:
    if not np.isfinite(rms_dbfs_degeri): return "belirsiz"
    if rms_dbfs_degeri < -36.0:          return "alçak"
    if rms_dbfs_degeri > -22.0:          return "yüksek"
    return "orta"

# Arka plan skoru --> flatness / zcr / centroid / düşük seviye ile 0-1 skoru üretir
def arka_plan_skoru_hesapla(flatness: float, zcr: float, centroid: float, rms_dbfs_degeri: float) -> float:
    # Arka plan skoru:
    # bg_score = 0.30 * score(flatness, 0.03, 0.11) + 0.25 * score(zcr,      0.07, 0.18) +  0.25 * score(centroid, 900, 2600) + 0.20 * score(-rms_dbfs, 36, 54)
    flatness_skoru     = dogrusal_skor(flatness, 0.03, 0.11)
    zcr_skoru          = dogrusal_skor(zcr, 0.07, 0.18)
    centroid_skoru     = dogrusal_skor(centroid, 900.0, 2600.0)
    dusuk_seviye_skoru = dogrusal_skor(-rms_dbfs_degeri, 36.0, 54.0)
    return float(np.clip(0.30 * flatness_skoru + 0.25 * zcr_skoru + 0.25 * centroid_skoru + 0.20 * dusuk_seviye_skoru, 0.0, 1.0))


# Arka plan etiketi --> sürekli gürültülü dememek için daha yumuşak etiket üretir
def arka_plan_etiketi(arka_plan_skoru: float) -> str:
    if not np.isfinite(arka_plan_skoru):        return "belirsiz"
    if arka_plan_skoru < 0.35:                  return "sessiz_arka_plan"
    if arka_plan_skoru < 0.62:                  return "hafif_arka_plan"
    return "belirgin_arka_plan"


# Konuşma enerjisi --> güç + parlaklık + ton oynaklığından 0-1 enerji skoru
def konusma_enerjisi_skoru(rms_dbfs_degeri: float, centroid: float, f0_std_degeri: float) -> float:
    # energy_score = 0.55 * score(rms_dbfs, -38, -20) + 0.20 * score(centroid, 250, 1200) + 0.25 * score(f0_std, 6,28)
    ses_skoru           = dogrusal_skor(rms_dbfs_degeri, -38.0, -20.0)
    parlaklik_skoru     = dogrusal_skor(centroid, 250.0, 1200.0)
    ton_oynaklik_skoru  = dogrusal_skor(f0_std_degeri, 6.0, 28.0)
    return float(np.clip(0.55 * ses_skoru + 0.20 * parlaklik_skoru + 0.25 * ton_oynaklik_skoru, 0.0, 1.0))


# Konuşma enerjisi etiketi --> 0-1 skoru düşük / orta / yüksek diye çevirir
def konusma_enerjisi_etiketi(enerji_skoru: float) -> str:
    if not np.isfinite(enerji_skoru):        return "belirsiz"
    if enerji_skoru < 0.33:                  return "düşük"
    if enerji_skoru > 0.67:                  return "yüksek"
    return "orta"

# Konuşma stili --> duygu demek yerine daha dürüst bir akustik özet etiketi verir
def konusma_stili_etiketi(enerji_skoru: float, arka_plan_skoru: float, f0_std_degeri: float, sesli_oran: float) -> str:
    if sesli_oran < 0.10:
        return "belirsiz"

    # Stil skorları:
    # sakin   = 0.55*(1-energy) + 0.30*temizlik + 0.15*(1-ton_hareket)
    # dengeli = 0.50*orta_enerji + 0.30*temizlik + 0.20*orta_ton_hareket
    # canlı   = 0.55*energy + 0.25*ton_hareket + 0.20*temizlik
    ton_hareket = dogrusal_skor(f0_std_degeri, 8.0, 24.0)
    temizlik    = 1.0 - arka_plan_skoru
    sakin       = 0.55 * (1.0 - enerji_skoru) + 0.30 * temizlik + 0.15 * (1.0 - ton_hareket)
    dengeli     = 0.50 * (1.0 - abs(enerji_skoru - 0.5) / 0.5) + 0.30 * temizlik + 0.20 * (1.0 - abs(ton_hareket - 0.5))
    canli       = 0.55 * enerji_skoru + 0.25 * ton_hareket + 0.20 * temizlik

    skorlar = {"sakin": sakin, "dengeli": dengeli, "canlı": canli}
    etiket, skor = max(skorlar.items(), key=lambda x: x[1])
    return etiket if skor >= 0.52 else "karışık"


# Konuşma güven skoru --> konuşma oranı + sesli oran + spektral yapı ile daha sıkı karar verir
def konusma_guven_skoru_hesapla(
    konusma_orani: float,
    sesli_oran: float,
    spektral_duzluk: float,
    spektral_merkez: float,
    rms_dbfs_degeri: float,
    arka_plan_skoru: float,
) -> float:
    # Konuşma güven skoru:
    # speech_conf =0.28 * score(konusma_orani, 0.12, 0.65) + 0.24 * score(sesli_oran,    0.08, 0.50) + 0.14 * (1 - score(flatness, 0.05, 0.16)) +
    # 0.10 * band_like_centroid_score + 0.14 * score(rms_dbfs, -42,-22) + 0.10 * (1 - arka_plan_skoru)
    konusma_skoru  = dogrusal_skor(konusma_orani, 0.12, 0.65)
    sesli_skoru    = dogrusal_skor(sesli_oran, 0.08, 0.50)
    duzluk_skoru   = 1.0 - dogrusal_skor(spektral_duzluk, 0.05, 0.16)
    merkez_skoru   = 1.0 if np.isfinite(spektral_merkez) and 120.0 <= spektral_merkez <= 2400.0 else 0.0
    seviye_skoru   = dogrusal_skor(rms_dbfs_degeri, -42.0, -22.0)
    temizlik_skoru = 1.0 - arka_plan_skoru
    skor = (        0.28 * konusma_skoru +        0.24 * sesli_skoru +        0.14 * duzluk_skoru +
                    0.10 * merkez_skoru +        0.14 * seviye_skoru +        0.10 * temizlik_skoru
    )
    return float(np.clip(skor, 0.0, 1.0))


# Konuşma oranı açıklaması --> konuşma-benzeri frame oranını yorumlar
def konusma_orani_aciklamasi(konusma_orani: float) -> str:
    if konusma_orani >= 0.85:        return "konuşma büyük ölçüde kesintisiz"
    if konusma_orani >= 0.60:        return "konuşma baskın, ama belirgin duraklamalar var"
    if konusma_orani >= 0.35:        return "konuşma aralıklı ilerliyor"
    if konusma_orani >= 0.15:        return "konuşma kısa patlamalar halinde geliyor"
    return "konuşma oldukça seyrek"


# Aktif ses açıklaması --> ortam sesi / nefes / arka plan dahil aktifliği anlatır
def aktif_ses_aciklamasi(aktif_ses_orani: float) -> str:
    if aktif_ses_orani >= 0.90:        return "arka plan dâhil aktif ses neredeyse sürekli var"
    if aktif_ses_orani >= 0.60:        return "aktif ses çoğunlukta"
    if aktif_ses_orani >= 0.30:        return "aktif ses aralıklı"
    return "aktif ses az"


# Sessizlik açıklaması --> gerçekten boş kalan bölümleri anlatır
def sessizlik_aciklamasi(sessizlik_orani: float) -> str:
    if sessizlik_orani >= 0.50:        return "uzun sessiz boşluklar var"
    if sessizlik_orani >= 0.20:        return "yer yer belirgin sessiz boşluklar var"
    if sessizlik_orani >= 0.05:        return "kısa sessiz boşluklar var"
    return "tam sessizlik çok az"


# Ses tonu açıklaması --> ortalama f0 değerini kalın / ince diye yorumlar
def ses_tonu_aciklamasi(f0_ortalama: float) -> str:
    if not np.isfinite(f0_ortalama):        return "ses tonu için yeterli veri yok"
    if f0_ortalama < 95:                    return "ses tonu oldukça kalın tarafta"
    if f0_ortalama < 120:                   return "ses tonu kalın tarafta"
    if f0_ortalama < 170:                   return "ses tonu orta bölgede"
    if f0_ortalama < 230:                   return "ses tonu ince tarafta"
    return "ses tonu oldukça ince tarafta"


# Ses tonu etiketi --> ortalama f0'ı kaba kalın / orta / ince etikete çevirir
def ses_tonu_etiketi(f0_ortalama: float) -> str:
    if not np.isfinite(f0_ortalama):        return "belirsiz"
    if f0_ortalama < 120:                   return "kalın"
    if f0_ortalama < 190:                   return "orta"
    return "ince"

# Ton değişim açıklaması --> f0 std ile iniş çıkış miktarını yorumlar
def ton_degisim_aciklamasi(f0_std_degeri: float) -> str:
    if not np.isfinite(f0_std_degeri):        return "ton değişimi için yeterli veri yok"
    if f0_std_degeri < 8:                     return "tonlama oldukça düz ve sabit"
    if f0_std_degeri < 18:                    return "tonlama hafif / orta değişken"
    return "tonlama belirgin biçimde hareketli"


# Arka plan açıklaması --> sessiz / hafif / belirgin arka plan oranlarından kısa yorum çıkarır
def arka_plan_aciklamasi(sayimlar: Dict[str, int]) -> str:
    toplam = sum(sayimlar.values())
    if toplam == 0:              return "arka plan durumu için yeterli veri yok"
    sessiz = sayimlar.get("sessiz_arka_plan", 0) / toplam
    hafif = sayimlar.get("hafif_arka_plan", 0) / toplam
    belirgin = sayimlar.get("belirgin_arka_plan", 0) / toplam
    if sessiz >= 0.70:           return "arka plan genel olarak temiz"
    if belirgin >= 0.35:         return "arka planda belirgin ek ses var"
    if hafif >= 0.40:            return "arka planda hafif seviye ek ses duyuluyor"
    return "arka plan büyük ölçüde kabul edilebilir"

# Ses seviyesi yorumu --> dağılımdan genel ses gücünü anlatır
def ses_seviyesi_yorumu(sayimlar: Dict[str, int]) -> str:
    toplam = sum(sayimlar.values())
    if toplam == 0:             return "ses seviyesi için veri yok"
    alcak = sayimlar.get("alçak", 0) / toplam
    orta = sayimlar.get("orta", 0) / toplam
    yuksek = sayimlar.get("yüksek", 0) / toplam
    if orta >= 0.55:            return "ses seviyesi çoğunlukla orta düzeyde"
    if yuksek >= 0.50:          return "ses seviyesi çoğunlukla yüksek"
    if alcak >= 0.50:           return "ses seviyesi çoğunlukla alçak"
    return "ses seviyesi bölüm içinde değişken dağılıyor"


# Konuşma enerjisi yorumu --> dağılımdan genel enerji seviyesini anlatır
def konusma_enerjisi_yorumu(sayimlar: Dict[str, int]) -> str:
    toplam = sum(sayimlar.values())
    if toplam == 0:                 return "konuşma enerjisi için veri yok"
    dusuk = sayimlar.get("düşük", 0) / toplam
    orta = sayimlar.get("orta", 0) / toplam
    yuksek = sayimlar.get("yüksek", 0) / toplam
    if orta >= 0.55:                return "genel konuşma enerjisi orta bölgede"
    if yuksek >= 0.45:              return "genel konuşma enerjisi yüksek tarafa yakın"
    if dusuk >= 0.45:               return "genel konuşma enerjisi düşük-sakin tarafa yakın"
    return "enerji düzeyi bölüm içinde değişken"


# Konuşma stili yorumu --> baskın etiketi kısa cümleye çevirir
def konusma_stili_yorumu(sayimlar: Dict[str, int]) -> str:
    toplam = sum(sayimlar.values())
    if toplam == 0:                 return "konuşma stili için veri yok"
    baskin = baskin_etiket(sayimlar)
    oran = sayimlar.get(baskin, 0) / toplam
    if oran < 0.40:                 return "tek bir baskın konuşma stili yok, yapı karışık"
    if baskin == "dengeli":         return "genel izlenim dengeli"
    if baskin == "canlı":           return "genel izlenim canlı"
    if baskin == "sakin":           return "genel izlenim sakin"
    return f"genel izlenim {baskin}"


# Özet hamı --> dakika ve genel özet için ortak veriyi çıkarır
def ozet_hami(saniye_df: pd.DataFrame) -> Dict[str, Any]:
    konusma_df = saniye_df[saniye_df["is_speech"]].copy()
    aktif_df = saniye_df[~saniye_df["is_silence"]].copy()

    return {
        "konusma_df": konusma_df,
        "aktif_df": aktif_df,
        "aktif_ses_orani": float(pd.to_numeric(saniye_df["active_ratio"], errors="coerce").fillna(0.0).mean()) if len(saniye_df) else 0.0,
        "konusma_orani": float(pd.to_numeric(saniye_df["speech_ratio"], errors="coerce").fillna(0.0).mean()) if len(saniye_df) else 0.0,
        "gercek_konusma_orani": float(pd.to_numeric(saniye_df["is_speech"], errors="coerce").fillna(0.0).mean()) if len(saniye_df) else 0.0,
        "tam_sessizlik_orani": float(pd.to_numeric(saniye_df["is_silence"], errors="coerce").fillna(0.0).mean()) if len(saniye_df) else 0.0,
        "konusma_disi_orani": float((saniye_df["segment_type"] == "konuşma_dışı").mean()) if len(saniye_df) else 0.0,
        "segment_turu_sayimlari": sayim_sozlugu(saniye_df["segment_type"]) if len(saniye_df) else {},
        "ses_sayimlari": sayim_sozlugu(konusma_df["ses_seviyesi"]) if len(konusma_df) else {},
        "arka_plan_sayimlari": sayim_sozlugu(aktif_df["arka_plan_durumu"]) if len(aktif_df) else {},
        "enerji_sayimlari": sayim_sozlugu(konusma_df["konusma_enerjisi"]) if len(konusma_df) else {},
        "stil_sayimlari": sayim_sozlugu(konusma_df["konusma_stili"]) if len(konusma_df) else {},
        "ort_rms_dbfs": guvenli_ortalama(konusma_df["rms_dbfs"]) if len(konusma_df) else float("nan"),
        "ort_f0": guvenli_ortalama(konusma_df["f0_mean"]) if len(konusma_df) else float("nan"),
        "ort_f0_std": guvenli_ortalama(konusma_df["f0_std"]) if len(konusma_df) else float("nan"),
        "ort_centroid": guvenli_ortalama(konusma_df["spectral_centroid"]) if len(konusma_df) else float("nan"),
    }


# Dakika özeti --> 1 dakikalık bloğu sayısal ve sözel olarak özetler
def dakika_ozeti(dakika_no: int, baslangic: int, bitis: int, dakika_df: pd.DataFrame) -> Dict[str, Any]:
    if len(dakika_df) == 0:
        return {
            "dakika_no": dakika_no,
            "zaman_araligi": f"{saniyeyi_yaz(baslangic)} - {saniyeyi_yaz(bitis)}",
            "ozet": "Bu dakika için analiz verisi bulunamadı."
        }

    o = ozet_hami(dakika_df)

    if o["gercek_konusma_orani"] < 0.15:
        ozet_cumlesi = (
            f"{dakika_no}. dakika: Bu bölümde gerçek konuşma az. "
            f"{aktif_ses_aciklamasi(o['aktif_ses_orani'])}. "
            f"{sessizlik_aciklamasi(o['tam_sessizlik_orani'])}."
        )
    else:
        ozet_cumlesi = (
            f"{dakika_no}. dakika: Konuşma-benzeri oran %{yuzde(o['konusma_orani'])}, "
            f"gerçek konuşma oranı %{yuzde(o['gercek_konusma_orani'])}. "
            f"{aktif_ses_aciklamasi(o['aktif_ses_orani'])}; "
            f"{sessizlik_aciklamasi(o['tam_sessizlik_orani'])}. "
            f"{ses_seviyesi_yorumu(o['ses_sayimlari'])}. "
            f"{arka_plan_aciklamasi(o['arka_plan_sayimlari'])}. "
            f"{ses_tonu_aciklamasi(o['ort_f0'])}; {ton_degisim_aciklamasi(o['ort_f0_std'])}. "
            f"{konusma_enerjisi_yorumu(o['enerji_sayimlari'])}. "
            f"{konusma_stili_yorumu(o['stil_sayimlari'])}."
        )

    return {
        "dakika_no": dakika_no,
        "zaman_araligi": f"{saniyeyi_yaz(baslangic)} - {saniyeyi_yaz(bitis)}",
        "aktif_ses_orani_yuzde": yuzde(o["aktif_ses_orani"]),
        "konusma_orani_yuzde": yuzde(o["konusma_orani"]),
        "gercek_konusma_orani_yuzde": yuzde(o["gercek_konusma_orani"]),
        "tam_sessizlik_orani_yuzde": yuzde(o["tam_sessizlik_orani"]),
        "konusma_disi_orani_yuzde": yuzde(o["konusma_disi_orani"]),
        "segment_turu_dagilimi": dagilim_yuzdesi(o["segment_turu_sayimlari"]),
        "ses_seviyesi_dagilimi": dagilim_yuzdesi(o["ses_sayimlari"]),
        "arka_plan_dagilimi": dagilim_yuzdesi(o["arka_plan_sayimlari"]),
        "konusma_enerjisi_dagilimi": dagilim_yuzdesi(o["enerji_sayimlari"]),
        "konusma_stili_dagilimi": dagilim_yuzdesi(o["stil_sayimlari"]),
        "ozet": ozet_cumlesi,
    }


# Genel özet --> tüm kayıt için tek paragraf ve ana dağılımları verir
def genel_ozet(saniye_df: pd.DataFrame, toplam_sure_sn: float) -> Dict[str, Any]:
    o = ozet_hami(saniye_df)

    kisa = (
        f"Kayıt süresi yaklaşık {toplam_sure_sn:.1f} saniye. "
        f"Konuşma-benzeri oran yaklaşık %{yuzde(o['konusma_orani'])}; "
        f"gerçek konuşma oranı yaklaşık %{yuzde(o['gercek_konusma_orani'])}. "
        f"Aktif ama konuşma dışı ses oranı yaklaşık %{yuzde(o['konusma_disi_orani'])}. "
        f"Aktif ses açısından {aktif_ses_aciklamasi(o['aktif_ses_orani'])}; "
        f"tam sessizlik açısından {sessizlik_aciklamasi(o['tam_sessizlik_orani'])}. "
        f"{ses_seviyesi_yorumu(o['ses_sayimlari'])}. "
        f"{arka_plan_aciklamasi(o['arka_plan_sayimlari'])}. "
        f"{ses_tonu_aciklamasi(o['ort_f0'])} ve {ton_degisim_aciklamasi(o['ort_f0_std'])}. "
        f"{konusma_enerjisi_yorumu(o['enerji_sayimlari'])}. "
        f"{konusma_stili_yorumu(o['stil_sayimlari'])}."
    )

    return {
        "toplam_sure_saniye": float(toplam_sure_sn),
        "toplam_sure_dakika": round(float(toplam_sure_sn) / 60.0, 2),
        "aktif_ses_orani_yuzde": yuzde(o["aktif_ses_orani"]),
        "konusma_orani_yuzde": yuzde(o["konusma_orani"]),
        "gercek_konusma_orani_yuzde": yuzde(o["gercek_konusma_orani"]),
        "tam_sessizlik_orani_yuzde": yuzde(o["tam_sessizlik_orani"]),
        "konusma_disi_orani_yuzde": yuzde(o["konusma_disi_orani"]),
        "segment_turu_dagilimi": dagilim_yuzdesi(o["segment_turu_sayimlari"]),
        "ses_seviyesi_dagilimi": dagilim_yuzdesi(o["ses_sayimlari"]),
        "arka_plan_dagilimi": dagilim_yuzdesi(o["arka_plan_sayimlari"]),
        "konusma_enerjisi_dagilimi": dagilim_yuzdesi(o["enerji_sayimlari"]),
        "konusma_stili_dagilimi": dagilim_yuzdesi(o["stil_sayimlari"]),
        "kisa_genel_ozet": kisa,
        "hesaplama_notu": "Konuşma oranı frame düzeyindeki konuşma-benzeri maskeden, gerçek konuşma oranı ise segment bazlı sıkı konuşma kararından gelir.",
        "uyari_notu": "Bu yorumlar psikolojik tanı değil, akustik özelliklerden türetilmiş yaklaşık teknik özetlerdir.",
    }


# Dakika özetleri --> kaydı 60 saniyelik bloklara ayırır
def dakika_ozetleri(saniye_df: pd.DataFrame, toplam_sure_sn: float) -> List[Dict[str, Any]]:
    toplam_dakika = int(math.ceil(toplam_sure_sn / 60.0))
    ozetler = []
    for i in range(toplam_dakika):
        bas = i * 60
        bit = min(int(math.ceil(toplam_sure_sn)), (i + 1) * 60)
        dakika_df = saniye_df[(saniye_df["start_sec"] >= bas) & (saniye_df["start_sec"] < bit)].copy()
        ozetler.append(dakika_ozeti(i + 1, bas, bit, dakika_df))
    return ozetler


# Pencere özeti --> konuşmama bloğu öncesi / sonrası pencere için kısa oran ve skor özeti çıkarır
def pencere_ozeti(saniye_df: pd.DataFrame, baslangic_sn: int, bitis_sn: int) -> Dict[str, Any]:
    pencere_df = saniye_df[(saniye_df["start_sec"] >= baslangic_sn) & (saniye_df["start_sec"] < bitis_sn)].copy()

    if len(pencere_df) == 0:
        return {
            "zaman_araligi": f"{saniyeyi_yaz(baslangic_sn)} - {saniyeyi_yaz(bitis_sn)}",            "aktif_ses_orani_yuzde": 0.0,
            "konusma_orani_yuzde": 0.0,               "gercek_konusma_orani_yuzde": 0.0,            "baskin_ses_seviyesi": "belirsiz",
            "baskin_ses_tonu": "belirsiz",            "baskin_arka_plan": "belirsiz",            "ozet": "Bu pencere için veri yok."
        }

    o = ozet_hami(pencere_df)
    aktif_df = pencere_df[~pencere_df["is_silence"]].copy()
    ses_sayimlari = sayim_sozlugu(aktif_df["ses_seviyesi"]) if len(aktif_df) else {}
    arka_plan_sayimlari = sayim_sozlugu(aktif_df["arka_plan_durumu"]) if len(aktif_df) else {}

    return {
        "zaman_araligi": f"{saniyeyi_yaz(baslangic_sn)} - {saniyeyi_yaz(bitis_sn)}",
        "aktif_ses_orani_yuzde": yuzde(o["aktif_ses_orani"]),
        "konusma_orani_yuzde": yuzde(o["konusma_orani"]),
        "gercek_konusma_orani_yuzde": yuzde(o["gercek_konusma_orani"]),
        "baskin_ses_seviyesi": baskin_etiket(ses_sayimlari),
        "baskin_ses_tonu": ses_tonu_etiketi(o["ort_f0"]),
        "baskin_arka_plan": baskin_etiket(arka_plan_sayimlari),
        "ozet": (
            f"{saniyeyi_yaz(baslangic_sn)} - {saniyeyi_yaz(bitis_sn)} arasında "
            f"aktif ses %{guvenli_yaz(yuzde(o['aktif_ses_orani']), 1)}, "
            f"konuşma-benzeri %{guvenli_yaz(yuzde(o['konusma_orani']), 1)}, "
            f"gerçek konuşma %{guvenli_yaz(yuzde(o['gercek_konusma_orani']), 1)}."
        )
    }


# Konuşmama blokları --> art arda gelen konuşmasız saniyeleri tek blok halinde toplar
def uzun_konusmama_bolgeleri(saniye_df: pd.DataFrame, min_sure_sn: int = 10) -> List[Tuple[int, int]]:
    if len(saniye_df) == 0:
        return []

    df = saniye_df.sort_values("start_sec").reset_index(drop=True)
    sec_list = df["start_sec"].astype(int).tolist()
    konusuyor_list = df["is_speech"].astype(bool).tolist()
    bolgeler = []
    bas = None

    for idx, (sec, konusuyor) in enumerate(zip(sec_list, konusuyor_list)):
        konusmuyor = not konusuyor
        if konusmuyor and bas is None:
            bas = sec

        kopuk = idx + 1 < len(sec_list) and sec_list[idx + 1] != sec + 1
        blok_sonu = idx == len(sec_list) - 1 or konusuyor_list[idx + 1] or kopuk

        if konusmuyor and blok_sonu:
            bit = int(math.ceil(float(df.iloc[idx]["end_sec"])))
            if bit - bas >= min_sure_sn:
                bolgeler.append((bas, bit))
            bas = None

    return bolgeler


# Konuşmama geçiş yorumu --> blok öncesi ve sonrası kısa fark cümlesi üretir
def konusmama_gecis_yorumu(once: Dict[str, Any], sonra: Dict[str, Any]) -> str:
    return (
        f"Öncesinde baskın ses seviyesi {once['baskin_ses_seviyesi']}, sonrasında {sonra['baskin_ses_seviyesi']}; "
        f"baskın ton {once['baskin_ses_tonu']} -> {sonra['baskin_ses_tonu']}. "
        f"Gerçek konuşma oranı {guvenli_yaz(sonra['gercek_konusma_orani_yuzde'] - once['gercek_konusma_orani_yuzde'], 1)} puan değişmiş."
    )


# Uzun konuşmama analizleri --> en az 10 saniye konuşulmayan yerlerin öncesi / sonrası özetini çıkarır
def uzun_konusmama_analizleri(saniye_df: pd.DataFrame, toplam_sure_sn: float, pencere_sn: int = 10, min_sure_sn: int = 10) -> List[Dict[str, Any]]:
    toplam_sure_int = int(math.ceil(toplam_sure_sn))
    analizler = []

    for blok_no, (bas, bit) in enumerate(uzun_konusmama_bolgeleri(saniye_df, min_sure_sn=min_sure_sn), start=1):
        once = pencere_ozeti(saniye_df, max(0, bas - pencere_sn), bas)
        blok = pencere_ozeti(saniye_df, bas, bit)
        sonra = pencere_ozeti(saniye_df, bit, min(toplam_sure_int, bit + pencere_sn))

        analizler.append({
            "blok_no": blok_no,
            "baslangic_sn": int(bas),
            "bitis_sn": int(bit),
            "sure_saniye": int(bit - bas),
            "oncesi_pencere": once,
            "konusmama_penceresi": blok,
            "sonrasi_pencere": sonra,
            "ozet": konusmama_gecis_yorumu(once, sonra),
        })

    return analizler


# Saniye detay metni --> her 1 saniyelik segmenti rapora satır satır yazar
def saniye_detay_metni(saniye_df: pd.DataFrame) -> str:
    satirlar = ["   SANİYE SANİYE DETAY", "-" * 80]

    for _, s in saniye_df.iterrows():
        bas = int(s["start_sec"])
        bit = int(math.ceil(s["end_sec"]))

        satirlar.append(
            f"{saniyeyi_yaz(bas)} - {saniyeyi_yaz(bit)} | "
            f"tur={s['segment_type']} | "
            f"aktif_oran=%{guvenli_yaz(float(s['active_ratio']) * 100.0, 1)} | "
            f"konusma_orani=%{guvenli_yaz(float(s['speech_ratio']) * 100.0, 1)} | "
            f"konusma_guveni=%{guvenli_yaz(float(s['konusma_guveni']) * 100.0, 1)} | "
            f"ses={s['ses_seviyesi']} | arka_plan={s['arka_plan_durumu']} | "
            f"enerji={s['konusma_enerjisi']} | stil={s['konusma_stili']} | "
            f"rms_dbfs={guvenli_yaz(s['rms_dbfs'], 2)} | zcr={guvenli_yaz(s['zcr'], 4)} | "
            f"f0={guvenli_yaz(s['f0_mean'], 2)} | f0_std={guvenli_yaz(s['f0_std'], 2)}"
        )

    satirlar.append("")
    return "\n".join(satirlar)


# Rapor metni --> txt raporun bütün gövdesini oluşturur
def rapor_metni(ozet: Dict[str, Any], saniye_df: pd.DataFrame) -> str:
    satirlar = [
        "           SES ANALİZ RAPORU",
        "=" * 80,
        "",
        "   GENEL BİLGİ", "-" * 80,
        f"Girdi dosyası: {ozet['girdi_dosyasi']}",
        f"Çıktı klasörü: {ozet['cikti_klasoru']}",
        f"Toplam süre              : {guvenli_yaz(ozet['genel_ozet']['toplam_sure_saniye'], 2)} sn (~ {guvenli_yaz(ozet['genel_ozet']['toplam_sure_dakika'], 2)} dk)",
        f"Aktif ses oranı          : %{guvenli_yaz(ozet['genel_ozet']['aktif_ses_orani_yuzde'], 2)}",
        f"Konuşma-benzeri oran     : %{guvenli_yaz(ozet['genel_ozet']['konusma_orani_yuzde'], 2)}",
        f"Gerçek konuşma oranı     : %{guvenli_yaz(ozet['genel_ozet']['gercek_konusma_orani_yuzde'], 2)}",
        f"Konuşma dışı ses         : %{guvenli_yaz(ozet['genel_ozet']['konusma_disi_orani_yuzde'], 2)}",
        f"Tam sessizlik oranı      : %{guvenli_yaz(ozet['genel_ozet']['tam_sessizlik_orani_yuzde'], 2)}",
        "",
        "   GENEL YORUM", "-" * 80,
        ozet["genel_ozet"]["kisa_genel_ozet"],
        ozet["genel_ozet"]["hesaplama_notu"],
        ozet["genel_ozet"]["uyari_notu"],
        "",
        "   DAĞILIMLAR", "-" * 80,
        "Segment türü dağılımı      : " + dagilim_metni(ozet["genel_ozet"]["segment_turu_dagilimi"], ["konuşma", "konuşma_dışı", "sessiz"]),
        "Ses seviyesi dağılımı      : " + dagilim_metni(ozet["genel_ozet"]["ses_seviyesi_dagilimi"], ["alçak", "orta", "yüksek"]),
        "Arka plan dağılımı         : " + dagilim_metni(ozet["genel_ozet"]["arka_plan_dagilimi"], ["sessiz_arka_plan", "hafif_arka_plan", "belirgin_arka_plan"]),
        "Konuşma enerjisi dağılımı  : " + dagilim_metni(ozet["genel_ozet"]["konusma_enerjisi_dagilimi"], ["düşük", "orta", "yüksek"]),
        "Konuşma stili dağılımı     : " + dagilim_metni(ozet["genel_ozet"]["konusma_stili_dagilimi"], ["sakin", "dengeli", "canlı", "karışık"]),
        "",
        "  DAKİKA DAKİKA ÖZET", "-" * 80,
    ]

    for dakika in ozet["dakika_ozetleri"]:
        satirlar += [
            f"{dakika['dakika_no']}. dakika ({dakika['zaman_araligi']}):",
            f"  {dakika['ozet']}",
            "",
        ]

    satirlar += ["  UZUN KONUŞMAMA ANALİZLERİ", "-" * 80]

    if not ozet.get("uzun_konusmama_analizleri"):
        satirlar += ["En az 10 saniyelik konuşmama bölgesi bulunamadı.", ""]
    else:
        for gecis in ozet["uzun_konusmama_analizleri"]:
            satirlar += [
                f"Blok {gecis['blok_no']} ({saniyeyi_yaz(gecis['baslangic_sn'])} - {saniyeyi_yaz(gecis['bitis_sn'])} | {gecis['sure_saniye']} sn):",
                f"  Öncesi   : {gecis['oncesi_pencere']['ozet']}",
                f"  Blok     : {gecis['konusmama_penceresi']['ozet']}",
                f"  Sonrası  : {gecis['sonrasi_pencere']['ozet']}",
                f"  Kısa yorum: {gecis['ozet']}",
                "",
            ]

    satirlar.append(saniye_detay_metni(saniye_df))
    return "\n".join(satirlar)

# Ana akış --> seç, çıkar, analiz et, json ve txt olarak kaydet
def ana_akis():
    medya_yolu = girdi_dosyasi_sec()
    cikti_klasoru = sonraki_video_klasoru(kayit_klasoru)
    ozet_json_yolu = cikti_klasoru / "ozet.json"
    ozet_txt_yolu = cikti_klasoru / "ozet.txt"
    gecici_wav = None

    try:
        print("1/8 --> Ses akışları inceleniyor...")
        ses_akislari = ffprobe_ses_akislari(medya_yolu)
        if not ses_akislari:
            raise RuntimeError("Seçilen dosyada hiç audio stream bulunamadı.")

        print("2/8 --> Ses çıkarılıyor / dönüştürülüyor...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            gecici_wav = Path(tmp.name)
        sesi_cikar(medya_yolu, gecici_wav, hedef_ornekleme_hizi)

        print("3/8 --> Geçici wav yükleniyor...")
        dalga, sr = torchaudio.load(str(gecici_wav))
        if sr != hedef_ornekleme_hizi:
            dalga = torchaudio.functional.resample(dalga, sr, hedef_ornekleme_hizi)
            sr = hedef_ornekleme_hizi

        if float(dalga.abs().max().item()) < 1e-7:
            raise RuntimeError("Çıkarılan ses neredeyse tamamen sessiz.")

        print("4/8 --> En iyi kanal seçiliyor...")
        secilen_kanal, kanal_bilgi = en_iyi_kanal(dalga)
        mono = dalga[secilen_kanal:secilen_kanal + 1].contiguous()

        toplam_sure_sn = mono.shape[1] / sr
        parca_sayisi = int(math.ceil(toplam_sure_sn / parca_suresi_sn))

        print("5/8 --> Frame özellikleri hazırlanıyor...")
        frame_len = int(sr * cerceve_ms / 1000)
        hop_len = int(sr * cerceve_kayma_ms / 1000)

        tum_rms = cerceve_rms(mono, frame_len, hop_len).numpy()
        tum_zcr = cerceve_zcr(mono, frame_len, hop_len).numpy()

        sifir_olmayan_rms = tum_rms[tum_rms > 1e-8]
        if sifir_olmayan_rms.size == 0:
            raise RuntimeError("Tüm frame RMS değerleri sıfır görünüyor. Ses problemli.")

        # Sessizlik eşiği:
        # silence_rms_threshold = max(mutlak_sessizlik_rms, 0.5 * percentile_15(RMS))
        sessizlik_rms_esigi = max(mutlak_sessizlik_rms, float(np.percentile(sifir_olmayan_rms, 15) * 0.5))

        # Konuşma eşiği:
        # speech_rms_threshold = max(2.2 * silence_rms_threshold, percentile_40(RMS))
        konusma_rms_esigi = max(sessizlik_rms_esigi * 2.2, float(np.percentile(sifir_olmayan_rms, 40)))

        aktif_maske = tum_rms >= sessizlik_rms_esigi

        print("6/8 --> Dönüşümler ve pitch hazırlanıyor...")
        mel_donusumu = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=fft_boyutu, hop_length=kayma_boyutu, n_mels=mel_bant_sayisi, power=2.0)
        spektrogram_donusumu = torchaudio.transforms.Spectrogram(n_fft=fft_boyutu, hop_length=kayma_boyutu, power=2.0)
        pitch_izi = tum_pitch_izi(mono.cpu(), sr)

        pitch_np = pitch_izi.numpy() if pitch_izi.numel() else np.full(len(tum_rms), np.nan, dtype=np.float32)
        ortak_frame_sayisi = min(len(tum_rms), len(tum_zcr), len(pitch_np))
        tum_rms, tum_zcr, aktif_maske, pitch_np = tum_rms[:ortak_frame_sayisi], tum_zcr[:ortak_frame_sayisi], aktif_maske[:ortak_frame_sayisi], pitch_np[:ortak_frame_sayisi]
        ortak_pitch_izi = torch.from_numpy(pitch_np.copy()) if ortak_frame_sayisi > 0 else torch.empty(0, dtype=torch.float32)

        # Güvenilir sesli frame maskesi:
        # voiced_frame =
        #   valid_f0 AND RMS >= 1.4 * silence_threshold AND ZCR <= 0.18
        ham_sesli_maske = (
            np.isfinite(pitch_np) &  (pitch_np >= f0_min) &
            (pitch_np <= f0_max)  &  (tum_rms >= max(sessizlik_rms_esigi * 1.4, mutlak_sessizlik_rms)) &
            (tum_zcr <= 0.18)
        )

        # Konuşma-benzeri frame maskesi:
        # speech_like_frame =
        #   RMS >= speech_rms_threshold AND
        #   0.01 <= ZCR <= 0.16 AND
        #   (voiced_frame OR (RMS çok yüksek ve ZCR kontrollü))
        ham_konusma_maske = (
            (tum_rms >= konusma_rms_esigi) &
            (tum_zcr >= 0.01) &
            (tum_zcr <= 0.16) &
            (ham_sesli_maske | ((tum_rms >= konusma_rms_esigi * 1.25) & (tum_zcr <= 0.10)))
        )

        sesli_maske = maske_duzgunlestir(ham_sesli_maske, 3) & aktif_maske
        konusma_maske = maske_duzgunlestir(ham_konusma_maske, 3) & aktif_maske

        print("7/8 --> Saniye bazlı analiz yapılıyor...")
        satirlar = []
        frame_hop_sn = hop_len / sr

        for i in range(parca_sayisi):
            bas_ornek = int(i * parca_suresi_sn * sr)
            bit_ornek = min(int((i + 1) * parca_suresi_sn * sr), mono.shape[1])
            ses_parcasi = mono[:, bas_ornek:bit_ornek]
            if ses_parcasi.shape[1] < sr // 4:
                continue

            bas_sn = bas_ornek / sr
            bit_sn = bit_ornek / sr
            parca_rms = float(torch.sqrt(torch.mean(ses_parcasi ** 2) + eps).item())
            parca_rms_dbfs = float(rmsi_dbfs(parca_rms))
            parca_peak = float(torch.max(torch.abs(ses_parcasi)).item())
            parca_zcr = zcr_orani(ses_parcasi)

            frame_bas = max(0, min(int(round(bas_sn / frame_hop_sn)), ortak_frame_sayisi))
            frame_bit = max(frame_bas, min(int(math.ceil(bit_sn / frame_hop_sn)), ortak_frame_sayisi))

            aktif_oran = float(np.mean(aktif_maske[frame_bas:frame_bit])) if frame_bit > frame_bas else 0.0
            konusma_orani = float(np.mean(konusma_maske[frame_bas:frame_bit])) if frame_bit > frame_bas else 0.0
            sesli_frame_orani = float(np.mean(sesli_maske[frame_bas:frame_bit])) if frame_bit > frame_bas else 0.0

            # Tam sessizlik kararı:
            # segment is silence if
            #   active_ratio < 0.06 AND RMS çok düşük AND peak çok düşük
            sessiz_mi = bool(
                aktif_oran < min_tam_sessizlik_aktif_oran and
                parca_rms < max(sessizlik_rms_esigi * 1.5, mutlak_sessizlik_rms * 2.0) and
                parca_peak < max(mutlak_sessizlik_peak * 2.0, sessizlik_rms_esigi * 6.0)
            )

            if sessiz_mi:
                segment_turu = "sessiz"
                spektral_merkez, spektral_duzluk, mel_enerjisi = np.nan, np.nan, 0.0
                f0_ort, f0_std, sesli_oran = np.nan, np.nan, 0.0
                arka_plan_skoru = np.nan
                konusma_guveni = 0.0
                enerji_skoru = np.nan
                ses_etiketi = "sessiz"
                arka_plan_durumu = "sessiz"
                konusma_enerjisi = "belirsiz"
                konusma_stili = "sessiz"
            else:
                spektral_merkez, spektral_duzluk, mel_enerjisi = spektral_ozellikler(
                    ses_parcasi=ses_parcasi.cpu(),
                    ornekleme_hizi=sr,
                    spektrogram_donusumu=spektrogram_donusumu,
                    mel_donusumu=mel_donusumu,
                )
                f0_ort, f0_std, sesli_oran = parca_pitch_istatistikleri(
                    pitch_izi=ortak_pitch_izi,
                    cerceve_rms_izi=tum_rms,
                    cerceve_zcr_izi=tum_zcr,
                    baslangic_sn=bas_sn,
                    bitis_sn=bit_sn,
                    sessizlik_rms_esigi=sessizlik_rms_esigi,
                )

                ses_etiketi = ses_seviyesi_etiketi(parca_rms_dbfs)
                arka_plan_skoru = arka_plan_skoru_hesapla(spektral_duzluk, parca_zcr, spektral_merkez, parca_rms_dbfs)
                arka_plan_durumu = arka_plan_etiketi(arka_plan_skoru)

                konusma_guveni = konusma_guven_skoru_hesapla(
                    konusma_orani=konusma_orani,
                    sesli_oran=sesli_frame_orani,
                    spektral_duzluk=spektral_duzluk,
                    spektral_merkez=spektral_merkez,
                    rms_dbfs_degeri=parca_rms_dbfs,
                    arka_plan_skoru=arka_plan_skoru,
                )

                # Gerçek konuşma kararı:
                # segment is speech if
                #   (speech_ratio >= 0.28 AND voiced_ratio >= 0.12 AND speech_conf >= 0.58)
                #   OR
                #   (speech_ratio >= 0.40 AND speech_conf >= 0.52)
                konusma_var_mi = bool(
                    (konusma_orani >= min_gercek_konusma_orani and sesli_frame_orani >= min_sesli_oran and konusma_guveni >= konusma_guven_esigi) or
                    (konusma_orani >= 0.40 and konusma_guveni >= 0.52)
                )

                segment_turu = "konuşma" if konusma_var_mi else "konuşma_dışı"

                if konusma_var_mi:
                    enerji_skoru = konusma_enerjisi_skoru(parca_rms_dbfs, spektral_merkez, f0_std)
                    konusma_enerjisi = konusma_enerjisi_etiketi(enerji_skoru)
                    konusma_stili = konusma_stili_etiketi(enerji_skoru, arka_plan_skoru, f0_std if np.isfinite(f0_std) else 0.0, sesli_oran)
                else:
                    enerji_skoru = np.nan
                    konusma_enerjisi = "belirsiz"
                    konusma_stili = "konuşma_dışı"

            satirlar.append({
                "start_sec": int(i),
                "end_sec": float(bit_sn),
                "segment_type": segment_turu,
                "is_silence": segment_turu == "sessiz",
                "is_speech": segment_turu == "konuşma",
                "active_ratio": float(aktif_oran),
                "speech_ratio": float(konusma_orani),
                "rms_dbfs": float(parca_rms_dbfs),
                "zcr": float(parca_zcr),
                "spectral_centroid": float(spektral_merkez) if np.isfinite(spektral_merkez) else np.nan,
                "spectral_flatness": float(spektral_duzluk) if np.isfinite(spektral_duzluk) else np.nan,
                "mel_energy": float(mel_enerjisi),
                "f0_mean": float(f0_ort) if np.isfinite(f0_ort) else np.nan,
                "f0_std": float(f0_std) if np.isfinite(f0_std) else np.nan,
                "voiced_ratio": float(sesli_oran),
                "arka_plan_skoru": float(arka_plan_skoru) if np.isfinite(arka_plan_skoru) else np.nan,
                "konusma_guveni": float(konusma_guveni),
                "enerji_skoru": float(enerji_skoru) if np.isfinite(enerji_skoru) else np.nan,
                "ses_seviyesi": ses_etiketi,
                "arka_plan_durumu": arka_plan_durumu,
                "konusma_enerjisi": konusma_enerjisi,
                "konusma_stili": konusma_stili,
            })

        saniye_df = pd.DataFrame(satirlar)
        if len(saniye_df) == 0:
            raise RuntimeError("Segment analizi boş döndü..")

        print("8/8 --> Rapor hazırlanıyor...")
        uzun_bolgeler = uzun_konusmama_analizleri(
            saniye_df=saniye_df,
            toplam_sure_sn=toplam_sure_sn,
            pencere_sn=gecis_penceresi_sn,
            min_sure_sn=uzun_konusmama_esigi_sn,
        )

        ozet = {
            "girdi_dosyasi": medya_yolu,
            "cikti_klasoru": cikti_klasoru,
            "analiz_bilgisi": {
                "ornekleme_hizi_hz": int(sr),
                "toplam_sure_saniye": float(toplam_sure_sn),
                "kullanilan_kanal": int(kanal_bilgi["secilen_kanal"]),
                "ses_akislari": ses_akislari,
                "sessizlik_esigi_rms": float(sessizlik_rms_esigi),
                "konusma_esigi_rms": float(konusma_rms_esigi),
            },
            "genel_ozet": genel_ozet(saniye_df, toplam_sure_sn),
            "dakika_ozetleri": dakika_ozetleri(saniye_df, toplam_sure_sn),
            "uzun_konusmama_var_mi": len(uzun_bolgeler) > 0,
            "uzun_konusmama_analizleri": uzun_bolgeler,
        }

        with open(ozet_json_yolu, "w", encoding="utf-8") as dosya:
            json.dump(json_guvenli(ozet), dosya, ensure_ascii=False, indent=2)

        metin_kaydet(ozet_txt_yolu, rapor_metni(ozet, saniye_df))

        print("")
        print("Bitti.")
        print("Çıktı klasörü   -->", cikti_klasoru)
        print("Oluşan dosyalar -->")
        print(" - ozet.json")
        print(" - ozet.txt")

    finally:
        if gecici_wav is not None and gecici_wav.exists():
            try:
                gecici_wav.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    ana_akis()