## test_videolar (lokal test için)

Bu klasör, geliştirirken hızlı denemeler için video koyabileceğin yerdir.

Önemli notlar:
- Bu repo’da `.gitignore` gereği `*.mp4`, `*.wav` vb. dosyalar **git’e eklenmez**.
- Bunun sebebi: dosya boyutu, gizlilik (KVKK/PII), ve repo şişmesini önlemek.

### Nereden ekleyeceğim?
- Videolarını bu klasöre kopyala: `test_videolar/<dosya>.mp4`

### Çıktıları (reports) nerede göreceğim?
- Analiz çalışınca raporlar otomatik `reports/` altına yazılır:
  - `reports/report_<interview_id>.json`
  - `reports/report_<interview_id>.html`
  - grafikler: `reports/charts/*.png`

### GitHub’a ilk push için öneri
- Kaynak kod + doküman + küçük text dosyaları push’la.
- Büyük video/rapor çıktıları için:
  - Lokal bırak (önerilen)
  - veya Git LFS / ayrı private storage / GitHub Release kullan

