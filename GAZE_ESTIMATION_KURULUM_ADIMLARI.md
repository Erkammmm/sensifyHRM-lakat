# Gaze Estimation Entegrasyon - Adım Adım Kurulum

## 📋 Manuel Adımlar (Sen Yapacaksın)

### ✅ Adım 1: Repo'yu Clone Et veya ZIP İndir

**Seçenek A - Git ile (Anaconda Prompt'ta)**:
```bash
cd C:\Users\cetki\Desktop
git clone https://github.com/yakhyo/gaze-estimation.git
```

**Seçenek B - ZIP İndir (Git yoksa)**:
1. Tarayıcıda aç: https://github.com/yakhyo/gaze-estimation/archive/refs/heads/main.zip
2. ZIP'i indir
3. `C:\Users\cetki\Desktop\` klasörüne çıkart
4. Klasör adı: `gaze-estimation-main` olacak (veya `gaze-estimation`)

**Sonuç**: `C:\Users\cetki\Desktop\gaze-estimation\` klasörü olmalı

---

### ✅ Adım 2: Model Kodlarını Kopyala

**Kaynak Klasör**: `C:\Users\cetki\Desktop\gaze-estimation\models\`  
**Hedef Klasör**: `C:\Users\cetki\Desktop\sensifyHRMülakay\src\models\`

**Windows Explorer'da Yapılacaklar**:

1. **Hedef klasörü oluştur** (eğer yoksa):
   - `C:\Users\cetki\Desktop\sensifyHRMülakay\src\` klasörüne git
   - Sağ tık → Yeni → Klasör → İsim: `models`

2. **Kaynak klasörü aç**:
   - `C:\Users\cetki\Desktop\gaze-estimation\models\` klasörünü aç
   - İçindeki **tüm `.py` dosyalarını** gör (örnek: `l2cs.py`, `mobilenet.py`, `resnet.py`, vb.)

3. **Dosyaları kopyala**:
   - Tüm `.py` dosyalarını seç (Ctrl+A veya tek tek seç)
   - Kopyala (Ctrl+C)
   - `C:\Users\cetki\Desktop\sensifyHRMülakay\src\models\` klasörüne git
   - Yapıştır (Ctrl+V)

**Kontrol**: `src\models\` klasöründe `.py` dosyaları olmalı

---

### ✅ Adım 3: Utils Klasörünü Kopyala (Gerekirse)

**Kaynak**: `C:\Users\cetki\Desktop\gaze-estimation\utils\`  
**Hedef**: `C:\Users\cetki\Desktop\sensifyHRMülakay\src\utils\`

**Eğer `models\` klasöründeki dosyalar `utils` import ediyorsa**:
1. `src\utils\` klasörü oluştur (yoksa)
2. `gaze-estimation\utils\` içindeki `.py` dosyalarını kopyala

**Not**: Önce model dosyalarını kopyala, sonra hata alırsan utils'i de kopyala.

---

### ✅ Adım 4: Config Dosyasını Kontrol Et (Gerekirse)

**Kaynak**: `C:\Users\cetki\Desktop\gaze-estimation\config.py`  
**Hedef**: `C:\Users\cetki\Desktop\sensifyHRMülakay\src\config.py` (veya `src\models\config.py`)

Eğer model dosyaları `config` import ediyorsa, bu dosyayı da kopyala.

---

## ✅ Benim Yapacağım (Kod Entegrasyonu)

Sen manuel adımları tamamladıktan sonra ben şunları yapacağım:

1. ✅ `src/gaze_estimation_model.py` - Model wrapper'ı tamamlayacağım (model architecture'ı import edeceğim)
2. ✅ `src/eye_contact_analyzer.py` - MediaPipe'ı kaldırıp gaze-estimation ekleyeceğim
3. ✅ Face detection ekleyeceğim (MediaPipe Face Detection - yüz crop için)
4. ✅ Pipeline'ı güncelleyeceğim (gerekirse)

---

## 🔍 Kontrol Listesi

Manuel adımları yaptıktan sonra kontrol et:

- [ ] `C:\Users\cetki\Desktop\gaze-estimation\` klasörü var mı?
- [ ] `C:\Users\cetki\Desktop\sensifyHRMülakay\src\models\` klasörü var mı?
- [ ] `src\models\` içinde `.py` dosyaları var mı? (en az 1-2 dosya)
- [ ] `src\models\__init__.py` dosyası var mı? (ben oluşturdum, kontrol et)

---

## 📝 Sonraki Adım

**Manuel adımları tamamladıktan sonra bana şunu yaz**:
- "hazır" veya "tamamladım" veya "kopyaladım"

Ben de kod entegrasyonunu tamamlayacağım ve test edeceğiz.

---

## ⚠️ Notlar

- Model weights **otomatik indirilecek** (kod hazır, sen bir şey yapmana gerek yok)
- İlk çalıştırmada `weights\gaze_estimation\` klasörü oluşturulacak ve model indirilecek
- Eğer model dosyalarında import hatası alırsan, eksik dosyaları da kopyala (utils, config, vb.)
