# Gaze Estimation Entegrasyon Rehberi

## 📋 Özet

MediaPipe FaceMesh tabanlı göz teması analizini, [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation) repo'sundaki pretrained CNN modeli ile değiştiriyoruz.

## 🎯 Neden Bu Değişiklik?

1. **Öğrenilmiş Model**: Rule-based geometrik hesaplamalar yerine CNN tabanlı, daha doğru
2. **Mobil Uyumlu**: MobileNet v2, MobileOne gibi hafif modeller
3. **Pretrained**: Gaze360/MPIIGaze ile eğitilmiş, ek eğitim gerekmez
4. **Real-time**: ONNX Runtime ile hızlı inference

## ✅ Otomatik İndirme

**Model weights otomatik olarak indirilir!** İlk kullanımda:
- Model weights GitHub'dan otomatik indirilir
- `weights/gaze_estimation/` klasörüne kaydedilir
- Sonraki kullanımlarda cache'den yüklenir

**Manuel işlem gerekmez!** Sadece `python test_example.py` çalıştır, model otomatik indirilir.

## 📦 Gereksinimler

### Bağımlılıkları Yükle

```bash
pip install torch torchvision
# veya ONNX Runtime için:
pip install onnxruntime
```

**Not**: Repo'yu clone etmeye gerek yok! Model weights otomatik indirilir.

## 🔧 Entegrasyon Adımları

### ⚠️ Önemli: Model Architecture Kodu

**Model architecture kodunu repo'dan kopyalamak gerekir:**
1. Repo'yu geçici olarak clone et: `git clone https://github.com/yakhyo/gaze-estimation.git temp_gaze`
2. `temp_gaze/models/` klasöründen model kodunu `src/models/` klasörüne kopyala
3. `temp_gaze` klasörünü sil (artık gerekmez, weights otomatik indirildi)

### Adım 1: Face Detection Ekle

Gaze estimation için yüz crop gerekir. Seçenekler:
- **MediaPipe Face Detection** (zaten var, hafif) ✅ Önerilen
- **uniface** (repo'da kullanılan, daha hızlı)

### Adım 2: Eye Contact Analyzer'ı Güncelle

`src/eye_contact_analyzer.py` dosyasını yeniden yaz:
- MediaPipe FaceMesh → Gaze Estimation Model
- Rule-based hesaplama → Model inference
- Yaw-pitch çıktısını eye contact score'a çevir

## 📊 Model Seçimi

| Model | MAE (derece) | Boyut | Hız | Önerilen |
|-------|--------------|-------|-----|----------|
| ResNet18 | ~4.5 | ~45MB | Orta | ✅ Genel kullanım |
| ResNet34 | ~4.2 | ~85MB | Yavaş | ❌ |
| ResNet50 | ~4.0 | ~100MB | Çok yavaş | ❌ |
| MobileNet v2 | ~5.0 | ~15MB | Hızlı | ✅ Mobil/Edge |
| MobileOne s0 | ~5.5 | ~5MB | Çok hızlı | ✅ En hafif |

**Öneri**: İlk faz için **MobileNet v2** (dengeli), production için **MobileOne s0** (en hızlı).

## 🔄 Çıktı Formatı

Model yaw-pitch çıktısı verir (radyan veya derece):
- **Yaw**: Yatay açı (-90° → +90°)
- **Pitch**: Dikey açı (-90° → +90°)

Eye contact score hesaplama:
```python
gaze_angle = sqrt(yaw² + pitch²)
eye_contact_score = max(0, 1.0 - (gaze_angle / 25.0))  # 0-25° arası iyi
```

## ⚠️ Notlar

1. **İlk çalıştırmada**: Model weights indirilecek (~15-45MB)
2. **Face detection**: Yüz crop gerekir (224x224 veya 448x448)
3. **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Inference**: CPU'da çalışır, GPU varsa daha hızlı

## 🚀 Hızlı Başlangıç

```python
from src.gaze_estimation_model import GazeEstimationModel
import cv2

# Model'i yükle
model = GazeEstimationModel(model_name="mobilenetv2")

# Frame'den yüz crop et (MediaPipe veya uniface ile)
face_crop = detect_and_crop_face(frame)

# Gaze estimation
gaze_result = model.estimate_gaze(face_crop)
yaw = gaze_result['yaw']
pitch = gaze_result['pitch']
```

## 📝 Sonraki Adımlar

1. ✅ Repo'yu clone et
2. ✅ Model weights'i indir
3. ✅ Model architecture kodunu kopyala
4. ✅ `eye_contact_analyzer.py`'yi güncelle
5. ✅ Test et
