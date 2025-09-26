# 🍇 Üzüm Hastalık Tespiti - CNN Transfer Learning

**VGG16 Transfer Learning ile %93+ doğrulukla üzüm yaprak hastalıklarını tespit eden yapay zeka projesi**

## 🎯 Proje Hakkında

Bu proje, tarım sektörünün en büyük sorunlarından biri olan üzüm hastalıklarının erken ve doğru tespiti için geliştirilmiş bir yapay zeka çözümüdür. Convolutional Neural Networks (CNN) ve Transfer Learning teknikleri kullanılarak, üzüm yapraklarının fotoğraflarından hastalık türünü %93+ doğrulukla tespit edebilir.

### 🦠 Tespit Edilen Hastalıklar

- **🔴 Black Rot** - Siyah leke hastalığı (fungal enfeksiyon)
- **🟡 ESCA** - Sarılaşma ve nekroz oluşturan hastalık
- **🟢 Healthy** - Sağlıklı yapraklar  
- **🟤 Leaf Blight** - Yaprak yanıklığı ve hasar belirtileri

## 🏆 Performans Sonuçları

- **Validation Accuracy**: %93.2
- **Training Accuracy**: %95.1
- **F1-Score**: 0.923
- **Precision**: 0.931
- **Recall**: 0.927
- **Model Size**: 57.8 MB (production-ready)

## 🚀 Kullanılan Teknolojiler

**Deep Learning Stack:**
- **TensorFlow 2.x** - Ana framework
- **Keras** - High-level API
- **VGG16** - Pre-trained model (ImageNet)

**Data Processing:**
- **OpenCV** - Görüntü işleme
- **NumPy** - Sayısal hesaplamalar
- **Pandas** - Veri analizi

**Visualization & Analysis:**
- **Matplotlib** - Grafik oluşturma
- **Seaborn** - İstatistiksel görselleştirme
- **Grad-CAM** - Model interpretability

## ⚡ Kurulum ve Kullanım

### Sistem Gereksinimleri    
- Python 3.8+
- 4GB+ RAM
- GPU (opsiyonel, ama önerilen)

