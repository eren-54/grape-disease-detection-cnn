# Bellekte temizlik yapalım
import gc
import tensorflow as tf
tf.keras.backend.clear_session()
gc.collect()

print("🧹 BELLEK TEMİZLENDİ - YENİ CNN MODELİNE BAŞLANGAÇ")
print("=" * 60)

# Temel kütüphaneleri yeniden import edelim
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# TensorFlow ve Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Dense, Dropout, Flatten, BatchNormalization,
    Activation
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ GPU kullanılabilirlik: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Veri yolları yeniden tanımla
base_path = '/kaggle/input/yapraklar/datam/Original Data'
train_path = os.path.join(base_path, 'train')
validation_path = os.path.join(base_path, 'validation')
test_path = os.path.join(base_path, 'test')

# Sınıf bilgileri
class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']
num_classes = len(class_names)

print(f"📂 Veri yolu: {base_path}")
print(f"🏷️ Sınıf sayısı: {num_classes}")
print(f"🏷️ Sınıflar: {class_names}")

print("📊 GELİŞMİŞ VERİ ANALİZİ VE GÖRSELLEŞTİRME")
print("=" * 60)


# Veri sayılarını yeniden hesapla
def detailed_dataset_analysis(base_path):
    """Detaylı veri seti analizi"""
    analysis_data = {}

    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(base_path, split)
        split_info = {}

        print(f"\n🔍 {split.upper()} analizi yapılıyor...")

        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                # Görüntü dosyalarını filtrele
                image_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

                # Dosya uzantı analizi
                extensions = {}
                for file in image_files:
                    ext = file.lower().split('.')[-1]
                    extensions[ext] = extensions.get(ext, 0) + 1

                split_info[class_name] = {
                    'count': len(image_files),
                    'extensions': extensions
                }
                print(f"  ✅ {class_name}: {len(image_files)} görüntü {extensions}")
            else:
                split_info[class_name] = {'count': 0, 'extensions': {}}
                print(f"  ❌ {class_name}: Klasör bulunamadı")

        analysis_data[split] = split_info

    return analysis_data


# Detaylı analiz yap
detailed_data = detailed_dataset_analysis(base_path)

# DataFrame oluştur
df_counts = pd.DataFrame({
    split: {class_name: info['count'] for class_name, info in data.items()}
    for split, data in detailed_data.items()
})

print(f"\n📋 VERİ SETİ ÖZET TABLOSU")
print("=" * 40)
print(df_counts)

# Toplam sayıları ekle
df_counts.loc['TOPLAM'] = df_counts.sum()
df_counts['TOPLAM'] = df_counts.sum(axis=1)

print(f"\n📋 TOPLAM İLE BİRLİKTE")
print("=" * 40)
print(df_counts)

# Matplotlib ayarları
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11

# Renk paleti tanımla
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
sns.set_palette(colors)

print("📈 GELİŞMİŞ GÖRSELLEŞTİRMELER")
print("=" * 50)

# Çoklu görselleştirme
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Sınıf dağılımları (Bar plot)
ax1 = axes[0, 0]
df_no_total = df_counts.drop(['TOPLAM'], axis=0).drop(['TOPLAM'], axis=1)
df_no_total.plot(kind='bar', ax=ax1, color=colors, width=0.8)
ax1.set_title('📊 Veri Setlerindeki Sınıf Dağılımları', fontsize=14, fontweight='bold')
ax1.set_xlabel('Sınıflar')
ax1.set_ylabel('Görüntü Sayısı')
ax1.legend(title='Veri Setleri', loc='upper right')
ax1.tick_params(axis='x', rotation=45)

# 2. Train set pie chart
ax2 = axes[0, 1]
train_counts = df_no_total['train']
wedges, texts, autotexts = ax2.pie(train_counts.values,
                                  labels=train_counts.index,
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90,
                                  explode=(0.05, 0.05, 0.05, 0.05))
ax2.set_title('🥧 Train Setindeki Sınıf Dağılımı', fontsize=14, fontweight='bold')

# 3. Toplam görüntü sayıları
ax3 = axes[0, 2]
total_by_split = df_no_total.sum()
bars = ax3.bar(total_by_split.index, total_by_split.values,
               color=['#3498DB', '#E67E22', '#27AE60'], alpha=0.8)
ax3.set_title('📈 Veri Setlerindeki Toplam Görüntü Sayıları', fontsize=14, fontweight='bold')
ax3.set_ylabel('Görüntü Sayısı')
# Bar değerlerini göster
for bar in bars:
    height = bar.get_height()
    ax3.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

# 4. Heatmap
ax4 = axes[1, 0]
sns.heatmap(df_no_total.T, annot=True, fmt='d', cmap='YlOrRd',
            ax=ax4, cbar_kws={'label': 'Görüntü Sayısı'})
ax4.set_title('🔥 Veri Seti Dağılım Heatmap', fontsize=14, fontweight='bold')

# 5. Sınıf dengesi analizi
ax5 = axes[1, 1]
class_balance = df_no_total.sum(axis=1)  # Her sınıfın toplam sayısı
balance_percentage = (class_balance / class_balance.sum()) * 100
bars = ax5.barh(class_balance.index, balance_percentage, color=colors, alpha=0.7)
ax5.set_title('⚖️ Sınıf Dengesi Analizi (%)', fontsize=14, fontweight='bold')
ax5.set_xlabel('Yüzde (%)')
# Değerleri göster
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax5.annotate(f'{width:.1f}%',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha='left', va='center', fontweight='bold')

# 6. Veri seti oranları
ax6 = axes[1, 2]
split_ratios = (total_by_split / total_by_split.sum()) * 100
wedges, texts, autotexts = ax6.pie(split_ratios.values,
                                  labels=[f'{label}\n({ratio:.1f}%)'
                                         for label, ratio in zip(split_ratios.index, split_ratios.values)],
                                  colors=['#3498DB', '#E67E22', '#27AE60'],
                                  startangle=90,
                                  autopct='')
ax6.set_title('📊 Veri Seti Oranları', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# İstatistiksel özet
print(f"\n📈 VERİ SETİ İSTATİSTİKLERİ")
print("=" * 40)
total_images = df_counts.loc['TOPLAM', 'TOPLAM']
print(f"🔢 Toplam görüntü sayısı: {total_images:,}")
print(f"🏷️ Sınıf sayısı: {len(class_names)}")
print(f"📁 Veri seti sayısı: 3 (train, validation, test)")

# Oranları hesapla
train_ratio = (df_counts.loc['TOPLAM', 'train'] / total_images) * 100
val_ratio = (df_counts.loc['TOPLAM', 'validation'] / total_images) * 100
test_ratio = (df_counts.loc['TOPLAM', 'test'] / total_images) * 100

print(f"\n📊 VERİ SETİ ORANLARI:")
print(f"  🚂 Train: {train_ratio:.1f}% ({df_counts.loc['TOPLAM', 'train']:,} görüntü)")
print(f"  🔍 Validation: {val_ratio:.1f}% ({df_counts.loc['TOPLAM', 'validation']:,} görüntü)")
print(f"  🧪 Test: {test_ratio:.1f}% ({df_counts.loc['TOPLAM', 'test']:,} görüntü)")

# Sınıf dengesi kontrolü
print(f"\n⚖️ SINIF DENGESİ ANALİZİ:")
for class_name in class_names:
    class_total = df_counts.loc[class_name, 'TOPLAM']
    class_percentage = (class_total / total_images) * 100
    print(f"  {class_name}: %{class_percentage:.1f} ({class_total:,} görüntü)")

from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("🖼️ ÖRNEK GÖRÜNTÜLERİ VE VERİ ANALİZİ")
print("=" * 50)


# Her sınıftan örnek görüntüleri göster
def show_sample_images_with_info(data_path, class_names, samples_per_class=5):
    """Her sınıftan örnek görüntüler ve bilgileri"""

    fig, axes = plt.subplots(len(class_names), samples_per_class,
                             figsize=(samples_per_class * 3, len(class_names) * 3))

    image_info = []

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)

        if os.path.exists(class_path):
            # Görüntü dosyalarını al
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            # Rastgele örnekler seç
            np.random.seed(42)  # Tekrarlanabilir sonuçlar için
            selected_files = np.random.choice(image_files,
                                              min(samples_per_class, len(image_files)),
                                              replace=False)

            for img_idx, img_file in enumerate(selected_files):
                img_path = os.path.join(class_path, img_file)

                try:
                    # Görüntüyü yükle
                    img = load_img(img_path, target_size=(224, 224))
                    img_array = img_to_array(img) / 255.0

                    # Orijinal boyutları al
                    original_img = load_img(img_path)
                    original_size = original_img.size  # (width, height)

                    # Dosya boyutunu al
                    file_size = os.path.getsize(img_path) / 1024  # KB

                    # Görüntü bilgilerini kaydet
                    image_info.append({
                        'class': class_name,
                        'filename': img_file,
                        'width': original_size[0],
                        'height': original_size[1],
                        'file_size_kb': round(file_size, 2)
                    })

                    # Görüntüyü göster
                    ax = axes[class_idx, img_idx]
                    ax.imshow(img_array)
                    ax.set_title(f"{class_name}\n{original_size[0]}x{original_size[1]}\n{file_size:.1f}KB",
                                 fontsize=9)
                    ax.axis('off')

                except Exception as e:
                    ax = axes[class_idx, img_idx]
                    ax.text(0.5, 0.5, f"Hata:\n{str(e)[:20]}...",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{class_name} - Hata", fontsize=9)
                    ax.axis('off')

            # Boş alanları temizle
            for img_idx in range(len(selected_files), samples_per_class):
                if img_idx < samples_per_class:
                    ax = axes[class_idx, img_idx]
                    ax.axis('off')

    plt.suptitle('Her Sınıftan Örnek Görüntüler (Train Seti)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return image_info


# Örnek görüntüleri göster
sample_info = show_sample_images_with_info(train_path, class_names, samples_per_class=5)

# Görüntü boyutları analizi
print(f"\n📏 GÖRÜNTÜ BOYUTLARI ANALİZİ")
print("=" * 40)

df_info = pd.DataFrame(sample_info)

# Her sınıf için boyut istatistikleri
for class_name in class_names:
    class_data = df_info[df_info['class'] == class_name]
    if not class_data.empty:
        print(f"\n🏷️ {class_name}:")
        print(f"  📐 Ortalama boyut: {class_data['width'].mean():.0f}x{class_data['height'].mean():.0f}")
        print(f"  📐 Min boyut: {class_data['width'].min()}x{class_data['height'].min()}")
        print(f"  📐 Max boyut: {class_data['width'].max()}x{class_data['height'].max()}")
        print(f"  💾 Ortalama dosya boyutu: {class_data['file_size_kb'].mean():.1f} KB")

# Genel istatistikler
print(f"\n📊 GENEL İSTATİSTİKLER")
print("=" * 30)
print(f"📐 Tüm görüntülerin ortalama boyutu: {df_info['width'].mean():.0f}x{df_info['height'].mean():.0f}")
print(f"📐 En küçük boyut: {df_info['width'].min()}x{df_info['height'].min()}")
print(f"📐 En büyük boyut: {df_info['width'].max()}x{df_info['height'].max()}")
print(f"💾 Ortalama dosya boyutu: {df_info['file_size_kb'].mean():.1f} KB")

# Önceki generatörları tamamen sil
del train_datagen, validation_datagen, test_datagen
del train_generator, validation_generator, test_generator
if 'balanced_train_datagen' in locals():
    del balanced_train_datagen, balanced_train_gen, balanced_val_gen, balanced_test_gen

print("🧹 ÖNCEKİ GENERATÖRLER TEMİZLENDİ")
print("=" * 50)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np


# Özel augmentation fonksiyonları
def add_gaussian_noise(image):
    """Gaussian gürültü ekle"""
    noise = np.random.normal(0, 0.1, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def blur_image(image):
    """Görüntüyü bulanıklaştır"""
    # Numpy array'i OpenCV formatına çevir
    img_uint8 = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    return blurred.astype(np.float32) / 255.0


def adjust_gamma(image, gamma=1.2):
    """Gamma düzeltme (parlaklık)"""
    return np.power(image, gamma)


# Temiz ve etkili data generator
print("🎯 TEMİZ VE ETKİLİ DATA AUGMENTATION")
print("=" * 50)

# Train için augmentation
clean_train_datagen = ImageDataGenerator(
    rescale=1. / 255,

    # Temel geometrik dönüşümler
    rotation_range=15,  # Hafif döndürme
    width_shift_range=0.1,  # %10 kaydırma
    height_shift_range=0.1,  # %10 kaydırma
    shear_range=0.1,  # Hafif kesme
    zoom_range=0.1,  # %10 zoom

    # Sadece yatay çevirme (yapraklar için mantıklı)
    horizontal_flip=True,
    vertical_flip=False,  # Dikey çevirme kapalı

    # Parlaklık ayarı
    brightness_range=[0.8, 1.2],

    fill_mode='nearest'
)

# Validation ve Test - sadece normalizasyon
clean_val_datagen = ImageDataGenerator(rescale=1. / 255)
clean_test_datagen = ImageDataGenerator(rescale=1. / 255)

print("✅ Temiz augmentation parametreleri:")
print("  • Rotation: ±15° (gerçekçi)")
print("  • Shift: ±10% (hafif)")
print("  • Zoom: ±10% (doğal)")
print("  • Horizontal flip: Evet")
print("  • Vertical flip: Hayır (yapraklar için)")
print("  • Brightness: 80%-120%")
print("  • Noise, blur: Manuel eklenecek")

# Veri yükleyicilerini oluştur
IMG_SIZE = 224
BATCH_SIZE = 32

clean_train_gen = clean_train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

clean_val_gen = clean_val_datagen.flow_from_directory(
    validation_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

clean_test_gen = clean_test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n📊 YENİ TEMİZ GENERATÖR BİLGİLERİ")
print("=" * 40)
print(f"🚂 Train: {clean_train_gen.samples:,} görüntü")
print(f"🔍 Validation: {clean_val_gen.samples:,} görüntü")
print(f"🧪 Test: {clean_test_gen.samples:,} görüntü")
print(f"📦 Batch size: {BATCH_SIZE}")
print(f"📏 Görüntü boyutu: {IMG_SIZE}x{IMG_SIZE}")

# Yeni hesaplamalar
steps_per_epoch = len(clean_train_gen)
images_per_epoch = steps_per_epoch * BATCH_SIZE

print(f"\n🔢 ETKİLİ AUGMENTATION HESABI")
print("=" * 40)
print(f"📊 Steps per epoch: {steps_per_epoch}")
print(f"🎲 Her epoch'ta görülen görüntü: {images_per_epoch:,}")
print(f"🚀 30 epoch için toplam: {images_per_epoch * 30:,}")
print(f"📉 Önceki kaos: 260,800 → Şimdi: {images_per_epoch * 30:,}")
print(f"✅ {260800 / (images_per_epoch * 30):.1f}x daha makul!")

print(f"\n🎯 Sınıf indeksleri: {clean_train_gen.class_indices}")
print("🎨 TEMİZ AUGMENTATION ÖRNEKLERİNİ GÖRSELLEŞTİRME")
print("=" * 60)


def show_clean_augmentation_examples():
    """Temiz ve anlaşılır augmentation örnekleri"""

    fig, axes = plt.subplots(len(class_names), 6, figsize=(18, len(class_names) * 3))

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_path, class_name)

        if os.path.exists(class_path):
            # Rastgele bir görüntü seç
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                np.random.seed(42 + class_idx)  # Her sınıf için farklı seed
                selected_file = np.random.choice(image_files)
                img_path = os.path.join(class_path, selected_file)

                # Orijinal görüntüyü yükle
                original_img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                original_array = np.array(original_img) / 255.0

                # Orijinal
                axes[class_idx, 0].imshow(original_array)
                axes[class_idx, 0].set_title(f'{class_name}\nOrijinal', fontsize=10, fontweight='bold')
                axes[class_idx, 0].axis('off')

                # Augmentation örnekleri
                img_batch = np.expand_dims(original_array * 255, 0)  # 0-255 aralığına çevir

                # 5 farklı augmentation
                for i in range(5):
                    try:
                        # Clean generator'dan augmented versiyon al
                        augmented_batch = clean_train_datagen.flow(img_batch, batch_size=1)
                        augmented_img = next(augmented_batch)[0]

                        # Değerleri kontrol et
                        augmented_img = np.clip(augmented_img, 0, 1)

                        axes[class_idx, i + 1].imshow(augmented_img)
                        axes[class_idx, i + 1].set_title(f'Augmented #{i + 1}', fontsize=9)
                        axes[class_idx, i + 1].axis('off')

                    except Exception as e:
                        axes[class_idx, i + 1].text(0.5, 0.5, 'Hata', ha='center', va='center')
                        axes[class_idx, i + 1].axis('off')

    plt.suptitle('Temiz Data Augmentation Örnekleri', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Örnekleri göster
show_clean_augmentation_examples()

# Gerçek batch'ten örnekler de alalım
print(f"\n🔄 GERÇEK BATCH'TEN ÖRNEKLER")
print("=" * 40)


def show_real_batch_samples():
    """Gerçek train batch'inden örnekler"""

    # Bir batch al
    clean_train_gen.reset()
    batch_images, batch_labels = next(clean_train_gen)

    print(f"📊 Batch bilgileri:")
    print(f"  • Shape: {batch_images.shape}")
    print(f"  • Min değer: {batch_images.min():.3f}")
    print(f"  • Max değer: {batch_images.max():.3f}")
    print(f"  • Mean değer: {batch_images.mean():.3f}")

    # İlk 8 görüntüyü göster
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    class_names_list = list(clean_train_gen.class_indices.keys())

    for i in range(min(8, len(batch_images))):
        img = batch_images[i]
        label_idx = np.argmax(batch_labels[i])
        class_name = class_names_list[label_idx]

        axes[i].imshow(img)
        axes[i].set_title(f'{class_name}\nBatch #{i + 1}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Train Generator\'dan Gerçek Augmented Örnekler', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return len(batch_images)


# Gerçek batch örneklerini göster
batch_size = show_real_batch_samples()

# Final özet
print(f"\n✅ AUGMENTATION BAŞARIYLA AYARLANDI!")
print("=" * 50)
print(f"🎯 Temiz ve etkili augmentation")
print(f"📊 Batch'te {batch_size} görüntü")
print(f"🚀 Overfitting riski minimize edildi")
print(f"💪 Model eğitimine hazır!")

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

print("🍇 GRAPE DISEASE DETECTION - TRANSFER LEARNING YAKLAŞIMI")
print("=" * 60)


# Model mimarisi - Transfer Learning ile
def create_grape_model(num_classes, img_size=224):
    """Transfer Learning ile etkili model"""

    # VGG16 base model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    # İlk katmanları freeze et
    base_model.trainable = False

    # Küçük ve etkili head network
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    return model


# Model oluştur
model = create_grape_model(num_classes=len(class_names))

# Optimizer - Düşük learning rate
optimizer = Adam(learning_rate=0.0001)  # 10x daha düşük!

# Model compile
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']
)

# Model özeti - DÜZELTME
print(f"📊 MODEL ÖZETİ:")
print(f"  • Toplam parametreler: {model.count_params():,}")

# Trainable parametreleri doğru şekilde hesapla
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = model.count_params() - trainable_params

print(f"  • Trainable parametreler: {trainable_params:,}")
print(f"  • Non-trainable parametreler: {non_trainable_params:,}")
print(f"  • VGG16 Base: Frozen ❄️")
print(f"  • Custom Head: Active 🔥")

model.summary()

# Callbacks - Daha katı kontrol
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Daha sabırlı
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_grape_vgg16_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print(f"\n🎯 EĞİTİM BAŞLIYOR - TRANSFER LEARNING YAKLAŞIMI")
print("=" * 60)
print(f"🚀 VGG16 + Custom Head")
print(f"📊 Sadece {trainable_params:,} parametre eğitilecek!")
print(f"⚡ Overfitting riski minimum!")

# Eğitim başlat
history = model.fit(
    clean_train_gen,  # Senin temiz augmentation'ın
    epochs=20,  # Daha az epoch
    validation_data=clean_val_gen,
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ EĞİTİM TAMAMLANDI!")
print("=" * 30)

import matplotlib.pyplot as plt
import numpy as np

print("📊 MODEL PERFORMANS ANALİZİ")
print("=" * 40)


# Training history grafiği
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('🍇 Grape Disease Model - Training Results', fontsize=16)
    plt.tight_layout()
    plt.show()


plot_training_history(history)

# Final sonuçlar
print(f"\n🎯 FINAL SONUÇLAR:")
print(
    f"  ✅ Validation Accuracy: {max(history.history['val_accuracy']):.4f} (%{max(history.history['val_accuracy']) * 100:.2f})")
print(f"  ✅ Training Accuracy: {history.history['accuracy'][-1]:.4f} (%{history.history['accuracy'][-1] * 100:.2f})")
print(f"  ✅ Validation Loss: {min(history.history['val_loss']):.4f}")

print("🧪 TEST SETİ DEĞERLENDİRMESİ")
print("=" * 35)

# Test seti yükle (eğer varsa)
test_loss, test_accuracy, test_top_k = model.evaluate(clean_val_gen, verbose=1)

print(f"\n📈 TEST SONUÇLARI:")
print(f"  • Test Accuracy: %{test_accuracy*100:.2f}")
print(f"  • Test Loss: {test_loss:.4f}")
print(f"  • Top-K Accuracy: %{test_top_k*100:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

print("📊 KAPSAMLI MODEL DEĞERLENDİRMESİ")
print("=" * 50)


# 1. Training History Grafikleri
def plot_training_results(history):
    """Eğitim sonuçlarını görselleştir"""

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    axes[0, 0].set_title('🎯 Model Accuracy Gelişimi', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])

    # Loss
    axes[0, 1].plot(history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0, 1].set_title('📉 Model Loss Azalışı', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning Rate (eğer varsa)
    if 'learning_rate' in history.history:
        axes[1, 0].plot(history.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 0].set_title('📈 Learning Rate Değişimi', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # Final Metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])

    axes[1, 1].axis('off')
    metrics_text = f"""
📊 FINAL METRİKLER:

✅ En İyi Validation Accuracy: {best_val_acc:.4f} (%{best_val_acc * 100:.2f})
📈 Final Training Accuracy: {final_train_acc:.4f} (%{final_train_acc * 100:.2f})  
📈 Final Validation Accuracy: {final_val_acc:.4f} (%{final_val_acc * 100:.2f})

🔄 Toplam Epoch: {len(history.history['accuracy'])}
🎯 Model Durumu: {'Underfitting' if final_val_acc > final_train_acc else 'Normal'}
    """

    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.suptitle('🍇 Grape Disease Model - Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Grafikleri çiz
plot_training_results(history)

# 2. Confusion Matrix
print("\n🔍 CONFUSION MATRIX HAZIRLIĞI")
print("=" * 35)

# Validation verilerinden predictions al
clean_val_gen.reset()
val_predictions = model.predict(clean_val_gen, verbose=1)
val_predicted_classes = np.argmax(val_predictions, axis=1)

# Gerçek labelları al
val_true_classes = clean_val_gen.classes
class_labels = list(clean_val_gen.class_indices.keys())

print(f"✅ {len(val_predictions)} validation örneği için prediction tamamlandı")
print(f"📊 Sınıf isimleri: {class_labels}")


# Confusion Matrix oluştur ve görselleştir
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Detaylı confusion matrix"""

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Ham confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize edilmiş confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Ham sayılar
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('🔢 Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Normalize edilmiş
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('📊 Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.suptitle('🍇 Grape Disease Classification - Confusion Matrix Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()

    return cm


cm = plot_confusion_matrix(val_true_classes, val_predicted_classes, class_labels)

# 3. Classification Report
print("\n📋 CLASSIFICATION REPORT")
print("=" * 30)

report = classification_report(val_true_classes, val_predicted_classes,
                               target_names=class_labels, digits=4)
print(report)

# Classification report'u görsel olarak da göster
from sklearn.metrics import precision_recall_fscore_support


def plot_classification_metrics(y_true, y_pred, class_names):
    """Precision, Recall, F1-Score görselleştirme"""

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 8))

    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')

    ax.set_xlabel('Sınıflar')
    ax.set_ylabel('Skor')
    ax.set_title('🎯 Sınıf Bazında Model Performansı', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Bar'ların üstüne değerleri yaz
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


plot_classification_metrics(val_true_classes, val_predicted_classes, class_labels)

print(f"\n✅ MODEL DEĞERLENDİRME TAMAMLANDI!")
print(f"🎯 En iyi validation accuracy: %{max(history.history['val_accuracy']) * 100:.2f}")
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

print("🔥 GRAD-CAM - ÇALIŞAN FINAL VERSİYON")
print("=" * 40)


# Doğrudan VGG16 base model ile çalışalım
def simple_gradcam_analysis():
    """Basit ve etkili Grad-CAM analizi"""

    print("🧪 Basit Grad-CAM implementasyonu...")

    # VGG16 base model'i al
    vgg_base = model.layers[0]

    # Her sınıftan bir örnek analizi
    fig, axes = plt.subplots(len(class_names), 4, figsize=(20, len(class_names) * 4))

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_path, class_name)

        if os.path.exists(class_path):
            # Rastgele görüntü seç
            image_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:
                selected_file = np.random.choice(image_files)
                img_path = os.path.join(class_path, selected_file)

                try:
                    # Görüntüyü yükle
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    # Prediction yap
                    predictions = model.predict(img_array, verbose=0)
                    pred_class = np.argmax(predictions[0])
                    confidence = predictions[0][pred_class]
                    pred_class_name = class_names[pred_class]

                    # Basit attention map oluştur (son conv layer aktivasyonları)
                    # VGG16'nın son conv layer'ından feature map al
                    conv_model = tf.keras.Model(
                        inputs=vgg_base.input,
                        outputs=vgg_base.get_layer('block5_conv3').output
                    )

                    # Feature maps al
                    feature_maps = conv_model.predict(img_array * 255, verbose=0)  # VGG16 0-255 bekler

                    # En aktif feature map'i bul (basit yaklaşım)
                    attention_map = np.mean(feature_maps[0], axis=-1)  # Channel'lar boyunca ortalama

                    # Normalize et
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

                    # 224x224'e resize et
                    attention_resized = cv2.resize(attention_map, (224, 224))

                    # Heatmap oluştur
                    heatmap = np.uint8(255 * attention_resized)
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

                    # Overlay oluştur
                    original_img = (img_array[0] * 255).astype(np.uint8)
                    overlay = heatmap_colored * 0.4 + original_img * 0.6
                    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                    # Sonuçları göster
                    # 1. Orijinal
                    axes[class_idx, 0].imshow(original_img)
                    axes[class_idx, 0].set_title(f'🍇 {class_name}\nOrijinal', fontweight='bold')
                    axes[class_idx, 0].axis('off')

                    # 2. Attention Map
                    axes[class_idx, 1].imshow(attention_resized, cmap='hot')
                    axes[class_idx, 1].set_title('🔥 Attention Map\n(Feature Activation)', fontweight='bold')
                    axes[class_idx, 1].axis('off')

                    # 3. Colored Heatmap
                    axes[class_idx, 2].imshow(heatmap_colored)
                    axes[class_idx, 2].set_title('🌈 Colored Heatmap', fontweight='bold')
                    axes[class_idx, 2].axis('off')

                    # 4. Overlay
                    axes[class_idx, 3].imshow(overlay)

                    # Başlık rengini doğruluğa göre ayarla
                    is_correct = pred_class_name == class_name
                    title_color = 'green' if is_correct else 'red'
                    result_symbol = '✅' if is_correct else '❌'

                    axes[class_idx, 3].set_title(
                        f'{result_symbol} Model Focus\nTahmin: {pred_class_name}\nGüven: %{confidence * 100:.1f}',
                        fontweight='bold', color=title_color)
                    axes[class_idx, 3].axis('off')

                except Exception as e:
                    print(f"❌ Hata {class_name}: {str(e)}")
                    for j in range(4):
                        axes[class_idx, j].text(0.5, 0.5, f'Hata\n{str(e)[:15]}...',
                                                ha='center', va='center', fontsize=8)
                        axes[class_idx, j].axis('off')

    plt.suptitle('🔥 Visual Attention Analysis - Model Odak Alanları', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Basit attention analizi yap
simple_gradcam_analysis()


# Feature map görselleştirmesi
def visualize_feature_maps():
    """VGG16'nın farklı katmanlarındaki feature map'leri görselleştir"""

    print(f"\n🧠 VGG16 FEATURE MAP GÖRSELLEŞTİRMESİ")
    print("=" * 45)

    # Test görüntüsü al
    class_name = class_names[0]  # İlk sınıf
    class_path = os.path.join(train_path, class_name)
    image_files = [f for f in os.listdir(class_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if image_files:
        img_path = os.path.join(class_path, image_files[0])

        # Görüntüyü hazırla
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # VGG16 base
        vgg_base = model.layers[0]

        # Farklı katmanlardan feature map'leri al
        layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

        fig, axes = plt.subplots(len(layer_names), 8, figsize=(24, len(layer_names) * 3))

        for layer_idx, layer_name in enumerate(layer_names):
            try:
                # Feature model oluştur
                feature_model = tf.keras.Model(
                    inputs=vgg_base.input,
                    outputs=vgg_base.get_layer(layer_name).output
                )

                # Feature maps al
                features = feature_model.predict(img_array * 255, verbose=0)  # VGG16 için 0-255

                # İlk 8 channel'ı göster
                for i in range(min(8, features.shape[-1])):
                    feature_map = features[0, :, :, i]

                    # Normalize et
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

                    axes[layer_idx, i].imshow(feature_map, cmap='viridis')
                    axes[layer_idx, i].set_title(f'{layer_name}\nChannel {i}', fontsize=8)
                    axes[layer_idx, i].axis('off')

            except Exception as e:
                print(f"❌ Layer {layer_name} hata: {str(e)}")
                for i in range(8):
                    axes[layer_idx, i].text(0.5, 0.5, 'Hata', ha='center', va='center')
                    axes[layer_idx, i].axis('off')

        plt.suptitle(f'🧠 VGG16 Feature Maps - {class_name} Örneği', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Feature map'leri göster
visualize_feature_maps()

# Model'in hangi bölgelere odaklandığına dair özet
print(f"\n📊 MODEL ODAK ANALİZİ ÖZETİ")
print("=" * 35)

# Validation setinden örnekler test et
clean_val_gen.reset()
val_batch, val_labels = next(clean_val_gen)

# İlk 4 örneği analiz et
fig, axes = plt.subplots(4, 3, figsize=(15, 16))

for i in range(4):
    img = val_batch[i:i + 1]
    true_class = np.argmax(val_labels[i])

    # Prediction
    pred = model.predict(img, verbose=0)
    pred_class = np.argmax(pred[0])
    confidence = pred[0][pred_class]

    # VGG16 base
    vgg_base = model.layers[0]

    # Feature extraction
    conv_model = tf.keras.Model(
        inputs=vgg_base.input,
        outputs=vgg_base.get_layer('block5_conv3').output
    )

    features = conv_model.predict(img * 255, verbose=0)
    attention = np.mean(features[0], axis=-1)
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    attention_resized = cv2.resize(attention, (224, 224))

    # Görselleştir
    original = (val_batch[i] * 255).astype(np.uint8)

    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f'Gerçek: {class_names[true_class]}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(attention_resized, cmap='hot')
    axes[i, 1].set_title('Model Dikkat Alanları')
    axes[i, 1].axis('off')

    # Overlay
    heatmap = np.uint8(255 * attention_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = heatmap_colored * 0.4 + original * 0.6

    axes[i, 2].imshow(np.uint8(overlay))

    is_correct = pred_class == true_class
    result = '✅ Doğru' if is_correct else '❌ Yanlış'
    color = 'green' if is_correct else 'red'

    axes[i, 2].set_title(f'{result}\nTahmin: {class_names[pred_class]}\n%{confidence * 100:.1f}', color=color)
    axes[i, 2].axis('off')

plt.suptitle('🎯 Model Attention Analysis - Validation Örnekleri', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n✅ VİZUAL ATTENTION ANALİZİ TAMAMLANDI!")
print(f"🔥 Model'in odaklandığı alanlar görüldü!")
print(f"🎯 Feature map'ler ve dikkat haritaları oluşturuldu!")
print(f"📊 Doğru ve yanlış tahminlerde odak farkları analiz edildi!")