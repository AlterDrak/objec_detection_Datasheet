# =============================================================
# GOOGLE COLAB - YOLOv8 Waste Bin Fill Level Detection Training
# =============================================================
# Copy setiap bagian (CELL) ke cell terpisah di Google Colab
# Pastikan Runtime > Change runtime type > GPU (T4)
#
# Dataset: download dari Roboflow format YOLOv8
# Upload ZIP dataset ke Google Drive > Project Uas ML
# =============================================================


# %% [CELL 1] - Install Dependencies
# ====================================
# Paste ke cell pertama di Colab (hapus tanda # di depan !pip):
#
# !pip install ultralytics==8.3.40


# %% [CELL 2] - Mount Google Drive & Import
# ===========================================
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import zipfile
from pathlib import Path
from collections import Counter

print("✅ Google Drive mounted!")


# %% [CELL 3] - Konfigurasi Path
# =================================
# ⚠️ SESUAIKAN PATH INI DENGAN LOKASI DI GOOGLE DRIVE KAMU

# Path folder Project Uas ML di Google Drive
DRIVE_PROJECT = Path("/content/drive/MyDrive/Project Uas ML")

# Nama file ZIP dataset YOLOv8 yang sudah di-download dari Roboflow
# ⚠️ GANTI NAMA FILE INI sesuai nama file ZIP yang kamu upload ke Google Drive
DATASET_ZIP_NAME = "waste-bin-fill-level-detect2.yolov8.zip"

# Working directory di Colab (jangan diubah)
WORK_DIR = Path("/content/waste_detection")
DATASET_DIR = WORK_DIR / "dataset"

print(f"📁 Project folder: {DRIVE_PROJECT}")
print(f"📦 Dataset ZIP: {DATASET_ZIP_NAME}")
print(f"🔧 Work dir: {WORK_DIR}")

# Cek apakah project folder ada
if DRIVE_PROJECT.exists():
    print("\n✅ Project folder ditemukan!")
    print("   Isi folder:")
    for item in sorted(DRIVE_PROJECT.iterdir()):
        print(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
else:
    print("\n❌ Project folder tidak ditemukan! Cek path-nya.")


# %% [CELL 4] - Extract Dataset
# ================================
print("=" * 60)
print("  STEP 1: Extract Dataset YOLOv8")
print("=" * 60)

# Bersihkan folder sebelumnya
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Cari file ZIP dataset
zip_path = DRIVE_PROJECT / DATASET_ZIP_NAME

if not zip_path.exists():
    # Coba cari file zip apapun yang ada di folder
    zip_files = list(DRIVE_PROJECT.glob("*.zip"))
    print(f"⚠️ File '{DATASET_ZIP_NAME}' tidak ditemukan!")
    print(f"   File ZIP yang tersedia:")
    for zf in zip_files:
        print(f"   📦 {zf.name} ({zf.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"\n   ⚠️ Ganti DATASET_ZIP_NAME di Cell 3 dengan nama file yang benar!")
else:
    print(f"📦 Extracting: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATASET_DIR)
    print(f"✅ Extracted ke: {DATASET_DIR}")

    # Tampilkan isi folder
    print("\n📁 Isi dataset:")
    for item in sorted(DATASET_DIR.rglob("*")):
        if item.is_file() and not item.name.startswith('.'):
            rel = item.relative_to(DATASET_DIR)
            print(f"   {rel}")
        elif item.is_dir():
            rel = item.relative_to(DATASET_DIR)
            print(f"   📁 {rel}/")


# %% [CELL 5] - Cari dan Validasi data.yaml
# =============================================
import yaml

print("=" * 60)
print("  STEP 2: Cari & Validasi data.yaml")
print("=" * 60)

# Cari data.yaml (bisa di root atau subfolder)
yaml_files = list(DATASET_DIR.rglob("data.yaml"))

if not yaml_files:
    print("❌ data.yaml tidak ditemukan!")
    print("   Pastikan dataset di-download dalam format YOLOv8")
else:
    yaml_path = yaml_files[0]
    print(f"✅ Ditemukan: {yaml_path}")

    # Baca data.yaml
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"\n📋 Isi data.yaml:")
    for key, value in data_config.items():
        print(f"   {key}: {value}")

    # Update path di data.yaml agar sesuai lokasi di Colab
    # Tentukan base directory (parent dari data.yaml)
    dataset_base = yaml_path.parent

    data_config['path'] = str(dataset_base)

    # Simpan ulang data.yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Path di data.yaml sudah diupdate ke: {dataset_base}")

    # Verifikasi folder train/valid/test
    print(f"\n📊 Verifikasi dataset:")
    for split_key in ['train', 'val', 'test']:
        split_val = data_config.get(split_key, '')

        # Handle path (bisa relative atau absolute)
        if split_val:
            split_path = dataset_base / split_val
            if split_path.exists():
                # Hitung images
                img_count = len(list(split_path.glob("*.jpg"))) + \
                            len(list(split_path.glob("*.png"))) + \
                            len(list(split_path.glob("*.jpeg")))
                print(f"   ✅ {split_key}: {img_count} images ({split_path})")
            else:
                # Mungkin folder images ada di dalam split
                alt_path = dataset_base / split_val.replace('/images', '')
                if alt_path.exists():
                    img_count = len(list(alt_path.glob("*.jpg")))
                    print(f"   ✅ {split_key}: {img_count} images ({alt_path})")
                else:
                    print(f"   ❌ {split_key}: folder tidak ditemukan ({split_path})")
        else:
            print(f"   ⚠️ {split_key}: tidak ada di data.yaml")

    # Tampilkan class names
    names = data_config.get('names', {})
    print(f"\n📊 Classes ({data_config.get('nc', '?')} total):")
    if isinstance(names, list):
        for i, name in enumerate(names):
            print(f"   {i}: {name}")
    elif isinstance(names, dict):
        for k, v in names.items():
            print(f"   {k}: {v}")


# %% [CELL 6] - Cek Distribusi Label
# ======================================
print("=" * 60)
print("  STEP 3: Cek Distribusi Label")
print("=" * 60)

# Dapatkan class names
names = data_config.get('names', {})
if isinstance(names, list):
    class_names = names
elif isinstance(names, dict):
    class_names = [names[k] for k in sorted(names.keys())]
else:
    class_names = []

print(f"📋 Classes: {class_names}")

# Hitung label per class di setiap split
for split_key in ['train', 'val', 'test']:
    split_val = data_config.get(split_key, '')
    if not split_val:
        continue

    # Cari folder labels yang sesuai
    img_path = dataset_base / split_val
    lbl_path_candidate = str(img_path).replace('images', 'labels')
    lbl_path = Path(lbl_path_candidate)

    if not lbl_path.exists():
        print(f"\n⚠️ {split_key}: folder labels tidak ditemukan ({lbl_path})")
        continue

    label_files = list(lbl_path.glob("*.txt"))
    class_counts = Counter()

    for lbl_file in label_files:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(class_names):
                        class_counts[class_names[cls_id]] += 1
                    else:
                        class_counts[f"unknown_{cls_id}"] += 1

    total = sum(class_counts.values())
    print(f"\n📊 {split_key} ({len(label_files)} files, {total} annotations):")
    for cls_name in class_names:
        count = class_counts.get(cls_name, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {cls_name:>15}: {count:>4} ({pct:5.1f}%) {bar}")


# %% [CELL 7] - Training YOLOv8
# ==================================
from ultralytics import YOLO

print("=" * 60)
print("  STEP 4: Training YOLOv8")
print("=" * 60)
print("  ⏱️ Estimasi waktu: 30-60 menit (tergantung GPU)")
print("=" * 60)

# Load pretrained YOLOv8 nano model
model = YOLO('yolov8n.pt')

# Training dengan parameter yang dioptimasi
results = model.train(
    data=str(yaml_path),

    # === Epoch & Batch ===
    epochs=100,           # Lebih banyak epoch untuk konvergensi
    batch=16,             # Batch size (sesuaikan jika out of memory, turunkan ke 8)
    imgsz=640,            # Image size

    # === Optimizer ===
    optimizer='auto',
    lr0=0.01,             # Learning rate awal
    lrf=0.01,             # Learning rate akhir (fraction)
    momentum=0.937,
    weight_decay=0.0005,

    # === Early Stopping ===
    patience=25,          # Stop jika tidak improve setelah 25 epoch

    # === Augmentation (penting untuk dataset kecil!) ===
    augment=True,
    hsv_h=0.015,          # Hue augmentation
    hsv_s=0.7,            # Saturation augmentation
    hsv_v=0.4,            # Value/brightness augmentation
    degrees=10.0,         # Rotation ±10°
    translate=0.1,        # Translation ±10%
    scale=0.5,            # Scale ±50%
    fliplr=0.5,           # Horizontal flip 50%
    flipud=0.0,           # No vertical flip
    mosaic=1.0,           # Mosaic augmentation
    mixup=0.1,            # Mixup augmentation

    # === Output ===
    project=str(WORK_DIR / 'runs'),
    name='waste_bin_detector',
    exist_ok=True,
    plots=True,
    save=True,
    verbose=True,

    # === Device ===
    device=0,             # GPU (0 = pertama)
)

print("\n🎉 Training selesai!")


# %% [CELL 8] - Evaluasi Model
# =================================
print("=" * 60)
print("  STEP 5: Evaluasi Model")
print("=" * 60)

# Load model terbaik
best_model_path = WORK_DIR / 'runs' / 'waste_bin_detector' / 'weights' / 'best.pt'
best_model = YOLO(str(best_model_path))

print(f"📊 Model classes: {best_model.names}")

# Evaluasi pada validation set
print("\n" + "-" * 40)
print("📊 Hasil Validasi:")
print("-" * 40)
val_results = best_model.val(data=str(yaml_path), split='val')

# Evaluasi pada test set (jika ada)
if data_config.get('test'):
    print("\n" + "-" * 40)
    print("📊 Hasil Test:")
    print("-" * 40)
    test_results = best_model.val(data=str(yaml_path), split='test')


# %% [CELL 9] - Visualisasi Hasil Prediksi
# ============================================
import matplotlib.pyplot as plt
import cv2

print("=" * 60)
print("  STEP 6: Test Prediksi Visual")
print("=" * 60)

# Ambil gambar test (atau valid jika test kosong)
test_split = data_config.get('test', data_config.get('val', ''))
test_img_dir = dataset_base / test_split

test_images = list(test_img_dir.glob('*.jpg'))[:6]
if not test_images:
    test_images = list(test_img_dir.glob('*.png'))[:6]

if test_images:
    n_cols = min(3, len(test_images))
    n_rows = (len(test_images) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, img_path in enumerate(test_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # Prediksi
        results = best_model(str(img_path), conf=0.25)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        ax.imshow(annotated_rgb)
        ax.set_title(img_path.name[:30], fontsize=10)
        ax.axis('off')

        # Print detections
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = best_model.names[cls_id]
            print(f"  {img_path.name[:30]}: {cls_name} ({conf:.0%})")

    # Hide empty subplots
    for idx in range(len(test_images), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Tidak ada gambar test")


# %% [CELL 10] - Tampilkan Training Metrics
# =============================================
from IPython.display import Image, display

print("=" * 60)
print("  STEP 7: Training Metrics")
print("=" * 60)

results_dir = WORK_DIR / 'runs' / 'waste_bin_detector'

metrics = [
    ('Confusion Matrix', 'confusion_matrix.png'),
    ('Confusion Matrix (Normalized)', 'confusion_matrix_normalized.png'),
    ('Training Results', 'results.png'),
    ('Precision-Recall Curve', 'BoxPR_curve.png'),
    ('F1 Curve', 'BoxF1_curve.png'),
]

for title, filename in metrics:
    filepath = results_dir / filename
    if filepath.exists():
        print(f"\n📊 {title}:")
        display(Image(filename=str(filepath), width=700))
    else:
        print(f"⚠️ {title}: file tidak ditemukan")


# %% [CELL 11] - Simpan Model ke Google Drive
# ================================================
print("=" * 60)
print("  STEP 8: Simpan Model ke Google Drive")
print("=" * 60)

# Path output di Google Drive
output_dir = DRIVE_PROJECT / 'model_output'
output_dir.mkdir(parents=True, exist_ok=True)

# File yang akan disimpan
files_to_save = [
    ('weights/best.pt', 'best.pt'),
    ('weights/last.pt', 'last.pt'),
    ('confusion_matrix.png', 'confusion_matrix.png'),
    ('confusion_matrix_normalized.png', 'confusion_matrix_normalized.png'),
    ('results.png', 'results.png'),
    ('BoxPR_curve.png', 'BoxPR_curve.png'),
    ('BoxF1_curve.png', 'BoxF1_curve.png'),
    ('args.yaml', 'args.yaml'),
]

for src_rel, dst_name in files_to_save:
    src = results_dir / src_rel
    dst = output_dir / dst_name
    if src.exists():
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  ✅ {dst_name} ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️ {dst_name} tidak ditemukan")

print(f"\n📁 Semua file disimpan di: {output_dir}")


# %% [CELL 12] - Verifikasi Final & Instruksi
# ================================================
print("=" * 60)
print("  VERIFIKASI FINAL")
print("=" * 60)

# Load dan verifikasi model
final_model = YOLO(str(output_dir / 'best.pt'))
print(f"\n📋 Class names di model yang sudah di-training:")
for k, v in final_model.names.items():
    print(f"   {k}: {v}")

print(f"""
{'=' * 60}
  ✅ TRAINING SELESAI!
{'=' * 60}

📋 Langkah selanjutnya:
   1. Download best.pt dari Google Drive:
      📁 {output_dir / 'best.pt'}

   2. Copy/replace ke folder website:
      📁 c:\\xampp\\htdocs\\deteksi_sampah\\best.pt

   3. Pastikan app.py sudah menggunakan class names yang sesuai
      dengan model baru ini (lihat daftar class di atas)

   4. Restart Flask server:
      > py app.py

   5. Test deteksi di http://127.0.0.1:5000
{'=' * 60}
""")
