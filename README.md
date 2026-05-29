# 🗑️ Smart Waste Detection

> Sistem deteksi kepenuhan tong sampah berbasis **YOLOv8** dan **Flask**  
> Proyek B — Bin Fill-Level Detection | Mata Kuliah Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/Dataset-CC%20BY%204.0-green)

---

## 📋 Deskripsi

**Smart Waste Detection** adalah aplikasi web yang mendeteksi dan mengklasifikasikan kondisi kepenuhan tong sampah secara otomatis menggunakan model Object Detection berbasis YOLOv8n. Sistem dapat mengenali 4 kondisi tong sampah:

| Label | Kondisi | Deskripsi |
|---|---|---|
| `empty` | 🟢 Kosong | Tong belum perlu dikosongkan |
| `half-full` | 🟡 Setengah Penuh | Masih aman, perlu dipantau |
| `full` | 🟠 Penuh | Perlu segera dijadwalkan pengangkutan |
| `overflowing` | 🔴 Meluap | Penanganan darurat diperlukan |

---

## ✨ Fitur

- 📤 **Upload Gambar** — Upload foto tong sampah (JPG/PNG), dapatkan hasil deteksi dengan bounding box, label kelas, dan confidence score
- 📷 **Live Camera** — Deteksi real-time menggunakan kamera perangkat via browser (WebRTC)
- 📊 **Detail Objek** — Tampilkan jumlah objek terdeteksi per kelas dalam satu gambar
- ⚡ **Fast Inference** — Menggunakan YOLOv8n (nano), inferensi cepat di CPU maupun GPU
- 🖼️ **Perbandingan Visual** — Gambar asli dan hasil anotasi YOLOv8 ditampilkan berdampingan

---

## 🗂️ Struktur Folder

```
smart-waste-ml/
├── app.py                  # Backend Flask utama
├── best.pt                 # Model YOLOv8 terlatih (wajib ada)
├── requirements.txt        # Dependensi Python
├── templates/
│   └── index.html          # Antarmuka web (HTML + CSS + JS)
├── static/
│   ├── uploads/            # Gambar yang diupload pengguna
│   └── results/            # Gambar hasil anotasi YOLOv8
└── README.md
```

---

## ⚙️ Instalasi & Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/[username]/smart-waste-ml.git
cd smart-waste-ml
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install flask ultralytics opencv-python numpy
```

### 3. Letakkan model

Salin file `best.pt` hasil training ke dalam folder project (satu level dengan `app.py`):

```
smart-waste-ml/
├── app.py
├── best.pt   ← letakkan di sini
```

> **Download model:** [Google Drive — best.pt](#) *(ganti dengan link aktual)*

### 4. Jalankan server

```bash
python app.py
```

### 5. Buka di browser

```
http://127.0.0.1:5000
```

---

## 🚀 Cara Penggunaan

### Mode Upload Gambar
1. Klik tab **Upload Gambar**
2. Drag & drop foto tong sampah, atau klik **Pilih Gambar**
3. Klik tombol **Deteksi Sekarang**
4. Lihat hasil: status kepenuhan, confidence score, dan gambar anotasi

### Mode Live Camera
1. Klik tab **Live Camera**
2. Klik **Start Camera** — izinkan akses kamera di browser
3. Klik **Start Deteksi** — deteksi berjalan otomatis per frame
4. Lihat hasil deteksi real-time di layar

---

## 🤖 Model & Dataset

| Aspek | Detail |
|---|---|
| **Arsitektur** | YOLOv8n (You Only Look Once v8 — Nano) |
| **Bobot Awal** | Pre-trained COCO (Transfer Learning) |
| **Dataset** | waste-bin-fill-level-detect2 v1 (Roboflow) |
| **Lisensi Dataset** | CC BY 4.0 |
| **Jumlah Gambar** | 469 gambar (327 train / 98 val / 44 test) |
| **Jumlah Kelas** | 4 kelas (empty, half-full, full, overflowing) |
| **Platform Training** | Google Colab — GPU NVIDIA T4 |
| **Input Size** | 640 × 640 px |
| **Framework** | Ultralytics YOLOv8 v8.3.40 |

**Link dataset:** https://universe.roboflow.com/image-data-mlcpz/waste-bin-fill-level-detect2-lxsf4-u3odm

---

## 📦 requirements.txt

```
flask
ultralytics==8.3.40
opencv-python
numpy
```

---

## 📁 API Endpoint

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Halaman utama aplikasi |
| `POST` | `/` | Upload gambar dan jalankan deteksi |
| `POST` | `/detect_frame` | Deteksi frame live camera (JSON base64) |

### Contoh response `/detect_frame`

```json
{
  "status": "full",
  "confidence": 87.76,
  "total_objects": 1,
  "full_count": 1,
  "half_count": 0,
  "empty_count": 0,
  "overflow_count": 0,
  "detection_time": 0.123,
  "detections": [
    {
      "class": "full",
      "confidence": 87.8,
      "bbox": { "x1": 0.12, "y1": 0.08, "x2": 0.88, "y2": 0.95 }
    }
  ]
}
```

---

## 👥 Anggota Tim

| Nama | NIM | Kontribusi |
|---|---|---|
| [Nama Anggota 1] | [NIM] | Dataset, Training Model |
| [Nama Anggota 2] | [NIM] | Pengembangan Aplikasi Web |
| [Nama Anggota 3] | [NIM] | Evaluasi Model, Dokumentasi |

---

## 🔗 Referensi

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Dataset Roboflow: https://universe.roboflow.com/image-data-mlcpz/waste-bin-fill-level-detect2-lxsf4-u3odm
- Flask Documentation: https://flask.palletsprojects.com

---

> *"Artificial Intelligence is not just about code; it's about solving human problems through data."*  
> — Panduan Tugas Machine Learning
