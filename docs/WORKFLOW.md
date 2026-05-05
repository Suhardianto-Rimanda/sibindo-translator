# Alur Tahapan — SiBindo Translator

Panduan lengkap dari setup lingkungan hingga deployment produksi.

---

## Gambaran Umum

```
Phase 0: Setup Lingkungan
    ↓
Phase 1: Siapkan Dataset YOLO (hand/person ROI)
    ↓
Phase 2: Training YOLOv8 (lokal / Colab)
    ↓
Phase 3: Kumpulkan Dataset Landmark BISINDO
    ↓
Phase 4: Inspeksi & Augmentasi Dataset
    ↓
Phase 5: Training LSTM (lokal / Colab)
    ↓
Phase 6: Evaluasi Model
    ↓
Phase 7: Jalankan Aplikasi Web
    ↓
Phase 8: Deploy Produksi (Docker)
```

---

## Phase 0 — Setup Lingkungan

**Prasyarat:** Python 3.10+, Git, Webcam (untuk koleksi data).

```bash
# 1. Clone / masuk ke folder project
cd tugas_akhir_projects

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 4. Install dependensi
pip install -r requirements.txt

# 5. Salin konfigurasi
copy .env.example .env
```

### Konfigurasi `.env`

Buka `.env` dan sesuaikan nilai berikut:

| Key | Default | Keterangan |
|-----|---------|------------|
| `MEDIAPIPE_MODEL_PATH` | `models/mediapipe/hand_landmarker.task` | Model HandLandmarker — **auto-download** ~8MB jika belum ada |
| `YOLO_WEIGHTS` | `models/yolov8/best.pt` | Path weight YOLOv8 hasil training |
| `YOLO_CONF_THRESHOLD` | `0.5` | Threshold deteksi ROI |
| `LSTM_MODEL_PATH` | `models/lstm/bisindo_lstm.h5` | Path model LSTM (kata/kalimat) |
| `LSTM_LABELS_PATH` | `models/lstm/labels.json` | Path file label LSTM |
| `LSTM_SEQUENCE_LENGTH` | `30` | **Harus sama** dengan `--frames` saat collect & train |
| `LSTM_CONF_THRESHOLD` | `0.7` | Filter confidence prediksi kata |
| `LETTER_MODEL_PATH` | `models/letter_mlp/bisindo_letter.h5` | Path model MLP (huruf/finger spelling) |
| `LETTER_LABELS_PATH` | `models/letter_mlp/labels.json` | Path file label huruf |
| `LETTER_CONF_THRESHOLD` | `0.7` | Filter confidence prediksi huruf |
| `TTS_LANG` | `id` | Bahasa gTTS |

> **Catatan:** Tanpa weight terlatih, aplikasi tetap bisa berjalan dalam **stub mode** — berguna untuk uji UI sebelum model siap.
>
> **MediaPipe:** Model `hand_landmarker.task` didownload otomatis (~8MB) ke `models/mediapipe/` saat pertama kali aplikasi atau script dijalankan. Butuh koneksi internet sekali saja.

### Verifikasi Setup

```bash
# Pastikan semua package terinstall
python -c "import flask, cv2, mediapipe, ultralytics, tensorflow; print('OK')"

# Jalankan test suite (harus lolos semua)
# Saat pertama kali: MediaPipe akan download hand_landmarker.task (~8MB)
pytest -v tests/
```

> **Python 3.12 wajib.** MediaPipe 0.10.x hanya support sampai Python 3.12. Gunakan `py -3.12 -m venv .venv`.

---

## Phase 1 — Siapkan Dataset YOLOv8

YOLOv8 bertugas mendeteksi **ROI (Region of Interest)** — area tangan/tubuh — sebelum dikirim ke MediaPipe.

### Opsi A: Download dari Roboflow / Kaggle

1. Download dataset deteksi tangan dalam **format YOLO** (ada `images/` dan `labels/`).
2. Letakkan di `data/raw/yolo_input/`:
   ```
   data/raw/yolo_input/
   ├── images/
   │   ├── img001.jpg
   │   └── ...
   └── labels/
       ├── img001.txt
       └── ...
   ```
3. Jalankan helper untuk split train/val/test + buat `data.yaml`:
   ```bash
   python scripts/prepare_yolo_dataset.py \
       --src data/raw/yolo_input \
       --classes hand person
   ```
   Output: `data/yolo/` dengan struktur siap pakai Ultralytics.

### Opsi B: Gunakan Dataset Publik Langsung

```bash
# Contoh: Roboflow CLI
pip install roboflow
# Kemudian ikuti instruksi di notebook 01_train_yolov8_colab.ipynb
```

---

## Phase 2 — Training YOLOv8

### Lokal (GPU, misal RTX 2050)

```bash
python scripts/train_yolov8.py \
    --data data/yolo/data.yaml \
    --epochs 100 \
    --imgsz 640
```

Setelah selesai, salin weight terbaik:

```bash
copy runs\detect\bisindo_roi\weights\best.pt models\yolov8\best.pt
```

### Google Colab (Recommended untuk GPU gratis)

1. Buka `notebooks/01_train_yolov8_colab.ipynb` di Google Colab.
2. Mount Google Drive.
3. Upload `data/yolo/` ke Drive.
4. Isi variabel konfigurasi di cell pertama (path, epochs, dll).
5. Jalankan semua cell secara berurutan.
6. Weight hasil training tersimpan otomatis di Drive → unduh ke `models/yolov8/best.pt`.

### Evaluasi YOLOv8

```bash
python scripts/evaluate_yolo.py \
    --weights models/yolov8/best.pt \
    --data data/yolo/data.yaml
```

Metrik yang dihasilkan: **mAP50**, **mAP50-95**, precision, recall per kelas.

---

## Phase 3 — Siapkan Dataset Landmark BISINDO

Sistem mendukung dua sumber dataset: **video file** (untuk kata/kalimat) dan **foto** (untuk huruf finger spelling), selain rekam langsung dari webcam.

---

### Phase 3A — Dataset Kata: Rekam dari Webcam

Setiap kata/frasa BISINDO direkam sebagai **30 frame** landmark dari webcam langsung.

Tentukan daftar kata yang akan dilatih, misalnya:

```
halo, terima_kasih, maaf, tolong, iya, tidak, selamat_pagi,
nama_saya, apa_kabar, sampai_jumpa
```

```bash
# Format: --label <nama_kata> --samples <jumlah_sampel> --frames <frame_per_sampel>
python scripts/collect_landmarks.py --label halo --samples 30 --frames 30
python scripts/collect_landmarks.py --label terima_kasih --samples 30 --frames 30
# ... ulangi untuk setiap kata
```

**Kontrol saat rekam:**
- `SPACE` — mulai merekam 1 sampel (30 frame otomatis)
- `ESC` — keluar (progress tersimpan, bisa dilanjutkan kapan saja)

> **Tips:** Pencahayaan cukup, variasikan jarak kamera. Minimal 30 sampel per kata.

Hasil disimpan di: `data/processed/<label>/<idx>.npy` — shape `(30, 126)`

---

### Phase 3B — Dataset Kata: Ekstrak dari File Video

Jika sudah punya dataset video (`.mp4`, `.avi`, `.mov`), susun di:

```
data/raw/videos/
├── saya/
│   ├── 001.mp4
│   └── 002.mp4
├── suka/
└── makan/
```

Lalu ekstrak landmark:

```bash
python scripts/extract_landmarks_from_video.py
# atau custom path:
python scripts/extract_landmarks_from_video.py --input data/raw/videos --out data/processed/words --frames 30
```

Output: `data/processed/words/<label>/<idx>.npy` — shape `(30, 126)`

Script otomatis **sample/pad** video ke 30 frame (linspace sampling jika video lebih panjang, pad frame terakhir jika lebih pendek).

```bash
make extract-videos   # shortcut
```

---

### Phase 3C — Dataset Huruf: Ekstrak dari Foto

Untuk huruf/finger spelling (A–Z), siapkan foto di:

```
data/raw/photos/
├── A/
│   ├── 001.jpg
│   └── 002.jpg
├── S/
├── Y/
└── ...
```

Lalu ekstrak landmark:

```bash
python scripts/extract_landmarks_from_photo.py
# atau custom path:
python scripts/extract_landmarks_from_photo.py --input data/raw/photos --out data/processed/letters
```

Output: `data/processed/letters/<label>/<idx>.npy` — shape `(126,)` (single-frame vector)

Script menggunakan `static_image_mode=True` untuk akurasi lebih baik pada foto statis.

```bash
make extract-photos   # shortcut
```

---

## Phase 4 — Inspeksi & Augmentasi Dataset

### Inspeksi Distribusi

```bash
python scripts/inspect_dataset.py
```

Output: tabel distribusi kelas, rasio frame kosong per kelas, flag kelas yang kekurangan sampel.

**Idealnya:**
- Semua kelas ≥ 30 sampel.
- Missing frame ratio < 20% per kelas.

### Augmentasi Offline (Opsional)

Bila kelas tertentu kekurangan data, augmentasi ke disk:

```bash
python scripts/offline_augment.py --factor 5
```

Ini menghasilkan 5× sampel dari data asli dengan variasi rotasi, scale, jitter, dan temporal shift.

> **Catatan:** Augmentasi *on-the-fly* sudah aktif otomatis saat training (`--augment-factor`). Augmentasi offline hanya perlu jika ingin menyimpan dataset yang sudah di-augment secara permanen.

---

## Phase 5 — Training Model

### Phase 5A — Training LSTM (Kata/Kalimat)

Input: `data/processed/words/<label>/*.npy` atau `data/processed/<label>/*.npy` (dari webcam)

```bash
# Dari dataset video
python scripts/train_lstm.py --data data/processed/words --epochs 100 --batch 16 --augment-factor 4

# Dari dataset webcam (path default)
python scripts/train_lstm.py --epochs 100 --batch 16 --augment-factor 4

# Shortcut Makefile (pakai data/processed/words)
make train
```

Parameter penting:

| Flag | Default | Keterangan |
|------|---------|------------|
| `--data` | `data/processed` | Root folder dataset kata |
| `--epochs` | 100 | Jumlah epoch training |
| `--batch` | 16 | Batch size |
| `--augment-factor` | 4 | Multiplier augmentasi per sampel |

Output:
```
models/lstm/
├── bisindo_lstm.h5
├── labels.json
└── training_history.json
```

**Google Colab:** Buka `notebooks/02_train_lstm_colab.ipynb`. Upload `data/processed/` ke Drive → jalankan semua cell.

---

### Phase 5B — Training MLP Letter (Huruf/Finger Spelling)

Input: `data/processed/letters/<label>/*.npy` (dari `extract_landmarks_from_photo.py`)

```bash
python scripts/train_letter_classifier.py --epochs 100 --batch 32

# Shortcut
make train-letter
```

Parameter penting:

| Flag | Default | Keterangan |
|------|---------|------------|
| `--data` | `data/processed/letters` | Root folder dataset huruf |
| `--epochs` | 100 | Jumlah epoch |
| `--batch` | 32 | Batch size (foto lebih banyak → batch lebih besar ok) |

Output:
```
models/letter_mlp/
├── bisindo_letter.h5
├── labels.json
└── training_history.json
```

Arsitektur MLP: `Dense(256) → BN → Dense(128) → BN → Dense(64) → Dense(N, softmax)` — tidak perlu LSTM karena huruf adalah gesture statis (single frame).

---

### Monitoring Training

Indikator konvergensi yang baik untuk keduanya:
- Val accuracy > 85% setelah 50 epoch.
- Val loss tidak naik (tidak overfit).
- Jika overfit: kurangi `--augment-factor` (LSTM) atau tambah data foto (MLP).

---

## Phase 6 — Evaluasi Model

### Evaluasi LSTM

```bash
python scripts/evaluate_lstm.py
```

Hasil di `models/lstm/eval/`:

| File | Isi |
|------|-----|
| `confusion_matrix.png` | Visualisasi confusion matrix |
| `classification_report.txt` | Precision, Recall, F1 per kelas |
| `accuracy_loss.png` | Grafik akurasi & loss per epoch |
| `metrics.json` | Ringkasan numerik |

**Target minimum** (sesuai proposal):
- Overall accuracy ≥ 80%
- F1-score per kelas ≥ 0.75

### Evaluasi YOLOv8

```bash
python scripts/evaluate_yolo.py \
    --weights models/yolov8/best.pt \
    --data data/yolo/data.yaml
```

**Target minimum:** mAP50 ≥ 0.85

### Benchmark Latency End-to-End

```bash
python scripts/benchmark_latency.py --frames 200
```

Output: percentile p50/p90/p95/p99 untuk setiap stage:

| Stage | Target |
|-------|--------|
| YOLO inference | < 50ms (p95) |
| MediaPipe extraction | < 20ms (p95) |
| LSTM inference | < 30ms (p95) |
| End-to-end | < 200ms (p95) |

---

## Phase 7 — Jalankan Aplikasi Web

Pastikan weight model sudah ada di path yang dikonfigurasi di `.env`.

```bash
python app.py
```

Buka browser: [http://localhost:5000](http://localhost:5000)

### Alur Penggunaan Aplikasi

```
1. Klik "Mulai Kamera"
2. Izinkan akses webcam di browser
3. Lakukan isyarat BISINDO di depan kamera
4. Sistem mendeteksi tangan → ekstrak landmark → prediksi kata
5. Kata yang terdeteksi muncul di panel kanan
6. Klik "Baca" untuk mendengar kalimat via TTS
7. Klik "Reset" untuk memulai kalimat baru
```

### Troubleshooting Umum

| Masalah | Kemungkinan Penyebab | Solusi |
|---------|---------------------|--------|
| Kamera tidak muncul | Permission browser ditolak | Izinkan akses kamera di browser settings |
| Tidak ada prediksi | Weight model belum ada | Jalankan training atau taruh weight di path `.env` |
| Latency tinggi | CPU/GPU terbatas | Kurangi resolusi frame di `app.js` (default 480px) |
| Prediksi acak | Model belum konvergen | Tambah epoch atau sampel dataset |
| Audio tidak muncul | gTTS butuh internet | Cek koneksi internet |

---

## Phase 8 — Deploy Produksi (Docker)

### Build & Jalankan

```bash
# Build image
docker compose build

# Jalankan di background
docker compose up -d

# Cek status
docker compose ps

# Lihat logs
docker compose logs -f
```

Akses: [http://localhost:5000](http://localhost:5000)

### Catatan Produksi

- **HTTPS wajib** untuk `getUserMedia` di browser selain localhost.  
  Gunakan Caddy atau Nginx + Let's Encrypt sebagai reverse proxy.
- **Volume mount** weight model agar tidak perlu rebuild image saat update model:
  ```yaml
  # di docker-compose.yml
  volumes:
    - ./models:/app/models
  ```
- **Variabel lingkungan** bisa di-override via `docker-compose.yml` tanpa edit `.env`.

---

## Ringkasan Perintah Cepat (Makefile)

```bash
make install          # pip install -r requirements.txt
make run              # python app.py
make test             # pytest -v tests/
make collect LBL=halo # rekam dataset kata dari webcam
make inspect          # cek distribusi dataset
make extract-videos   # ekstrak landmark dari data/raw/videos/ → data/processed/words/
make extract-photos   # ekstrak landmark dari data/raw/photos/ → data/processed/letters/
make train            # training LSTM kata (data/processed/words/)
make train-letter     # training MLP huruf (data/processed/letters/)
make eval             # evaluasi LSTM
make bench            # benchmark latency
make docker           # docker compose up -d
```

---

## Checklist Sebelum Sidang

- [ ] YOLOv8 mAP50 ≥ 0.85 pada test set
- [ ] LSTM accuracy ≥ 80% pada validation set (mode kata)
- [ ] MLP letter accuracy ≥ 80% pada validation set (mode huruf)
- [ ] Semua kelas F1-score ≥ 0.75
- [ ] Latency end-to-end p95 < 200ms
- [ ] Semua unit test lulus (`pytest -v tests/`)
- [ ] Confusion matrix tersimpan di `models/lstm/eval/`
- [ ] Aplikasi berjalan tanpa error di browser (mode kata & mode huruf)
- [ ] TTS menghasilkan audio untuk kalimat yang terdeteksi
- [ ] Dataset terdokumentasi (jumlah sampel per kelas, total frame/foto)
