# Struktur Folder — SiBindo Translator

Dokumen ini menjelaskan kegunaan setiap folder dan file dalam project untuk mempermudah navigasi dan pemahaman arsitektur kode.

---

## Daftar Isi

- [Root Level](#root-level)
- [`app/` — Source Aplikasi Web](#app--source-aplikasi-web)
  - [`app/services/` — Modul AI/ML](#appservices--modul-aiml)
  - [`app/routes/` — HTTP Endpoints](#approutes--http-endpoints)
  - [`app/utils/` — Helper Sistem](#apputils--helper-sistem)
  - [`app/templates/` — Jinja2 HTML](#apptemplates--jinja2-html)
  - [`app/static/` — Asset Statis](#appstatic--asset-statis)
- [`scripts/` — Data + Training Pipeline](#scripts--data--training-pipeline)
- [`notebooks/` — Colab-Ready Training](#notebooks--colab-ready-training)
- [`tests/` — Unit + Smoke Tests](#tests--unit--smoke-tests)
- [`models/` — Trained Weights](#models--trained-weights)
- [`data/` — Dataset](#data--dataset)
- [Alur Data Antar Folder](#alur-data-antar-folder)
- [Mapping ke Proposal](#mapping-ke-proposal)
- [Aturan Maintain](#aturan-maintain)

---

## Root Level

```
tugas_akhir_projects/
├── app.py              # Entry point — jalanin Flask via `python app.py`
├── config.py           # Load .env, expose Config class ke seluruh app
├── requirements.txt    # Pin version pip dependencies
├── pytest.ini          # Konfig test runner
├── Dockerfile          # Image production (gunicorn + Python 3.10-slim)
├── docker-compose.yml  # Stack deploy: web service + volume mount + healthcheck
├── Makefile            # Shortcut commands (make run, make train, dst.)
├── .env.example        # Template env — copy ke .env lalu edit
├── .gitignore          # Exclude __pycache__, model weights, audio output
├── .dockerignore       # Exclude file dari image (notebooks, data, tests)
└── README.md           # Dokumentasi end-to-end
```

| File | Kegunaan |
|------|----------|
| `app.py` | Boot Flask app via factory pattern. Jalanin dengan `python app.py`. |
| `config.py` | Centralized config dari environment variable. Hindari hardcoded value di service. |
| `requirements.txt` | Pin versi dependency. Reproducible install. |
| `Dockerfile` | Production image. OS deps (libgl1, libglib) + gunicorn server. |
| `docker-compose.yml` | Orkestrasi deploy: port mapping, volume mount untuk model + audio, healthcheck. |
| `Makefile` | Shortcut command. Hindari ngetik panjang. Contoh: `make run`, `make train`, `make eval`. |

---

## `app/` — Source Aplikasi Web

```
app/
├── __init__.py    # Application factory `create_app()`
└── pipeline.py    # Orkestrator two-stage detection
```

| File | Kegunaan |
|------|----------|
| `__init__.py` | Factory pattern Flask. Register blueprint, error handler global, spawn cleanup thread. |
| `pipeline.py` | **Otak sistem.** Pegang state per session (buffer landmark + NLP processor). Method utama `process_frame()` jalanin pipeline lengkap: YOLO → MediaPipe → buffer → LSTM → NLP → gTTS. |

---

### `app/services/` — Modul AI/ML

Setiap file = satu tanggung jawab (single responsibility principle).

```
services/
├── yolo_detector.py
├── mediapipe_extractor.py
├── landmark_normalizer.py
├── landmark_augmenter.py
├── lstm_classifier.py
├── letter_classifier.py
├── nlp_processor.py
└── tts_service.py
```

| File | Stage | Kegunaan |
|------|-------|----------|
| `yolo_detector.py` | 1 | Detect ROI bbox (hand/person) via YOLOv8. Fallback whole-frame jika weight ga ada. |
| `mediapipe_extractor.py` | 2 | Ekstraksi 21 landmark × 3 dim × 2 hands = 126 fitur via **MediaPipe Tasks API** (`HandLandmarker`). Model `hand_landmarker.task` auto-download ~8MB ke `models/mediapipe/` saat pertama kali jalan. `static_image_mode=False` → `RunningMode.VIDEO`; `True` → `RunningMode.IMAGE`. |
| `landmark_normalizer.py` | 2.5 | Translasi + skala invariance. Pusatkan ke wrist, normalize ke max distance. **Wajib** biar model tidak tergantung posisi/jarak kamera. |
| `landmark_augmenter.py` | training | Augmentasi: rotate, scale, jitter, temporal shift. Cuma dipakai saat training, bukan inference. |
| `lstm_classifier.py` | 3 (kata) | Wrap model Keras `.h5` LSTM. Predict urutan 30-frame landmark → label kata + confidence. |
| `letter_classifier.py` | 3 (huruf) | Wrap model Keras `.h5` MLP. Predict single-frame 126-dim vector → label huruf + confidence. Fallback stub jika weight belum ada. |
| `nlp_processor.py` | 4 | Rule-based: smoothing window (3 prediksi konsisten), dedup kata berulang, sentence assembly + capitalize. Dipakai oleh kedua mode. |
| `tts_service.py` | 5 | gTTS sintesis text → mp3. Simpan ke `static/audio/`, return URL ke frontend. |

---

### `app/routes/` — HTTP Endpoints

```
routes/
├── main.py    # Halaman web (HTML)
└── api.py     # JSON API
```

| File | Endpoint | Method | Kegunaan |
|------|----------|--------|----------|
| `main.py` | `/` | GET | Halaman penerjemah (webcam UI) |
| `main.py` | `/panduan` | GET | Halaman panduan penggunaan |
| `main.py` | `/tentang` | GET | Halaman info penulis + arsitektur |
| `api.py` | `/api/predict` | POST | Terima `frame` (base64) + `session_id` + `mode` (`"word"`/`"letter"`), return prediksi + bbox + landmarks |
| `api.py` | `/api/reset` | POST | Reset buffer + sentence per session |
| `api.py` | `/api/health` | GET | Liveness check + jumlah session aktif |

---

### `app/utils/` — Helper Sistem

```
utils/
├── logger.py
└── audio_cleanup.py
```

| File | Kegunaan |
|------|----------|
| `logger.py` | Logger terpusat dengan format timestamp + level + name. Pakai di semua service. |
| `audio_cleanup.py` | Daemon thread — hapus mp3 lebih lama dari 10 menit setiap 5 menit. Cegah disk penuh saat aplikasi jalan lama. |

---

### `app/templates/` — Jinja2 HTML

```
templates/
├── base.html     # Layout dasar (header, navbar, footer)
├── index.html    # Halaman penerjemah
├── panduan.html  # Halaman panduan
└── tentang.html  # Halaman info
```

| File | Kegunaan |
|------|----------|
| `base.html` | Layout induk. Halaman lain `extends` ini biar konsisten. |
| `index.html` | Webcam feed + status panel (kata + confidence + buffer) + history kata + audio player. |
| `panduan.html` | Tutorial setup lingkungan + cara pakai + tips akurasi + indikator visual. |
| `tentang.html` | Info penulis (Suhardianto Rimanda, NIM 6103230046) + arsitektur teknis + tech stack. |

---

### `app/static/` — Asset Statis

```
static/
├── css/style.css   # Styling
├── js/app.js       # Frontend logic
└── audio/          # Output mp3 (auto-cleanup)
```

| File/Folder | Kegunaan |
|-------------|----------|
| `css/style.css` | Styling lengkap: nav, card, video wrapper, history chip, prose page. CSS variable untuk theme. |
| `js/app.js` | Webcam capture (`getUserMedia`) → kirim frame ke `/api/predict` → render bbox + landmark + history + auto-play audio. Session ID di-generate di `localStorage`. |
| `audio/` | Output mp3 hasil gTTS. Otomatis di-cleanup oleh `audio_cleanup.py`. |

---

## `scripts/` — Data + Training Pipeline

```
scripts/
├── collect_landmarks.py
├── extract_landmarks_from_video.py
├── extract_landmarks_from_photo.py
├── inspect_dataset.py
├── offline_augment.py
├── prepare_yolo_dataset.py
├── train_yolov8.py
├── train_lstm.py
├── train_letter_classifier.py
├── evaluate_yolo.py
├── evaluate_lstm.py
└── benchmark_latency.py
```

| Script | Kategori | Kegunaan |
|--------|----------|----------|
| `collect_landmarks.py` | Data collection | Rekam dataset kata dari webcam. SPACE per sample. Output: `data/processed/<label>/*.npy` shape `(30, 126)`. |
| `extract_landmarks_from_video.py` | Data collection | Ekstrak landmark dari file video (`.mp4/.avi/.mov`). Sample/pad ke 30 frame. Output: `data/processed/words/<label>/*.npy` shape `(30, 126)`. |
| `extract_landmarks_from_photo.py` | Data collection | Ekstrak landmark dari foto (`.jpg/.png`). `static_image_mode=True`. Output: `data/processed/letters/<label>/*.npy` shape `(126,)`. |
| `inspect_dataset.py` | Data analysis | Cek distribusi kelas + missing frame ratio + warning imbalance > 2x. Pakai sebelum training. |
| `offline_augment.py` | Data augmentation | Persistent augmentation — write augmented `.npy` ke disk. Alternatif on-the-fly saat training. |
| `prepare_yolo_dataset.py` | Data preparation | Split flat folder (`images/labels`) → `train/val/test` + generate `data.yaml`. Untuk format YOLO. |
| `train_yolov8.py` | Training | Train YOLOv8 (Stage 1 ROI detector). Output: `runs/detect/.../weights/best.pt`. |
| `train_lstm.py` | Training | Train LSTM untuk kata/kalimat. Input: `(30, 126)` sequences. Pakai normalization + augmentation + class weights. |
| `train_letter_classifier.py` | Training | Train MLP untuk huruf/finger spelling. Input: `(126,)` single-frame vectors. Dense 256→128→64→N. |
| `evaluate_yolo.py` | Evaluation | mAP@0.5, mAP@0.5:0.95, precision, recall via `model.val()`. |
| `evaluate_lstm.py` | Evaluation | Confusion matrix.png + classification_report.txt + accuracy_loss.png + metrics.json. **Wajib bab 1.3.d proposal.** |
| `benchmark_latency.py` | Performance | End-to-end latency benchmark. Percentile p50/p90/p95/p99 per stage + fps. **Wajib bab 1.3.d proposal.** |

---

## `notebooks/` — Colab-Ready Training

```
notebooks/
├── 01_train_yolov8_colab.ipynb
└── 02_train_lstm_colab.ipynb
```

| Notebook | Kegunaan |
|----------|----------|
| `01_train_yolov8_colab.ipynb` | Train YOLOv8 di GPU Colab. Mount Drive, download Roboflow, train, val, export `best.pt`. |
| `02_train_lstm_colab.ipynb` | Train LSTM di GPU Colab. Load `.npy` dari Drive, normalize + augment, train, plot, eval, export `.h5`. |

> **Kapan pakai notebook?** Saat GPU lokal kurang kuat. Colab Pro Free Tier (T4/A100) jauh lebih cepat dari RTX 2050 lokal.

---

## `tests/` — Unit + Smoke Tests

```
tests/
├── test_nlp_processor.py
├── test_landmark_normalizer.py
├── test_landmark_augmenter.py
└── test_api.py
```

| Test File | Cakupan |
|-----------|---------|
| `test_nlp_processor.py` | Smoothing window, dedup, stopword, capitalize, max_words window, reset state. |
| `test_landmark_normalizer.py` | Translation invariance, scale invariance, zero-hand handling, sequence shape preservation. |
| `test_landmark_augmenter.py` | Rotate/scale/jitter/temporal shift preserve shape, jitter only perturbs non-zero entries. |
| `test_api.py` | Flask client smoke test: `/health`, predict missing field, predict blank frame, reset. |

> Run dengan `pytest -v tests/` atau `make test`.

---

## `models/` — Trained Weights

```
models/
├── mediapipe/
│   └── hand_landmarker.task     # auto-download saat pertama run
├── yolov8/
│   ├── best.pt
│   └── eval/metrics.json
├── lstm/
│   ├── bisindo_lstm.h5
│   ├── labels.json
│   ├── training_history.json
│   └── eval/
│       ├── confusion_matrix.png
│       ├── classification_report.txt
│       ├── accuracy_loss.png
│       └── metrics.json
└── letter_mlp/
    ├── bisindo_letter.h5
    ├── labels.json
    └── training_history.json
```

| File | Kegunaan |
|------|----------|
| `mediapipe/hand_landmarker.task` | Model binary MediaPipe HandLandmarker (~8MB). **Auto-download** dari Google CDN jika belum ada. Bisa juga download manual. |
| `yolov8/best.pt` | Hasil training YOLO. Salin manual dari `runs/detect/.../weights/best.pt`. |
| `lstm/bisindo_lstm.h5` | Hasil training LSTM untuk kata/kalimat (Keras format). |
| `lstm/labels.json` | Mapping index → nama kata. **WAJIB sinkron** dengan urutan training. |
| `lstm/training_history.json` | Riwayat loss + accuracy per epoch LSTM. |
| `letter_mlp/bisindo_letter.h5` | Hasil training MLP untuk huruf/finger spelling. |
| `letter_mlp/labels.json` | Mapping index → huruf. |
| `*/eval/` | Output script `evaluate_*.py`. |

> **Catatan:** Folder ini di-gitignore karena weight besar. Simpan di Google Drive / cloud storage untuk backup.

---

## `data/` — Dataset

```
data/
├── raw/
│   ├── yolo_input/
│   │   ├── images/
│   │   └── labels/
│   ├── videos/             # input video per label untuk kata
│   │   ├── saya/
│   │   └── makan/
│   └── photos/             # input foto per label untuk huruf
│       ├── A/
│       └── S/
├── processed/
│   ├── <label>/            # dari collect_landmarks.py (webcam)
│   │   ├── 000.npy         # shape (30, 126)
│   │   └── ...
│   ├── words/              # dari extract_landmarks_from_video.py
│   │   ├── saya/
│   │   └── makan/
│   └── letters/            # dari extract_landmarks_from_photo.py
│       ├── A/              # shape (126,)
│       └── S/
├── augmented/
└── yolo/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── data.yaml
```

| Folder | Isi | Asal |
|--------|-----|------|
| `raw/` | Dataset mentah sebelum diproses. | Manual |
| `raw/yolo_input/` | Input untuk `prepare_yolo_dataset.py`. | Manual |
| `raw/videos/<label>/` | Video gerakan per kata. | Manual |
| `raw/photos/<label>/` | Foto per huruf (A–Z, finger spelling). | Manual |
| `processed/<label>/*.npy` | Sequence dari webcam, shape `(30, 126)`. | `collect_landmarks.py` |
| `processed/words/<label>/*.npy` | Sequence dari video, shape `(30, 126)`. | `extract_landmarks_from_video.py` |
| `processed/letters/<label>/*.npy` | Single-frame dari foto, shape `(126,)`. | `extract_landmarks_from_photo.py` |
| `augmented/` | Hasil augmentasi offline. | `offline_augment.py` |
| `yolo/` | Dataset YOLO siap training. | `prepare_yolo_dataset.py` |

> **Catatan:** Folder ini di-gitignore karena ukuran besar.

---

## Alur Data Antar Folder

```
┌─────────────────────────────────────────────────────────────────┐
│                       PIPELINE TRAINING                         │
└─────────────────────────────────────────────────────────────────┘

[Roboflow/Kaggle]
       │
       ▼
   data/raw/  ──[prepare_yolo_dataset.py]──►  data/yolo/
                                                  │
                                                  ▼
                                          [train_yolov8.py]
                                                  │
                                                  ▼
                                          models/yolov8/best.pt


[Webcam]                         [File Video]                  [Foto]
   │                                   │                          │
   ▼                                   ▼                          ▼
[collect_landmarks.py]    [extract_landmarks_from_video.py]  [extract_landmarks_from_photo.py]
   │                                   │                          │
   ▼                                   ▼                          ▼
data/processed/<label>/    data/processed/words/<label>/   data/processed/letters/<label>/
(30, 126)                          (30, 126)                    (126,)
   │                                   │                          │
   └──────────────┬────────────────────┘                          │
                  ▼                                               ▼
          [train_lstm.py]                            [train_letter_classifier.py]
          (normalize + augment)                      (normalize + class weights)
                  │                                               │
                  ▼                                               ▼
   models/lstm/bisindo_lstm.h5             models/letter_mlp/bisindo_letter.h5


┌─────────────────────────────────────────────────────────────────┐
│                       PIPELINE INFERENCE                        │
└─────────────────────────────────────────────────────────────────┘

[Webcam Browser]
       │ (HTTP POST: frame + session_id + mode)
       ▼
   app/routes/api.py  ──► validasi mode ("word" | "letter")
       │
       ▼
   app/pipeline.py
       │
       ├── mode="word" ──► buffer 30 frame ──► lstm_classifier  ──► models/lstm/*.h5
       │
       └── mode="letter" ──► single frame ──► letter_classifier ──► models/letter_mlp/*.h5
       │
       ├──► yolo_detector.py  (ROI — shared)
       ├──► mediapipe_extractor.py  (landmark — shared)
       ├──► landmark_normalizer.py  (normalisasi — shared)
       ├──► nlp_processor.py  (smoothing + assembly — shared)
       └──► tts_service.py    ──► app/static/audio/*.mp3
                                       │
                                       ▼
                              [Browser Audio Player]
```

---

## Mapping ke Proposal

| Bab Proposal | Folder/File Terkait |
|--------------|---------------------|
| 3.2.1 Studi Literatur | `README.md` arsitektur + `docs/STRUCTURE.md` |
| 3.2.3 Perancangan Sistem | `app/pipeline.py` |
| 3.2.4 Implementasi | `app/services/`, `scripts/train_*.py` |
| 3.2.5 Pengujian Sistem | `scripts/evaluate_*.py`, `scripts/benchmark_latency.py`, `tests/` |
| 3.3.3 Flowchart Sistem | `app/pipeline.py` `process_frame()` |
| 3.3.4 Activity Diagram | `app/routes/` + `app/static/js/app.js` |
| 3.3.5 Use Case Diagram | `app/routes/main.py` |
| 3.3.6 Arsitektur Sistem | `app/pipeline.py` + diagram di README |
| 3.3.7 User Interface | `app/templates/`, `app/static/` |
| 1.3.d Evaluasi (Confusion Matrix, mAP, latency) | `scripts/evaluate_*.py`, `scripts/benchmark_latency.py` |
| 4.1 Jadwal Pelaksanaan | `README.md` workflow |
| 4.2 RAB (VPS + Domain) | `Dockerfile`, `docker-compose.yml` |

---

## Aturan Maintain

| Skenario | Action |
|----------|--------|
| MediaPipe error `no attribute 'solutions'` | Sudah tidak bisa — gunakan versi terbaru kode; model `hand_landmarker.task` auto-download |
| Tambah kata baru dari webcam | `collect_landmarks.py` → `train_lstm.py` → restart app |
| Tambah kata dari video | taruh di `data/raw/videos/<label>/` → `make extract-videos` → `make train` |
| Tambah huruf dari foto | taruh di `data/raw/photos/<label>/` → `make extract-photos` → `make train-letter` |
| Ubah threshold confidence kata | Edit `.env` `LSTM_CONF_THRESHOLD` |
| Ubah threshold confidence huruf | Edit `.env` `LETTER_CONF_THRESHOLD` |
| Ubah arsitektur LSTM | Edit `scripts/train_lstm.py` `build_model()` |
| Ubah arsitektur MLP letter | Edit `scripts/train_letter_classifier.py` `build_model()` |
| Ubah aturan NLP (smoothing, dedup, stopword) | Edit `app/services/nlp_processor.py` |
| Tambah halaman web | `app/templates/` + register di `app/routes/main.py` |
| Tambah API endpoint | `app/routes/api.py` |
| Ubah panjang sequence buffer | Edit `.env` (`LSTM_SEQUENCE_LENGTH`) — sinkron dengan `--frames` saat collect & train |
| Deploy production | `docker compose up -d` (atau `make docker-up`) |
| Run test | `pytest -v tests/` (atau `make test`) |
| Cek dataset balance | `python scripts/inspect_dataset.py` |
| Benchmark performa | `python scripts/benchmark_latency.py` (atau `make bench`) |

---

## Quick Reference Command

```bash
# Setup
pip install -r requirements.txt
copy .env.example .env

# Dataset kata — dari webcam
python scripts/collect_landmarks.py --label halo --samples 30 --frames 30

# Dataset kata — dari video file
python scripts/extract_landmarks_from_video.py   # data/raw/videos/ → data/processed/words/

# Dataset huruf — dari foto
python scripts/extract_landmarks_from_photo.py   # data/raw/photos/ → data/processed/letters/

# Inspect distribusi
python scripts/inspect_dataset.py

# Train
python scripts/train_yolov8.py --data data/yolo/data.yaml --epochs 100
python scripts/train_lstm.py --data data/processed/words --epochs 100 --augment-factor 4
python scripts/train_letter_classifier.py --epochs 100

# Evaluate
python scripts/evaluate_yolo.py --weights models/yolov8/best.pt --data data/yolo/data.yaml
python scripts/evaluate_lstm.py
python scripts/benchmark_latency.py --frames 200

# Run
python app.py                      # dev server
docker compose up -d               # production

# Test
pytest -v tests/
```

---

> Dokumentasi ini **hidup**. Update setiap kali tambah folder/file baru biar tim tetap selaras.
