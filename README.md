# SiBindo Translator — Flask Starter Kit

Implementasi Computer Vision untuk klasifikasi kalimat BISINDO berbasis webcam dengan arsitektur **Two-Stage Detection System**: YOLOv8 → MediaPipe → LSTM → NLP rule-based → gTTS.

Tugas Akhir D-III Teknik Informatika — Politeknik Negeri Bengkalis (Suhardianto Rimanda, 6103230046).

## Arsitektur

```
Webcam ─► YOLOv8 (ROI)
            │
            ▼
       Crop ROI ─► MediaPipe (21 lm × 3 dim × 2 hands = 126 fitur)
                    │
                    ▼
            Landmark Normalizer (translation + scale invariant)
                    │
          ┌─────────┴───────────────┐
          │ mode="word"             │ mode="letter"
          ▼                         ▼
   Buffer 30 frame          Single frame langsung
   ─► LSTM (kata)           ─► MLP (huruf/A–Z)
          │                         │
          └─────────┬───────────────┘
                    ▼
              NLP rule-based
        (smoothing + dedup + sentence)
                    │
                    ▼
           gTTS ─► Audio + Teks
```

## Struktur Folder

```
.
├── app.py                       # Entry point Flask
├── config.py                    # Config dari .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pytest.ini
├── .env.example
├── app/
│   ├── __init__.py              # Application factory + cleanup thread
│   ├── pipeline.py              # Orkestrator two-stage + session state
│   ├── routes/
│   │   ├── main.py              # Penerjemah, Panduan, Tentang
│   │   └── api.py               # /api/predict, /api/reset, /api/health
│   ├── services/
│   │   ├── yolo_detector.py
│   │   ├── mediapipe_extractor.py
│   │   ├── landmark_normalizer.py   # translasi + scale invariance
│   │   ├── landmark_augmenter.py    # rotation/scale/jitter/temporal
│   │   ├── lstm_classifier.py       # LSTM untuk kata (30-frame sequence)
│   │   ├── letter_classifier.py     # MLP untuk huruf (single-frame)
│   │   ├── nlp_processor.py     # smoothing + dedup + assembly
│   │   └── tts_service.py       # gTTS
│   ├── utils/
│   │   ├── logger.py
│   │   └── audio_cleanup.py
│   ├── templates/               # base, index, panduan, tentang
│   └── static/                  # css, js, audio output
├── scripts/
│   ├── collect_landmarks.py             # Rekam dataset kata dari webcam
│   ├── extract_landmarks_from_video.py  # Ekstrak dari video → data/processed/words/
│   ├── extract_landmarks_from_photo.py  # Ekstrak dari foto → data/processed/letters/
│   ├── inspect_dataset.py               # Cek distribusi kelas
│   ├── offline_augment.py               # Persistent augmentation
│   ├── prepare_yolo_dataset.py          # Split train/val/test + data.yaml
│   ├── train_yolov8.py                  # Train YOLOv8
│   ├── train_lstm.py                    # Train LSTM kata (with augmentation)
│   ├── train_letter_classifier.py       # Train MLP huruf
│   ├── evaluate_yolo.py                 # mAP eval
│   ├── evaluate_lstm.py                 # Confusion matrix + report + plots
│   └── benchmark_latency.py             # End-to-end latency benchmark
├── notebooks/
│   ├── 01_train_yolov8_colab.ipynb
│   └── 02_train_lstm_colab.ipynb
├── tests/
│   ├── test_nlp_processor.py
│   ├── test_landmark_normalizer.py
│   ├── test_landmark_augmenter.py
│   └── test_api.py
├── models/
│   ├── yolov8/                  # taruh best.pt di sini
│   ├── lstm/                    # bisindo_lstm.h5 + labels.json + eval/
│   └── letter_mlp/              # bisindo_letter.h5 + labels.json
└── data/
    ├── raw/
    │   ├── videos/              # video per kata → extract_landmarks_from_video.py
    │   └── photos/              # foto per huruf → extract_landmarks_from_photo.py
    └── processed/
        ├── <label>/             # dari webcam (collect_landmarks.py)
        ├── words/               # dari video (shape 30×126)
        └── letters/             # dari foto (shape 126)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
copy .env.example .env
```

## Workflow Training

### A. YOLOv8 (ROI Detector)

**Lokal (GPU lokal, RTX 2050):**

1. Siapkan dataset hand/person dari Roboflow/Kaggle (format YOLO).
2. Susun struktur dengan helper:
   ```bash
   python scripts/prepare_yolo_dataset.py \
       --src data/raw/yolo_input \
       --classes hand person
   ```
3. Latih:
   ```bash
   python scripts/train_yolov8.py --data data/yolo/data.yaml --epochs 100
   ```
4. Salin best weights:
   ```bash
   copy runs\detect\bisindo_roi\weights\best.pt models\yolov8\best.pt
   ```
5. Evaluasi:
   ```bash
   python scripts/evaluate_yolo.py --weights models/yolov8/best.pt --data data/yolo/data.yaml
   ```

**Google Colab:**
- Buka `notebooks/01_train_yolov8_colab.ipynb` di Colab.
- Mount Drive, isi konfigurasi, jalankan semua sel.
- Output `bisindo_roi/weights/best.pt` otomatis disimpan ke Drive.

### B. LSTM untuk Kata (Sequence Classifier)

Dataset bisa dari **webcam** atau **file video**:

```bash
# Opsi 1 — rekam dari webcam
python scripts/collect_landmarks.py --label halo --samples 30 --frames 30
# (SPACE rekam, ESC keluar)

# Opsi 2 — ekstrak dari video file
# Susun: data/raw/videos/<label>/*.mp4  lalu:
python scripts/extract_landmarks_from_video.py
```

```bash
# Inspeksi distribusi
python scripts/inspect_dataset.py

# Train LSTM
python scripts/train_lstm.py --data data/processed/words --epochs 100 --batch 16 --augment-factor 4
```

Output: `models/lstm/bisindo_lstm.h5`, `models/lstm/labels.json`

```bash
# Evaluasi
python scripts/evaluate_lstm.py
```

**Google Colab:** Buka `notebooks/02_train_lstm_colab.ipynb`. Upload `data/processed/words/` ke Drive lalu jalankan.

### C. MLP untuk Huruf (Letter/Finger Spelling)

```bash
# Susun foto: data/raw/photos/<huruf>/*.jpg  lalu ekstrak:
python scripts/extract_landmarks_from_photo.py

# Train MLP
python scripts/train_letter_classifier.py --epochs 100 --batch 32
```

Output: `models/letter_mlp/bisindo_letter.h5`, `models/letter_mlp/labels.json`

### C. Benchmark Latency (end-to-end)

```bash
python scripts/benchmark_latency.py --frames 200
```
Output: percentile p50/p90/p95/p99 untuk setiap stage (YOLO / MediaPipe / LSTM) + throughput fps.

## Menjalankan Aplikasi

```bash
python app.py
```
Buka [http://localhost:5000](http://localhost:5000), klik **Mulai Kamera**.

### Atau Docker:

```bash
docker compose up -d
```

## Testing

```bash
pytest -v tests/
```

## Konfigurasi `.env`

```ini
MEDIAPIPE_MODEL_PATH=models/mediapipe/hand_landmarker.task

YOLO_WEIGHTS=models/yolov8/best.pt
YOLO_CONF_THRESHOLD=0.5

LSTM_MODEL_PATH=models/lstm/bisindo_lstm.h5
LSTM_LABELS_PATH=models/lstm/labels.json
LSTM_SEQUENCE_LENGTH=30          # HARUS sama dengan --frames saat collect & train
LSTM_CONF_THRESHOLD=0.7

LETTER_MODEL_PATH=models/letter_mlp/bisindo_letter.h5
LETTER_LABELS_PATH=models/letter_mlp/labels.json
LETTER_CONF_THRESHOLD=0.7

TTS_LANG=id
TTS_OUTPUT_DIR=app/static/audio
```

## Catatan Penting

- **Python 3.12 wajib.** MediaPipe 0.10.x hanya support Python 3.9–3.12. Buat venv dengan `py -3.12 -m venv .venv`.
- **MediaPipe Tasks API** — kode memakai `HandLandmarker` (bukan `mp.solutions` yang dihapus di 0.10.31+). Model `hand_landmarker.task` (~8MB) auto-download ke `models/mediapipe/` saat pertama run.
- **Tanpa weight terlatih**, sistem tetap jalan: `YoloDetector` fallback ke whole-frame, `LstmClassifier`/`LetterClassifier` keluarkan stub. Berguna untuk uji UI.
- **Dua mode inference** — kirim field `mode` di payload:
  - `"word"` (default) — buffer 30 frame → LSTM → label kata
  - `"letter"` — single frame → MLP → label huruf (finger spelling)
- **Frontend** kirim frame ~5 fps via `POST /api/predict` (base64 JPEG, lebar 480 px).
- **Session per-user** tracking via `session_id` di payload (auto-generated di `localStorage`).
- **Audio cleanup** thread otomatis hapus `.mp3` lebih lama dari 10 menit setiap 5 menit.
- **Landmark normalization** wajib aktif agar model invariant terhadap posisi + jarak kamera.

## Evaluasi (Bab 1.3.d Proposal)

| Metrik | Tool | Script |
|--------|------|--------|
| mAP YOLOv8 | Ultralytics `model.val()` | `evaluate_yolo.py` |
| Confusion Matrix LSTM | scikit-learn | `evaluate_lstm.py` |
| Precision / Recall / F1 | scikit-learn | `evaluate_lstm.py` |
| Latency end-to-end | timeit | `benchmark_latency.py` |

## RAB & Deployment

- VPS + Domain `.my.id` sesuai RAB → `docker compose up -d`.
- Catatan: `getUserMedia` butuh **HTTPS** di non-localhost. Pakai Caddy / Nginx + Let's Encrypt sebagai reverse proxy.

## Lisensi

Proyek akademik — bebas dipakai untuk keperluan pendidikan.
