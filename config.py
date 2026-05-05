import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


_DEV_SECRET_KEY = "dev-secret-key"  # sentinel — refused at app start when DEBUG=0


class Config:
    # Real check happens in create_app(); keep import side-effect-free for tests.
    SECRET_KEY = os.getenv("SECRET_KEY", _DEV_SECRET_KEY)
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "5")) * 1024 * 1024
    MAX_FRAME_PIXELS = int(os.getenv("MAX_FRAME_PIXELS", str(1920 * 1080)))
    MAX_FRAME_DIMENSION = int(os.getenv("MAX_FRAME_DIMENSION", "4096"))

    RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "10 per second; 600 per minute")
    RATE_LIMIT_RESET = os.getenv("RATE_LIMIT_RESET", "5 per second; 60 per minute")
    RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "120 per minute")

    MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))

    MEDIAPIPE_MODEL_PATH = os.getenv("MEDIAPIPE_MODEL_PATH", "models/mediapipe/hand_landmarker.task")

    YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "models/yolov8/best.pt")
    YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", 0.5))

    LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm/bisindo_lstm.h5")
    LSTM_LABELS_PATH = os.getenv("LSTM_LABELS_PATH", "models/lstm/labels.json")
    LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 30))
    LSTM_CONF_THRESHOLD = float(os.getenv("LSTM_CONF_THRESHOLD", 0.7))

    LETTER_MODEL_PATH = os.getenv("LETTER_MODEL_PATH", "models/letter_mlp/bisindo_letter.h5")
    LETTER_LABELS_PATH = os.getenv("LETTER_LABELS_PATH", "models/letter_mlp/labels.json")
    LETTER_CONF_THRESHOLD = float(os.getenv("LETTER_CONF_THRESHOLD", 0.7))

    TTS_LANG = os.getenv("TTS_LANG", "id")
    TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "app/static/audio")
