import json
import os
import threading

import numpy as np


class LstmClassifier:
    """Wraps the trained LSTM model. Returns (label, confidence).

    If the model file is missing, returns a stub prediction so the rest of
    the pipeline can be developed before training is finished.
    """

    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
        else:
            print(f"[LstmClassifier] labels file not found at {self.labels_path}")

        if not os.path.exists(self.model_path):
            print(f"[LstmClassifier] model not found at {self.model_path}, predictions disabled")
            return

        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path)
        except Exception as exc:
            print(f"[LstmClassifier] failed to load model: {exc}")
            self.model = None

    def predict(self, sequence: np.ndarray):
        if self.model is None or not self.labels:
            return ("__unknown__", 0.0)

        x = np.expand_dims(sequence, axis=0)
        # Keras model.predict is not reliably reentrant; serialize calls.
        with self._lock:
            probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return (self.labels[idx], float(probs[idx]))
