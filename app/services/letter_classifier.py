import json
import os
import threading

import numpy as np


class LetterClassifier:
    """MLP classifier for static letter/finger-spelling gestures.

    Input: single 126-dim landmark vector (one frame).
    Returns (label, confidence). Falls back to stub if weights missing.
    """

    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels: list[str] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.labels = [data[str(i)] for i in range(len(data))] if isinstance(data, dict) else data
        else:
            print(f"[LetterClassifier] labels not found at {self.labels_path}")

        if not os.path.exists(self.model_path):
            print(f"[LetterClassifier] model not found at {self.model_path}, predictions disabled")
            return

        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(self.model_path)
        except Exception as exc:
            print(f"[LetterClassifier] failed to load: {exc}")

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """features: (126,) → (label, confidence)"""
        if self.model is None or not self.labels:
            return ("__unknown__", 0.0)
        x = np.expand_dims(features, axis=0)
        with self._lock:
            probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return (self.labels[idx], float(probs[idx]))
