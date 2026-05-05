import hashlib
import os
import threading
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.services.landmark_normalizer import (
    NUM_HAND_LANDMARKS,
    LANDMARK_DIMS,
    FEATURE_VECTOR_SIZE,
    normalize_feature_vector,
)

_DEFAULT_MODEL_PATH = "models/mediapipe/hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
# Optional: when set, downloaded/cached model must match this digest. Pin via
# MEDIAPIPE_MODEL_SHA256 env after recording the digest from a trusted run.
_EXPECTED_SHA256 = (os.getenv("MEDIAPIPE_MODEL_SHA256") or "").strip().lower()


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model(model_path: str) -> str:
    if os.path.exists(model_path):
        if _EXPECTED_SHA256:
            actual = _sha256_of(model_path)
            if actual != _EXPECTED_SHA256:
                raise RuntimeError(
                    f"MediaPipe model SHA256 mismatch at {model_path}: "
                    f"expected {_EXPECTED_SHA256}, got {actual}. "
                    "Delete the file to redownload, or update MEDIAPIPE_MODEL_SHA256."
                )
        return model_path
    parent = os.path.dirname(model_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = model_path + ".part"
    print(f"[MediapipeExtractor] downloading hand_landmarker.task to {model_path} ...")
    urllib.request.urlretrieve(_MODEL_URL, tmp_path)
    actual = _sha256_of(tmp_path)
    if _EXPECTED_SHA256 and actual != _EXPECTED_SHA256:
        os.remove(tmp_path)
        raise RuntimeError(
            f"Downloaded MediaPipe model SHA256 mismatch: "
            f"expected {_EXPECTED_SHA256}, got {actual}"
        )
    os.replace(tmp_path, model_path)
    if _EXPECTED_SHA256:
        print(f"[MediapipeExtractor] download verified (sha256={actual})")
    else:
        print(
            f"[MediapipeExtractor] download complete (sha256={actual}). "
            "Pin this value via MEDIAPIPE_MODEL_SHA256 env to enforce integrity on next start."
        )
    return model_path


class MediapipeExtractor:
    """Extract 21x3 landmarks per hand (up to 2 hands) into a flat vector.

    Uses MediaPipe Tasks API (HandLandmarker). Model file auto-downloads on first run.

    Output:
        features: (126,) flat vector — left (63) + right (63), zero-padded for missing hand
        viz: dict with raw landmark coords (for frontend overlay drawing)
    """

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5,
                 normalize: bool = True, static_image_mode: bool = True,
                 model_path: str = _DEFAULT_MODEL_PATH):
        # static_image_mode defaults True so a single shared instance is safe
        # across web sessions: VIDEO mode keeps per-stream tracking state, which
        # is corrupted when frames from different sessions are interleaved.
        self.normalize = normalize
        self.static_image_mode = static_image_mode
        self._last_timestamp_ms = 0
        self._lock = threading.Lock()

        model_path = _ensure_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        running_mode = mp_vision.RunningMode.IMAGE if static_image_mode else mp_vision.RunningMode.VIDEO

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def extract(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # HandLandmarker is not thread-safe; serialize all calls.
        with self._lock:
            if self.static_image_mode:
                result = self.landmarker.detect(mp_image)
            else:
                ts = int(time.monotonic() * 1000)
                if ts <= self._last_timestamp_ms:
                    ts = self._last_timestamp_ms + 1
                self._last_timestamp_ms = ts
                result = self.landmarker.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return None, None

        left = np.zeros(NUM_HAND_LANDMARKS * LANDMARK_DIMS, dtype=np.float32)
        right = np.zeros(NUM_HAND_LANDMARKS * LANDMARK_DIMS, dtype=np.float32)
        viz = {"hands": []}

        for hand_landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )
            flat = coords.flatten()
            label = handedness_list[0].category_name  # "Left" or "Right"

            if label == "Left":
                left = flat
            else:
                right = flat

            viz["hands"].append({
                "label": label,
                "points": coords[:, :2].tolist(),
            })

        features = np.concatenate([left, right])
        if self.normalize:
            features = normalize_feature_vector(features)

        return features, viz

    def close(self):
        self.landmarker.close()
