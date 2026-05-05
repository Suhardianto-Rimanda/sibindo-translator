"""Two-stage detection pipeline orchestrator.

Stage 1: YOLOv8 detects hand/person ROI.
Stage 2: MediaPipe extracts landmarks from cropped ROI (with normalization).
Buffer fills until sequence_length, then LSTM classifies the gesture.
NLP module dedupes/smooths predictions and assembles a sentence.

Sessions: each session_id has its own buffer + NLP state. Models are shared.
"""
import time
from collections import OrderedDict, deque
from threading import Lock

import numpy as np

from app.services.yolo_detector import YoloDetector
from app.services.mediapipe_extractor import MediapipeExtractor
from app.services.lstm_classifier import LstmClassifier
from app.services.letter_classifier import LetterClassifier
from app.services.nlp_processor import NlpProcessor
from app.services.tts_service import TtsService
from app.utils.logger import log


class SessionState:
    __slots__ = ("buffer", "nlp", "last_seen", "lock")

    def __init__(self, sequence_length: int):
        self.buffer = deque(maxlen=sequence_length)
        self.nlp = NlpProcessor()
        self.last_seen = time.time()
        self.lock = Lock()


class Pipeline:
    SESSION_TTL_SECONDS = 1800  # 30 minutes idle
    DEFAULT_MAX_SESSIONS = 1000

    def __init__(self, config):
        self.config = config
        self.sequence_length = config["LSTM_SEQUENCE_LENGTH"]
        self.lstm_threshold = config["LSTM_CONF_THRESHOLD"]
        self.max_sessions = int(config.get("MAX_SESSIONS", self.DEFAULT_MAX_SESSIONS))

        self.detector = YoloDetector(
            weights_path=config["YOLO_WEIGHTS"],
            conf_threshold=config["YOLO_CONF_THRESHOLD"],
        )
        self.extractor = MediapipeExtractor(
            normalize=True,
            static_image_mode=True,
            model_path=config["MEDIAPIPE_MODEL_PATH"],
        )
        self.classifier = LstmClassifier(
            model_path=config["LSTM_MODEL_PATH"],
            labels_path=config["LSTM_LABELS_PATH"],
        )
        self.letter_classifier = LetterClassifier(
            model_path=config["LETTER_MODEL_PATH"],
            labels_path=config["LETTER_LABELS_PATH"],
        )
        self.letter_threshold = config["LETTER_CONF_THRESHOLD"]

        self.tts = TtsService(
            lang=config["TTS_LANG"],
            output_dir=config["TTS_OUTPUT_DIR"],
        )

        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._sessions_lock = Lock()

    def _get_session(self, session_id: str) -> SessionState:
        with self._sessions_lock:
            self._evict_stale_locked()
            state = self._sessions.get(session_id)
            if state is None:
                while len(self._sessions) >= self.max_sessions:
                    evicted_sid, _ = self._sessions.popitem(last=False)
                    log.info(f"LRU evicted session (cap={self.max_sessions}): {evicted_sid}")
                state = SessionState(self.sequence_length)
                self._sessions[session_id] = state
                log.info(f"new session: {session_id}")
            else:
                self._sessions.move_to_end(session_id)
            state.last_seen = time.time()
            return state

    def _evict_stale_locked(self):
        now = time.time()
        stale = [
            sid for sid, s in self._sessions.items()
            if now - s.last_seen > self.SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del self._sessions[sid]
            log.info(f"evicted stale session: {sid}")

    def process_frame(self, frame_bgr, session_id: str = "default", mode: str = "word"):
        if mode == "letter":
            return self._process_letter(frame_bgr, session_id)
        return self._process_word(frame_bgr, session_id)

    def _extract_roi_features(self, frame_bgr, result, session):
        """Shared ROI + landmark extraction. Returns (features, viz) or (None, None) on failure."""
        bbox = self.detector.detect(frame_bgr)
        if bbox is None:
            return None, None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        # Clamp to frame and reject degenerate boxes. YOLO can return coords
        # slightly outside the image or with x1>=x2 after rounding.
        x1 = max(0, min(int(x1), w))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h))
        y2 = max(0, min(int(y2), h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None, None
        bbox = (x1, y1, x2, y2)

        result["detected"] = True
        result["bbox"] = bbox
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        features, viz = self.extractor.extract(roi)
        if features is None:
            return None, None

        if viz and viz.get("hands"):
            roi_w, roi_h = (x2 - x1), (y2 - y1)
            for hand in viz["hands"]:
                hand["points"] = [
                    [(x1 + p[0] * roi_w) / w, (y1 + p[1] * roi_h) / h]
                    for p in hand["points"]
                ]
        result["landmarks"] = viz
        return features, viz

    def _base_result(self, session):
        return {
            "detected": False,
            "bbox": None,
            "landmarks": None,
            "buffer_size": 0,
            "word": None,
            "confidence": 0.0,
            "sentence": session.nlp.sentence(),
            "history": session.nlp.words(),
            "audio_url": None,
            "latency_ms": 0,
            "mode": None,
        }

    def _process_word(self, frame_bgr, session_id: str):
        t0 = time.time()
        session = self._get_session(session_id)
        result = self._base_result(session)
        result["mode"] = "word"

        features, _ = self._extract_roi_features(frame_bgr, result, session)
        if features is None:
            return self._finalize(result, session, t0)

        with session.lock:
            session.buffer.append(features)
            result["buffer_size"] = len(session.buffer)

            if len(session.buffer) < self.sequence_length:
                return self._finalize(result, session, t0)

            sequence = np.array(session.buffer)

        label, confidence = self.classifier.predict(sequence)
        result["word"] = label
        result["confidence"] = float(confidence)

        if confidence < self.lstm_threshold:
            return self._finalize(result, session, t0)

        added = session.nlp.add_word(label)
        result["sentence"] = session.nlp.sentence()
        result["history"] = session.nlp.words()

        if added:
            audio_path = self.tts.synthesize(label)
            result["audio_url"] = audio_path
            log.info(f"[{session_id}] word accepted: {label} (conf={confidence:.2f})")

        return self._finalize(result, session, t0)

    def _process_letter(self, frame_bgr, session_id: str):
        t0 = time.time()
        session = self._get_session(session_id)
        result = self._base_result(session)
        result["mode"] = "letter"
        result["buffer_size"] = 1  # letter mode always single-frame

        features, _ = self._extract_roi_features(frame_bgr, result, session)
        if features is None:
            return self._finalize(result, session, t0)

        label, confidence = self.letter_classifier.predict(features)
        result["word"] = label
        result["confidence"] = float(confidence)

        if confidence < self.letter_threshold:
            return self._finalize(result, session, t0)

        added = session.nlp.add_word(label)
        result["sentence"] = session.nlp.sentence()
        result["history"] = session.nlp.words()

        if added:
            log.info(f"[{session_id}] letter accepted: {label} (conf={confidence:.2f})")

        return self._finalize(result, session, t0)

    def _finalize(self, result, session, t0):
        result["sentence"] = session.nlp.sentence()
        result["history"] = session.nlp.words()
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    def reset(self, session_id: str = "default"):
        session = self._get_session(session_id)
        with session.lock:
            session.buffer.clear()
            session.nlp.reset()
        log.info(f"[{session_id}] pipeline reset")

    def session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)
