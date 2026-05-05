import os
import threading


class YoloDetector:
    """YOLOv8 ROI detector. Returns bounding box (x1, y1, x2, y2) or None.

    Falls back to a whole-frame bbox if weights are missing — useful for
    UI development before the model is trained.
    """

    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not os.path.exists(self.weights_path):
            print(f"[YoloDetector] weights not found at {self.weights_path}, using fallback bbox")
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
        except Exception as exc:
            print(f"[YoloDetector] failed to load weights: {exc}")
            self.model = None

    def detect(self, frame_bgr):
        if self.model is None:
            h, w = frame_bgr.shape[:2]
            return (0, 0, w, h)

        # Ultralytics YOLO inference is not safe under concurrent calls
        # against the same model instance.
        with self._lock:
            results = self.model.predict(
                frame_bgr,
                conf=self.conf_threshold,
                verbose=False,
            )
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        return (int(x1), int(y1), int(x2), int(y2))
