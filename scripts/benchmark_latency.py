"""Benchmark end-to-end pipeline latency on a synthetic frame stream.

Measures: YOLOv8 detect -> MediaPipe extract -> LSTM predict (when buffer full).
Outputs percentiles (p50/p90/p95/p99) and stage breakdown.

Usage:
    python scripts/benchmark_latency.py --frames 200 --width 640 --height 480
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from app.services.yolo_detector import YoloDetector
from app.services.mediapipe_extractor import MediapipeExtractor
from app.services.lstm_classifier import LstmClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--source", default="synthetic", help="'synthetic' or path to video file")
    p.add_argument("--seq-len", type=int, default=30)
    return p.parse_args()


def synthetic_frame(w, h, t):
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    cx = int(w / 2 + 100 * np.sin(t))
    cy = int(h / 2 + 50 * np.cos(t * 0.7))
    cv2.circle(frame, (cx, cy), 60, (200, 200, 200), -1)
    cv2.rectangle(frame, (cx - 80, cy - 80), (cx + 80, cy + 80), (180, 180, 180), 4)
    return frame


def percentiles(arr):
    a = np.array(arr, dtype=np.float32)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def main():
    args = parse_args()

    detector = YoloDetector(Config.YOLO_WEIGHTS, Config.YOLO_CONF_THRESHOLD)
    extractor = MediapipeExtractor(normalize=True)
    classifier = LstmClassifier(Config.LSTM_MODEL_PATH, Config.LSTM_LABELS_PATH)

    if args.source == "synthetic":
        frame_iter = (synthetic_frame(args.width, args.height, i * 0.1)
                      for i in range(args.frames))
        total = args.frames
    else:
        cap = cv2.VideoCapture(args.source)
        def gen():
            for _ in range(args.frames):
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
        frame_iter = gen()
        total = args.frames

    yolo_times, mp_times, lstm_times, e2e_times = [], [], [], []
    buffer = []

    print(f"[bench] running {total} frames...")
    for frame in frame_iter:
        t0 = time.perf_counter()

        ts = time.perf_counter()
        bbox = detector.detect(frame)
        yolo_times.append((time.perf_counter() - ts) * 1000)

        if bbox is None:
            e2e_times.append((time.perf_counter() - t0) * 1000)
            continue

        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        ts = time.perf_counter()
        feats, _ = extractor.extract(roi) if roi.size > 0 else (None, None)
        mp_times.append((time.perf_counter() - ts) * 1000)

        if feats is not None:
            buffer.append(feats)
            if len(buffer) > args.seq_len:
                buffer = buffer[-args.seq_len:]
            if len(buffer) == args.seq_len:
                ts = time.perf_counter()
                classifier.predict(np.array(buffer))
                lstm_times.append((time.perf_counter() - ts) * 1000)

        e2e_times.append((time.perf_counter() - t0) * 1000)

    print("\n=== Latency Benchmark ===")
    print(f"Frames processed: {len(e2e_times)}")
    if yolo_times:
        print(f"\nYOLOv8 detect: {percentiles(yolo_times)}")
    if mp_times:
        print(f"MediaPipe extract: {percentiles(mp_times)}")
    if lstm_times:
        print(f"LSTM predict: {percentiles(lstm_times)}")
    print(f"\nEnd-to-end: {percentiles(e2e_times)}")
    print(f"Throughput: {len(e2e_times) / (sum(e2e_times) / 1000):.1f} fps")


if __name__ == "__main__":
    main()
