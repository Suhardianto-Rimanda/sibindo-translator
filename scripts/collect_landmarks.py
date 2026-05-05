"""Collect MediaPipe landmark sequences from webcam for LSTM training.

Usage:
    python scripts/collect_landmarks.py --label halo --samples 30 --frames 30

Saves: data/processed/<label>/<sample_idx>.npy  with shape (frames, 126).
Press SPACE to start each sample, ESC to quit.

Note: Stored landmarks are RAW (not normalized) — normalization is applied
during training/inference. This keeps raw data flexible.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.mediapipe_extractor import MediapipeExtractor
from app.services.landmark_normalizer import FEATURE_VECTOR_SIZE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="gesture label name")
    p.add_argument("--samples", type=int, default=30, help="number of sequences")
    p.add_argument("--frames", type=int, default=30, help="frames per sequence")
    p.add_argument("--out", default="data/processed", help="output root dir")
    p.add_argument("--camera", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = MediapipeExtractor(normalize=False, static_image_mode=False)  # RAW landmarks, VIDEO tracking
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    sample_idx = len(list(out_dir.glob("*.npy")))
    print(f"[collect] starting at sample #{sample_idx} for label '{args.label}'")
    print("[collect] SPACE = record sequence, ESC = quit")

    while sample_idx < args.samples:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, f"Label: {args.label}  Sample: {sample_idx}/{args.samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to record, ESC to quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("collect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE
            sequence = np.zeros((args.frames, FEATURE_VECTOR_SIZE), dtype=np.float32)
            for i in range(args.frames):
                ok, frame = cap.read()
                if not ok:
                    break
                features, _ = extractor.extract(frame)
                if features is not None:
                    sequence[i] = features
                cv2.putText(frame, f"REC {i+1}/{args.frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("collect", frame)
                cv2.waitKey(1)

            path = out_dir / f"{sample_idx:03d}.npy"
            np.save(path, sequence)
            print(f"[collect] saved {path}")
            sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()
