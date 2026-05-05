"""Extract landmark sequences from video files for LSTM (word) training.

Expected input layout:
    data/raw/videos/<label>/*.mp4   (or .avi, .mov)

Output:
    data/processed/words/<label>/<idx>.npy   shape (30, 126)

Usage:
    python scripts/extract_landmarks_from_video.py
    python scripts/extract_landmarks_from_video.py --input data/raw/videos --out data/processed/words --frames 30
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.mediapipe_extractor import MediapipeExtractor
from app.services.landmark_normalizer import FEATURE_VECTOR_SIZE

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/videos")
    p.add_argument("--out", default="data/processed/words")
    p.add_argument("--frames", type=int, default=30, help="sequence length (frames)")
    return p.parse_args()


def sample_frames(video_path: Path, target_frames: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        return []

    if len(all_frames) >= target_frames:
        idxs = np.linspace(0, len(all_frames) - 1, target_frames, dtype=int)
        return [all_frames[i] for i in idxs]

    # pad by repeating last frame
    pad = [all_frames[-1]] * (target_frames - len(all_frames))
    return all_frames + pad


def extract_sequence(video_path: Path, extractor: MediapipeExtractor, target_frames: int) -> np.ndarray:
    frames = sample_frames(video_path, target_frames)
    if not frames:
        return None

    sequence = np.zeros((target_frames, FEATURE_VECTOR_SIZE), dtype=np.float32)
    for i, frame in enumerate(frames):
        features, _ = extractor.extract(frame)
        if features is not None:
            sequence[i] = features
    return sequence


def main():
    args = parse_args()
    input_dir = Path(args.input)
    out_dir = Path(args.out)

    if not input_dir.exists():
        raise RuntimeError(f"Input dir not found: {input_dir}")

    extractor = MediapipeExtractor(normalize=False, static_image_mode=False)

    total = 0
    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        save_dir = out_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(save_dir.glob("*.npy")))
        videos = [f for f in sorted(label_dir.iterdir()) if f.suffix.lower() in VIDEO_EXTS]

        for i, video_file in enumerate(videos):
            seq = extract_sequence(video_file, extractor, args.frames)
            if seq is None:
                print(f"[skip] {video_file.name} — no frames extracted")
                continue
            save_path = save_dir / f"{existing + i:04d}.npy"
            np.save(save_path, seq)
            print(f"[saved] {save_path}  shape={seq.shape}")
            total += 1

    extractor.close()
    print(f"\nDone. {total} sequences saved to {out_dir}")


if __name__ == "__main__":
    main()
