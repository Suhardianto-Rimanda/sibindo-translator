"""Extract single-frame landmark vectors from photo files for letter/MLP training.

Expected input layout:
    data/raw/photos/<label>/*.jpg   (or .png, .jpeg)

Output:
    data/processed/letters/<label>/<idx>.npy   shape (126,)

Usage:
    python scripts/extract_landmarks_from_photo.py
    python scripts/extract_landmarks_from_photo.py --input data/raw/photos --out data/processed/letters
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.mediapipe_extractor import MediapipeExtractor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/photos")
    p.add_argument("--out", default="data/processed/letters")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    out_dir = Path(args.out)

    if not input_dir.exists():
        raise RuntimeError(f"Input dir not found: {input_dir}")

    # static_image_mode=True: better accuracy for individual photos (no tracking)
    extractor = MediapipeExtractor(normalize=False, static_image_mode=True)

    total = 0
    skipped = 0
    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        save_dir = out_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(save_dir.glob("*.npy")))
        images = [f for f in sorted(label_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]

        for i, img_file in enumerate(images):
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"[skip] {img_file.name} — cannot read image")
                skipped += 1
                continue

            features, _ = extractor.extract(frame)
            if features is None:
                print(f"[skip] {img_file.name} — no hand detected")
                skipped += 1
                continue

            save_path = save_dir / f"{existing + i:04d}.npy"
            np.save(save_path, features)
            print(f"[saved] {save_path}  shape={features.shape}")
            total += 1

    extractor.close()
    print(f"\nDone. {total} vectors saved, {skipped} skipped — output: {out_dir}")


if __name__ == "__main__":
    main()
