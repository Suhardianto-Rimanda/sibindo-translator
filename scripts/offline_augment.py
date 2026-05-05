"""Offline augmentation: write augmented .npy files to disk.

Use when you want to enlarge the dataset persistently rather than
augmenting on-the-fly during training.

Usage:
    python scripts/offline_augment.py --factor 5
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.landmark_augmenter import augment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="data/augmented")
    p.add_argument("--factor", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    src = Path(args.data)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    for label_dir in sorted(src.iterdir()):
        if not label_dir.is_dir():
            continue
        out_label = dst / label_dir.name
        out_label.mkdir(parents=True, exist_ok=True)

        files = sorted(label_dir.glob("*.npy"))
        for npy in files:
            seq = np.load(npy)
            np.save(out_label / npy.name, seq)
            for k in range(args.factor):
                aug = augment(seq)
                np.save(out_label / f"{npy.stem}_aug{k:02d}.npy", aug)

        produced = len(list(out_label.glob('*.npy')))
        print(f"[augment] {label_dir.name}: {len(files)} -> {produced}")


if __name__ == "__main__":
    main()
