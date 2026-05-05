"""Inspect collected landmark dataset: class distribution, sequence stats, missing-frame ratio.

Usage:
    python scripts/inspect_dataset.py
"""
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"[inspect] {data_dir} does not exist")
        return

    print(f"\n=== Dataset Report: {data_dir} ===\n")
    print(f"{'Label':<20}{'Samples':>10}{'Frames':>10}{'FeatDim':>10}{'EmptyRatio':>14}")
    print("-" * 64)

    total_samples = 0
    label_counts = {}

    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        files = list(label_dir.glob("*.npy"))
        if not files:
            continue

        empty_frames = 0
        total_frames = 0
        seq_shape = None

        for npy in files:
            seq = np.load(npy)
            seq_shape = seq.shape
            total_frames += seq.shape[0]
            empty_frames += int(np.sum(np.all(seq == 0, axis=1)))

        empty_ratio = empty_frames / total_frames if total_frames else 0
        print(f"{label_dir.name:<20}{len(files):>10}{seq_shape[0]:>10}"
              f"{seq_shape[1]:>10}{empty_ratio:>14.2%}")

        total_samples += len(files)
        label_counts[label_dir.name] = len(files)

    print("-" * 64)
    print(f"{'TOTAL':<20}{total_samples:>10}")

    if label_counts:
        counts = list(label_counts.values())
        print(f"\nClass balance:")
        print(f"  min: {min(counts)}  max: {max(counts)}  mean: {np.mean(counts):.1f}")
        ratio = max(counts) / max(min(counts), 1)
        print(f"  imbalance ratio: {ratio:.2f}x")
        if ratio > 2.0:
            print("  WARNING: class imbalance > 2x — collect more samples for minority classes")


if __name__ == "__main__":
    main()
