"""Evaluate trained LSTM with confusion matrix, classification report, accuracy plots.

Usage:
    python scripts/evaluate_lstm.py
    python scripts/evaluate_lstm.py --test-split 0.2

Outputs to models/lstm/eval/:
    confusion_matrix.png
    classification_report.txt
    accuracy_loss.png  (if training_history.json exists)
    metrics.json
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.landmark_normalizer import normalize_sequence


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--model", default="models/lstm/bisindo_lstm.h5")
    p.add_argument("--labels", default="models/lstm/labels.json")
    p.add_argument("--history", default="models/lstm/training_history.json")
    p.add_argument("--split", default="models/lstm/split.npz",
                   help="split file written by train_lstm.py; falls back to re-split if missing")
    p.add_argument("--out", default="models/lstm/eval")
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_test_set_from_split(split_path: Path, labels: list):
    """Authoritative path: replay the exact val indices recorded at training time."""
    data = np.load(split_path, allow_pickle=False)
    val_idx = data["val_idx"]
    files = data["files"]
    saved_labels = list(data["labels"])
    if saved_labels != labels:
        raise RuntimeError(
            f"label set mismatch between split file and labels.json: "
            f"{saved_labels} vs {labels}. Retrain or pass --labels to match."
        )
    normalize = bool(data["normalize"]) if "normalize" in data.files else True

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    X, y = [], []
    for i in val_idx:
        path = Path(files[i])
        seq = np.load(path)
        if normalize:
            seq = normalize_sequence(seq)
        X.append(seq)
        y.append(label_to_idx[path.parent.name])
    return np.array(X, dtype=np.float32), np.array(y)


def load_test_set_fallback(data_dir: Path, labels: list, test_split: float, seed: int):
    """Legacy path: replicate train script's stratified split using the same seed.
    Only correct if seed matches the seed used at training time."""
    X, y = [], []
    for idx, label in enumerate(labels):
        for npy in sorted((data_dir / label).glob("*.npy")):
            seq = np.load(npy)
            X.append(normalize_sequence(seq))
            y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    return X_test, y_test


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.6),) * 2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — LSTM BISINDO")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_history(history_path, out_path):
    if not Path(history_path).exists():
        return False
    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h.get("accuracy", []), label="train")
    axes[0].plot(h.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(h.get("loss", []), label="train")
    axes[1].plot(h.get("val_loss", []), label="val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    split_path = Path(args.split)
    if split_path.exists():
        print(f"[eval] loading val set from split file {split_path}")
        X_test, y_test = load_test_set_from_split(split_path, labels)
    else:
        print(
            f"[eval] split file {split_path} not found — falling back to seeded re-split. "
            "WARNING: results are only valid if --seed matches the training run."
        )
        X_test, y_test = load_test_set_fallback(Path(args.data), labels, args.test_split, args.seed)
    print(f"[eval] test size: {len(X_test)}")

    print(f"[eval] loading model {args.model}")
    model = load_model(args.model)
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n=== Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    print("\n" + report)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels, out_dir / "confusion_matrix.png")
    print(f"[eval] saved confusion matrix")

    if plot_history(args.history, out_dir / "accuracy_loss.png"):
        print(f"[eval] saved accuracy/loss plot")

    metrics = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
        "labels": labels,
        "test_size": len(X_test),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] saved metrics.json -> {out_dir}")


if __name__ == "__main__":
    main()
