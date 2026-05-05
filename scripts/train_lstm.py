"""Train LSTM on collected landmark sequences with augmentation + normalization.

Expected layout (from collect_landmarks.py):
    data/processed/<label>/*.npy   each shape (frames, 126)

Outputs:
    models/lstm/bisindo_lstm.h5
    models/lstm/labels.json
    models/lstm/training_history.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.landmark_normalizer import normalize_sequence
from app.services.landmark_augmenter import augment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--model-out", default="models/lstm/bisindo_lstm.h5")
    p.add_argument("--labels-out", default="models/lstm/labels.json")
    p.add_argument("--history-out", default="models/lstm/training_history.json")
    p.add_argument("--split-out", default="models/lstm/split.npz",
                   help="train/val split indices + file list, consumed by evaluate_lstm.py")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment-factor", type=int, default=4,
                   help="generate N augmented samples per original")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(data_dir: Path, normalize: bool = True):
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not labels:
        raise RuntimeError(f"No label folders found in {data_dir}")

    X, y, files = [], [], []
    for idx, label in enumerate(labels):
        for npy in sorted((data_dir / label).glob("*.npy")):
            seq = np.load(npy)
            if normalize:
                seq = normalize_sequence(seq)
            X.append(seq)
            y.append(idx)
            files.append(str(npy))

    return np.array(X, dtype=np.float32), np.array(y), labels, files


def augment_dataset(X, y, factor: int):
    if factor <= 0:
        return X, y
    aug_X, aug_y = [list(X)], [list(y)]
    for _ in range(factor):
        for seq, lbl in zip(X, y):
            aug_X[0].append(augment(seq))
            aug_y[0].append(lbl)
    return np.array(aug_X[0], dtype=np.float32), np.array(aug_y[0])


def build_model(seq_len, feat_dim, num_classes, lr):
    model = Sequential([
        LSTM(64, return_sequences=True, activation="tanh",
             input_shape=(seq_len, feat_dim)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, return_sequences=False, activation="tanh"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    X, y, labels, files = load_dataset(data_dir, normalize=not args.no_normalize)
    print(f"[train] loaded {len(X)} sequences across {len(labels)} classes: {labels}")
    print(f"[train] class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    seq_len, feat_dim = X.shape[1], X.shape[2]

    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=args.seed, stratify=y
    )
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"[train] train: {len(X_train)}, val: {len(X_val)}")

    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.split_out,
        train_idx=train_idx,
        val_idx=val_idx,
        files=np.array(files),
        labels=np.array(labels),
        seed=args.seed,
        normalize=not args.no_normalize,
    )
    print(f"[train] saved split indices to {args.split_out}")

    X_train, y_train = augment_dataset(X_train, y_train, args.augment_factor)
    print(f"[train] after augmentation -> train: {len(X_train)}")

    y_train_cat = to_categorical(y_train, num_classes=len(labels))
    y_val_cat = to_categorical(y_val, num_classes=len(labels))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(labels)),
        y=y_train,
    )
    cw_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"[train] class weights: {cw_dict}")

    model = build_model(seq_len, feat_dim, len(labels), args.lr)
    model.summary()

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        class_weight=cw_dict,
    )

    with open(args.labels_out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"[train] saved labels to {args.labels_out}")

    with open(args.history_out, "w", encoding="utf-8") as f:
        json.dump({k: list(map(float, v)) for k, v in history.history.items()}, f, indent=2)
    print(f"[train] saved history to {args.history_out}")


if __name__ == "__main__":
    main()
