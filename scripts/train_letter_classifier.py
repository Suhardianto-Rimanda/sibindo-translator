"""Train MLP classifier for static letter/finger-spelling gestures.

Expected layout (from extract_landmarks_from_photo.py):
    data/processed/letters/<label>/*.npy   each shape (126,)

Outputs:
    models/letter_mlp/bisindo_letter.h5
    models/letter_mlp/labels.json
    models/letter_mlp/training_history.json

Usage:
    python scripts/train_letter_classifier.py
    python scripts/train_letter_classifier.py --epochs 150 --batch 32
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.landmark_normalizer import normalize_feature_vector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/letters")
    p.add_argument("--model-out", default="models/letter_mlp/bisindo_letter.h5")
    p.add_argument("--labels-out", default="models/letter_mlp/labels.json")
    p.add_argument("--history-out", default="models/letter_mlp/training_history.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_dataset(data_dir: Path, normalize: bool = True):
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not labels:
        raise RuntimeError(f"No label folders found in {data_dir}")

    X, y = [], []
    for idx, label in enumerate(labels):
        for npy in sorted((data_dir / label).glob("*.npy")):
            vec = np.load(npy)
            if normalize:
                vec = normalize_feature_vector(vec)
            X.append(vec)
            y.append(idx)

    return np.array(X, dtype=np.float32), np.array(y), labels


def build_model(feat_dim: int, num_classes: int, lr: float) -> Sequential:
    model = Sequential([
        Dense(256, activation="relu", input_shape=(feat_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
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
    X, y, labels = load_dataset(data_dir, normalize=not args.no_normalize)
    print(f"[train_letter] loaded {len(X)} samples across {len(labels)} classes: {labels}")
    print(f"[train_letter] class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    feat_dim = X.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"[train_letter] train: {len(X_train)}, val: {len(X_val)}")

    y_train_cat = to_categorical(y_train, num_classes=len(labels))
    y_val_cat = to_categorical(y_val, num_classes=len(labels))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(labels)),
        y=y_train,
    )
    cw_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"[train_letter] class weights: {cw_dict}")

    model = build_model(feat_dim, len(labels), args.lr)
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
    print(f"[train_letter] saved labels to {args.labels_out}")

    with open(args.history_out, "w", encoding="utf-8") as f:
        json.dump({k: list(map(float, v)) for k, v in history.history.items()}, f, indent=2)
    print(f"[train_letter] saved history to {args.history_out}")


if __name__ == "__main__":
    main()
