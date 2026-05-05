"""Prepare YOLO dataset structure + data.yaml from a flat folder of images + labels.

Expected input layout:
    raw/
        images/   (.jpg or .png)
        labels/   (matching .txt YOLO format)

Output:
    data/yolo/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml

Usage:
    python scripts/prepare_yolo_dataset.py --src data/raw/yolo_input --classes hand person
"""
import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder with images/ and labels/ subdirs")
    p.add_argument("--out", default="data/yolo")
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    src = Path(args.src)
    img_dir = src / "images"
    lbl_dir = src / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"Missing {img_dir} or {lbl_dir}")

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    random.shuffle(images)

    n = len(images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    split = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    out = Path(args.out)
    for s in split:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    for s, imgs in split.items():
        for img in imgs:
            shutil.copy2(img, out / "images" / s / img.name)
            label = lbl_dir / (img.stem + ".txt")
            if label.exists():
                shutil.copy2(label, out / "labels" / s / label.name)
        print(f"[prep] {s}: {len(imgs)} images")

    yaml_path = out / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(args.classes)}\n")
        f.write("names:\n")
        for i, c in enumerate(args.classes):
            f.write(f"  {i}: {c}\n")
    print(f"[prep] wrote {yaml_path}")


if __name__ == "__main__":
    main()
