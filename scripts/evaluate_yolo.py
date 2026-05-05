"""Evaluate trained YOLOv8 model. Reports mAP@0.5, mAP@0.5:0.95, precision, recall.

Usage:
    python scripts/evaluate_yolo.py --weights models/yolov8/best.pt --data data/yolo/data.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/yolov8/best.pt")
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="models/yolov8/eval")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, save_json=True)

    summary = {
        "mAP_0.5": float(metrics.box.map50),
        "mAP_0.5_0.95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness),
    }
    print(f"\n=== YOLOv8 Eval ===")
    for k, v in summary.items():
        print(f"{k:>15}: {v:.4f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] saved -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
