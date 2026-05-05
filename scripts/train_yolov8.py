"""Train YOLOv8 hand/person detector.

Prepare data.yaml with Roboflow/Kaggle dataset, then run:
    python scripts/train_yolov8.py --data data/yolo/data.yaml --epochs 100
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--base", default="yolov8n.pt", help="base weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="bisindo_roi")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )
    print("[train] done. best weights at runs/detect/{}/weights/best.pt".format(args.name))


if __name__ == "__main__":
    main()
