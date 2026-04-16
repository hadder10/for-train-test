import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True, help="Файл или папка с изображениями")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/predict")
    parser.add_argument("--name", type=str, default="empty_shelf_test")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=True,
        project=args.project,
        name=args.name,
        verbose=True
    )

    print(f"Saved to: {Path(args.project) / args.name}")
    print(results)


if __name__ == "__main__":
    main()