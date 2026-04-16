from pathlib import Path
import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Путь к data.yaml")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Базовая модель")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="empty_shelf_yolo11m")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU недоступен, а указан device не cpu.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
        patience=args.patience,
        save=True,
        device=args.device,
        workers=args.workers,
        optimizer="auto",
        cos_lr=True,
        mosaic=0.5,
        mixup=0.0,
        degrees=3.0,
        translate=0.08,
        scale=0.20,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        close_mosaic=10,
        cache=False,
        plots=True,
        val=True,
    )

    print(results)
    run_dir = Path(args.project) / args.name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()