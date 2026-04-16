import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Путь к best.pt")
    parser.add_argument("--data", type=str, required=True, help="Путь к data.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True
    )

    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()