"""Step 4 — Fine-tune YOLO11n on the custom dataset via transfer learning."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from utils import DATA_DIR, RUNS, RUN_NAME


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None, help="auto-detect if omitted")
    parser.add_argument("--freeze", type=int, default=0, help="freeze first N layers (0 = full fine-tune)")
    args = parser.parse_args()

    device = args.device or pick_device()
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run src/download_dataset.py first.", file=sys.stderr)
        return 1

    print(f"Device: {device}")
    print(f"Data:   {data_yaml}")
    print(f"Config: epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} freeze={args.freeze}")

    model = YOLO("yolo11n.pt")  # start from COCO weights -> transfer learning
    t0 = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        project=str(RUNS),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        freeze=args.freeze,
        seed=42,
        plots=True,
    )
    dt = time.time() - t0
    print(f"\nTraining finished in {dt/60:.1f} min")

    best = RUNS / RUN_NAME / "weights" / "best.pt"
    print(f"Best weights: {best}  (exists={best.exists()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
