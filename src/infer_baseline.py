"""Step 3 — Run pre-trained YOLO11n on the dataset's test split and visualize."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import (
    DATA_DIR,
    OUTPUTS,
    list_test_images,
    pick_examples,
    plot_image_grid,
)

BASELINE_WEIGHTS = "yolo11n.pt"  # auto-downloaded by ultralytics on first use


def main() -> int:
    images = list_test_images(DATA_DIR)
    print(f"Found {len(images)} test images.")

    model = YOLO(BASELINE_WEIGHTS)
    print(f"Loaded baseline model: {BASELINE_WEIGHTS} (COCO-pretrained, {len(model.names)} classes)")

    # Predict on the whole test split. Ultralytics writes annotated images
    # under outputs/baseline_preds/predict/.
    save_dir = OUTPUTS / "baseline_preds"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.predict(
        source=[str(p) for p in images],
        save=True,
        conf=0.25,
        project=str(save_dir),
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    # Build a 6-image grid that we'll reuse in Step 5 for the side-by-side.
    examples = pick_examples(images, n=6)
    annotated = []
    titles = []
    for p in examples:
        res = model.predict(str(p), conf=0.25, verbose=False)[0]
        annotated.append(res.plot())  # BGR numpy with bboxes/labels/conf drawn
        n_det = 0 if res.boxes is None else len(res.boxes)
        titles.append(f"{p.name}  ({n_det} det)")

    grid_path = OUTPUTS / "comparison" / "baseline_grid.png"
    plot_image_grid(annotated, titles, grid_path)
    print(f"Saved baseline grid -> {grid_path}")

    # Persist the picked example paths so evaluate.py uses the same images.
    examples_file = OUTPUTS / "comparison" / "examples.txt"
    examples_file.write_text("\n".join(str(p) for p in examples))
    print(f"Recorded example image list -> {examples_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
