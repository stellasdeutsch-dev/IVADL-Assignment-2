"""Step 5 — Evaluate baseline vs. fine-tuned model on the test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from utils import (
    DATA_DIR,
    OUTPUTS,
    RUN_NAME,
    RUNS,
    plot_side_by_side,
)


def metrics_dict(result, class_names: list[str]) -> dict:
    """Pull precision/recall/mAP from an Ultralytics val result."""
    box = result.box
    p, r, map50, map5095 = box.mean_results()
    per_class = {}
    try:
        ap50 = box.ap50  # array of length nc
        ap5095 = box.ap  # column 0 is map50, but `box.maps` is per-class map50-95
        maps = box.maps  # per-class mAP50-95
        for i, name in enumerate(class_names):
            if i < len(ap50):
                per_class[name] = {
                    "mAP50": float(ap50[i]),
                    "mAP50-95": float(maps[i]),
                }
    except Exception:  # noqa: BLE001 — best-effort per-class breakdown
        pass
    return {
        "precision": float(p),
        "recall": float(r),
        "mAP50": float(map50),
        "mAP50-95": float(map5095),
        "per_class": per_class,
    }


def annotate(model: YOLO, image_path: Path):
    res = model.predict(str(image_path), conf=0.25, verbose=False)[0]
    return res.plot()


def main() -> int:
    data_yaml = DATA_DIR / "data.yaml"
    with data_yaml.open() as f:
        meta = yaml.safe_load(f)
    class_names = list(meta.get("names", []))
    print(f"Classes ({len(class_names)}): {class_names}")

    best = RUNS / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        print(f"ERROR: fine-tuned weights not found at {best}. Train first.", file=sys.stderr)
        return 1

    print("\n--- Evaluating baseline (yolo11n.pt, COCO classes) on custom data ---")
    baseline = YOLO("yolo11n.pt")
    # NOTE: the baseline has COCO classes, so mAP against custom labels will be ~0.
    # This is the expected baseline and what the assignment asks us to compare.
    baseline_res = baseline.val(
        data=str(data_yaml),
        split="test",
        project=str(OUTPUTS / "val_baseline"),
        name="run",
        exist_ok=True,
        verbose=False,
    )
    baseline_metrics = metrics_dict(baseline_res, class_names)

    print("\n--- Evaluating fine-tuned model on test split ---")
    finetuned = YOLO(str(best))
    ft_res = finetuned.val(
        data=str(data_yaml),
        split="test",
        project=str(OUTPUTS / "val_finetuned"),
        name="run",
        exist_ok=True,
        verbose=False,
    )
    ft_metrics = metrics_dict(ft_res, class_names)

    (OUTPUTS / "metrics_baseline.json").write_text(json.dumps(baseline_metrics, indent=2))
    (OUTPUTS / "metrics_finetuned.json").write_text(json.dumps(ft_metrics, indent=2))

    # Comparison table (overall + per-class).
    rows = [
        {
            "scope": "overall",
            "precision_baseline": baseline_metrics["precision"],
            "recall_baseline": baseline_metrics["recall"],
            "mAP50_baseline": baseline_metrics["mAP50"],
            "mAP50-95_baseline": baseline_metrics["mAP50-95"],
            "precision_finetuned": ft_metrics["precision"],
            "recall_finetuned": ft_metrics["recall"],
            "mAP50_finetuned": ft_metrics["mAP50"],
            "mAP50-95_finetuned": ft_metrics["mAP50-95"],
        }
    ]
    for name in class_names:
        b = baseline_metrics["per_class"].get(name, {"mAP50": 0.0, "mAP50-95": 0.0})
        f = ft_metrics["per_class"].get(name, {"mAP50": 0.0, "mAP50-95": 0.0})
        rows.append({
            "scope": name,
            "precision_baseline": None,
            "recall_baseline": None,
            "mAP50_baseline": b["mAP50"],
            "mAP50-95_baseline": b["mAP50-95"],
            "precision_finetuned": None,
            "recall_finetuned": None,
            "mAP50_finetuned": f["mAP50"],
            "mAP50-95_finetuned": f["mAP50-95"],
        })
    df = pd.DataFrame(rows)
    csv_path = OUTPUTS / "metrics_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Bar chart.
    keys = ["precision", "recall", "mAP50", "mAP50-95"]
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    base_vals = [baseline_metrics[k] for k in keys]
    ft_vals = [ft_metrics[k] for k in keys]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, base_vals, width, label="Pre-trained")
    ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs. Fine-tuned YOLO11n (test split)")
    for i, (a, b) in enumerate(zip(base_vals, ft_vals)):
        ax.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    chart_path = OUTPUTS / "comparison" / "metrics_comparison.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {chart_path}")

    # Side-by-side visualization on the same example images used in Step 3.
    examples_file = OUTPUTS / "comparison" / "examples.txt"
    if examples_file.exists():
        example_paths = [Path(p) for p in examples_file.read_text().splitlines() if p.strip()]
        pairs = []
        names = []
        for p in example_paths:
            pairs.append((annotate(baseline, p), annotate(finetuned, p)))
            names.append(p.name)
        out = OUTPUTS / "comparison" / "side_by_side.png"
        plot_side_by_side(pairs, names, out)
        print(f"Wrote {out}")
    else:
        print("Skipped side-by-side: examples.txt missing (run infer_baseline.py first).")

    print("\n=== Summary ===")
    print(f"Baseline  mAP50={baseline_metrics['mAP50']:.4f}  mAP50-95={baseline_metrics['mAP50-95']:.4f}")
    print(f"Finetuned mAP50={ft_metrics['mAP50']:.4f}  mAP50-95={ft_metrics['mAP50-95']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
