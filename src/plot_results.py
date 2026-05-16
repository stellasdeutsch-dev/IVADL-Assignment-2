"""Generate all result visualizations for the assignment report."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from ultralytics import YOLO

from utils import OUTPUTS, RUNS, RUN_NAME, plot_side_by_side

# ── colour palette ───────────────────────────────────────────────────────────
C_BASE = "#4C72B0"   # steel blue  – baseline
C_FT   = "#DD8452"   # warm orange – fine-tuned
C_ECZ  = "#55A868"   # green       – Eczema
C_MEL  = "#C44E52"   # red         – melanoma-disease
GREY   = "#AAAAAA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ── load metrics ─────────────────────────────────────────────────────────────
def load_metrics():
    with (OUTPUTS / "metrics_baseline.json").open() as f:
        base = json.load(f)
    with (OUTPUTS / "metrics_finetuned.json").open() as f:
        ft = json.load(f)
    return base, ft


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Grouped bar: overall P / R / mAP50 / mAP50-95 (baseline vs fine-tuned)
# ─────────────────────────────────────────────────────────────────────────────
def plot_overall_comparison(base, ft, out: Path):
    keys   = ["precision", "recall", "mAP50", "mAP50-95"]
    labels = ["Precision", "Recall", "mAP50", "mAP50‑95"]
    bv = [base[k] for k in keys]
    fv = [ft[k]   for k in keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_b = ax.bar(x - w/2, bv, w, color=C_BASE, label="Pre-trained (COCO baseline)", zorder=3)
    bars_f = ax.bar(x + w/2, fv, w, color=C_FT,   label="Fine-tuned (transfer learning)", zorder=3)

    for bar, val in zip(list(bars_b) + list(bars_f), bv + fv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Overall Detection Metrics — Baseline vs. Fine-tuned YOLO11n",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Per-class mAP50 grouped bar (4 groups: 2 classes × 2 models)
# ─────────────────────────────────────────────────────────────────────────────
def plot_perclass_map50(base, ft, out: Path):
    classes = ["Eczema", "melanoma-disease"]
    bv = [base["per_class"][c]["mAP50"] for c in classes]
    fv = [ft["per_class"][c]["mAP50"]   for c in classes]
    x  = np.arange(len(classes))
    w  = 0.36

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b = ax.bar(x - w/2, bv, w, color=C_BASE, label="Pre-trained", zorder=3)
    bars_f = ax.bar(x + w/2, fv, w, color=C_FT,   label="Fine-tuned",  zorder=3)

    for bar, val in zip(list(bars_b) + list(bars_f), bv + fv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x, classes, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mAP50", fontsize=11)
    ax.set_title("Per-class mAP50 — Baseline vs. Fine-tuned", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Per-class mAP50-95 grouped bar
# ─────────────────────────────────────────────────────────────────────────────
def plot_perclass_map5095(base, ft, out: Path):
    classes = ["Eczema", "melanoma-disease"]
    bv = [base["per_class"][c]["mAP50-95"] for c in classes]
    fv = [ft["per_class"][c]["mAP50-95"]   for c in classes]
    x  = np.arange(len(classes))
    w  = 0.36

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b = ax.bar(x - w/2, bv, w, color=C_BASE, label="Pre-trained", zorder=3)
    bars_f = ax.bar(x + w/2, fv, w, color=C_FT,   label="Fine-tuned",  zorder=3)

    for bar, val in zip(list(bars_b) + list(bars_f), bv + fv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x, classes, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mAP50‑95", fontsize=11)
    ax.set_title("Per-class mAP50‑95 — Baseline vs. Fine-tuned", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Radar / spider chart (overall metrics)
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar(base, ft, out: Path):
    labels = ["Precision", "Recall", "mAP50", "mAP50‑95"]
    keys   = ["precision", "recall", "mAP50", "mAP50-95"]
    bv = [base[k] for k in keys]
    fv = [ft[k]   for k in keys]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    bv_c = bv + bv[:1]
    fv_c = fv + fv[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, bv_c, "o-", color=C_BASE, linewidth=2, label="Pre-trained")
    ax.fill(angles, bv_c, alpha=0.15, color=C_BASE)
    ax.plot(angles, fv_c, "o-", color=C_FT, linewidth=2, label="Fine-tuned")
    ax.fill(angles, fv_c, alpha=0.15, color=C_FT)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Metrics Radar — Baseline vs. Fine-tuned", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Heatmap: all metrics × {overall, Eczema, melanoma} × {baseline, ft}
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(base, ft, out: Path):
    rows  = ["Overall", "Eczema", "melanoma-disease"]
    cols  = ["Precision\n(baseline)", "Recall\n(baseline)",
             "mAP50\n(baseline)", "mAP50‑95\n(baseline)",
             "Precision\n(fine-tuned)", "Recall\n(fine-tuned)",
             "mAP50\n(fine-tuned)", "mAP50‑95\n(fine-tuned)"]

    def row_vals(m, pc_key=None):
        if pc_key:
            return [None, None,
                    m["per_class"][pc_key]["mAP50"],
                    m["per_class"][pc_key]["mAP50-95"],
                    None, None, None, None]
        return [m["precision"], m["recall"], m["mAP50"], m["mAP50-95"],
                None, None, None, None]

    data = np.full((3, 8), np.nan)
    # overall baseline
    data[0, 0] = base["precision"]
    data[0, 1] = base["recall"]
    data[0, 2] = base["mAP50"]
    data[0, 3] = base["mAP50-95"]
    # overall fine-tuned
    data[0, 4] = ft["precision"]
    data[0, 5] = ft["recall"]
    data[0, 6] = ft["mAP50"]
    data[0, 7] = ft["mAP50-95"]
    # per-class baseline
    for i, cls in enumerate(["Eczema", "melanoma-disease"]):
        data[i+1, 2] = base["per_class"][cls]["mAP50"]
        data[i+1, 3] = base["per_class"][cls]["mAP50-95"]
        data[i+1, 6] = ft["per_class"][cls]["mAP50"]
        data[i+1, 7] = ft["per_class"][cls]["mAP50-95"]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Score", fraction=0.03, pad=0.02)

    ax.set_xticks(range(8), cols, fontsize=9)
    ax.set_yticks(range(3), rows, fontsize=10)
    ax.set_title("Metrics Heatmap (green = high, red = low)", fontsize=12,
                 fontweight="bold", pad=10)

    for i in range(3):
        for j in range(8):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="black" if 0.3 < data[i, j] < 0.8 else "white")

    # vertical separator between baseline / fine-tuned columns
    ax.axvline(3.5, color="white", linewidth=3)
    ax.text(1.5, -0.7, "BASELINE", ha="center", fontsize=9, color=C_BASE,
            fontweight="bold", transform=ax.transData)
    ax.text(5.5, -0.7, "FINE-TUNED", ha="center", fontsize=9, color=C_FT,
            fontweight="bold", transform=ax.transData)

    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Side-by-side detection grid (3 cols: original | baseline | fine-tuned)
# ─────────────────────────────────────────────────────────────────────────────
def plot_detection_comparison(example_paths: list[Path], out: Path):
    baseline = YOLO("yolo11n.pt")
    finetuned = YOLO(str(RUNS / RUN_NAME / "weights" / "best.pt"))

    n = len(example_paths)
    fig = plt.figure(figsize=(15, n * 4))
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.04, wspace=0.02)

    col_titles = ["Original Image", "Pre-trained (COCO baseline)", "Fine-tuned YOLO11n"]
    col_colors = ["#333333", C_BASE, C_FT]

    for col, (title, color) in enumerate(zip(col_titles, col_colors)):
        fig.text((col + 0.5) / 3, 0.995, title,
                 ha="center", va="top", fontsize=13, fontweight="bold", color=color)

    for row, img_path in enumerate(example_paths):
        orig = cv2.imread(str(img_path))
        base_res = baseline.predict(str(img_path), conf=0.25, verbose=False)[0]
        ft_res   = finetuned.predict(str(img_path), conf=0.25, verbose=False)[0]

        n_base = 0 if base_res.boxes is None else len(base_res.boxes)
        n_ft   = 0 if ft_res.boxes   is None else len(ft_res.boxes)

        panels = [
            (cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), img_path.name, "#333333"),
            (cv2.cvtColor(base_res.plot(), cv2.COLOR_BGR2RGB),
             f"{n_base} detection{'s' if n_base!=1 else ''}", C_BASE),
            (cv2.cvtColor(ft_res.plot(), cv2.COLOR_BGR2RGB),
             f"{n_ft} detection{'s' if n_ft!=1 else ''}", C_FT),
        ]

        for col, (img_arr, subtitle, color) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img_arr)
            ax.set_xlabel(subtitle, fontsize=9, color=color, fontweight="bold", labelpad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Improvement delta bar (how much each metric improved)
# ─────────────────────────────────────────────────────────────────────────────
def plot_improvement(base, ft, out: Path):
    categories = {
        "Precision\n(overall)":         ft["precision"]  - base["precision"],
        "Recall\n(overall)":            ft["recall"]     - base["recall"],
        "mAP50\n(overall)":             ft["mAP50"]      - base["mAP50"],
        "mAP50‑95\n(overall)":          ft["mAP50-95"]   - base["mAP50-95"],
        "mAP50\n(Eczema)":              ft["per_class"]["Eczema"]["mAP50"]           - base["per_class"]["Eczema"]["mAP50"],
        "mAP50\n(melanoma-disease)":    ft["per_class"]["melanoma-disease"]["mAP50"] - base["per_class"]["melanoma-disease"]["mAP50"],
        "mAP50‑95\n(Eczema)":           ft["per_class"]["Eczema"]["mAP50-95"]        - base["per_class"]["Eczema"]["mAP50-95"],
        "mAP50‑95\n(melanoma-disease)": ft["per_class"]["melanoma-disease"]["mAP50-95"] - base["per_class"]["melanoma-disease"]["mAP50-95"],
    }

    labels = list(categories.keys())
    deltas = list(categories.values())
    colors = [C_FT if d >= 0 else C_MEL for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), deltas, color=colors, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, deltas):
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.03
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"+{val:.3f}" if val >= 0 else f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylabel("Absolute Improvement (Fine-tuned − Baseline)", fontsize=10)
    ax.set_title("Metric Improvement After Fine-tuning", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_ylim(min(deltas) - 0.08, max(deltas) + 0.08)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    base, ft = load_metrics()
    out = OUTPUTS / "plots"
    out.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_overall_comparison(base, ft,  out / "01_overall_comparison.png")
    plot_perclass_map50(base, ft,      out / "02_perclass_mAP50.png")
    plot_perclass_map5095(base, ft,    out / "03_perclass_mAP50_95.png")
    plot_radar(base, ft,               out / "04_radar_chart.png")
    plot_heatmap(base, ft,             out / "05_metrics_heatmap.png")
    plot_improvement(base, ft,         out / "07_improvement_delta.png")

    examples_file = OUTPUTS / "comparison" / "examples.txt"
    examples = [Path(p) for p in examples_file.read_text().splitlines() if p.strip()]
    plot_detection_comparison(examples, out / "06_detection_comparison.png")

    print(f"\nAll plots saved to {out}/")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
