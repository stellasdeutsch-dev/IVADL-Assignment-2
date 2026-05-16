"""Shared helpers: paths, example-image picker, grid plotting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "skin-disease"
OUTPUTS = ROOT / "outputs"
RUNS = ROOT / "runs"
RUN_NAME = "skin_yolo11n"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_test_images(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return sorted test images. Falls back to valid/ if test/ is missing."""
    for split in ("test", "valid", "val"):
        d = data_dir / split / "images"
        if d.exists():
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            if imgs:
                return imgs
    raise FileNotFoundError(f"No test/valid images found under {data_dir}")


def pick_examples(images: list[Path], n: int = 6) -> list[Path]:
    """Deterministically pick N images spread across the sorted list."""
    if len(images) <= n:
        return images
    step = len(images) / n
    return [images[int(i * step)] for i in range(n)]


def plot_image_grid(image_arrays: Iterable[np.ndarray], titles: list[str], out_path: Path, cols: int = 3):
    """Save a grid figure of BGR or RGB numpy images."""
    imgs = list(image_arrays)
    n = len(imgs)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.atleast_1d(axes).flatten()
    for ax, img, title in zip(axes, imgs, titles):
        if img.ndim == 3 and img.shape[2] == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_side_by_side(pairs: list[tuple[np.ndarray, np.ndarray]], names: list[str], out_path: Path):
    """One row per pair: left=baseline, right=fine-tuned. Inputs are BGR arrays."""
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 4))
    if rows == 1:
        axes = np.array([axes])
    for i, ((lhs, rhs), name) in enumerate(zip(pairs, names)):
        axes[i, 0].imshow(cv2.cvtColor(lhs, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Pre-trained  ·  {name}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cv2.cvtColor(rhs, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(f"Fine-tuned  ·  {name}", fontsize=9)
        axes[i, 1].axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
