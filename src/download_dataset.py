"""Step 1 — Download the skin-disease dataset from Roboflow Universe in YOLO format.

Requires ROBOFLOW_API_KEY in the environment (or in a .env file at the repo root).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils import DATA_DIR, ROOT

WORKSPACE = "skin-diseases-iyitj"
PROJECT = "skin-disease-d6tg8"
FORMAT = "yolov11"


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key or api_key == "your_key_here":
        print(
            "ERROR: ROBOFLOW_API_KEY is not set.\n"
            "1) Get a key at https://app.roboflow.com/settings/api\n"
            "2) cp .env.example .env  and paste the key in\n"
            "3) Re-run this script.",
            file=sys.stderr,
        )
        return 1

    from roboflow import Roboflow  # imported here so missing key fails fast

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    versions = project.versions()
    if not versions:
        print(f"ERROR: project {WORKSPACE}/{PROJECT} has no versions.", file=sys.stderr)
        return 2
    version = versions[-1]  # latest
    print(f"Downloading {WORKSPACE}/{PROJECT} v{version.version} ({FORMAT}) -> {DATA_DIR}")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    version.download(FORMAT, location=str(DATA_DIR))

    yaml_path = DATA_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"ERROR: data.yaml not found at {yaml_path}", file=sys.stderr)
        return 3

    with yaml_path.open() as f:
        meta = yaml.safe_load(f)

    # Roboflow writes paths relative to data.yaml; rewrite them to absolute so
    # Ultralytics works regardless of cwd.
    for key in ("train", "val", "test"):
        if key in meta and meta[key]:
            p = (yaml_path.parent / meta[key]).resolve()
            meta[key] = str(p)
    with yaml_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print("\n=== Dataset summary ===")
    print(f"  classes ({meta.get('nc', '?')}): {meta.get('names')}")
    for split in ("train", "val", "test"):
        p = Path(meta.get(split, "")) if meta.get(split) else None
        if p and p.exists():
            n = len(list(p.iterdir()))
            print(f"  {split}: {n} images at {p}")
    print(f"  data.yaml: {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
