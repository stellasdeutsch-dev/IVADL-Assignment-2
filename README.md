# Assignment 2 — Object Detection with YOLO11n

End-to-end YOLO pipeline: pre-trained baseline → fine-tune on a Roboflow Universe skin-disease dataset → compare.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Roboflow API key (https://app.roboflow.com/settings/api)
cp .env.example .env
# edit .env and paste your key
```

Verify Apple-MPS is visible:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Run the five steps

```bash
# Step 1 — download dataset (skin-diseases-iyitj/skin-disease-d6tg8)
python src/download_dataset.py

# Step 3 — baseline inference (pre-trained YOLO11n on COCO)
python src/infer_baseline.py

# Step 4 — fine-tune (transfer learning)
python src/train_finetune.py --epochs 50 --imgsz 640 --batch 16

# Step 5 — evaluate & compare
python src/evaluate.py
```

Or run the whole walkthrough as a notebook:
```bash
jupyter notebook notebook/assignment2.ipynb
```

## Outputs

| Path | Contents |
| --- | --- |
| `runs/skin_yolo11n/` | training curves (`results.png`), best/last weights |
| `outputs/baseline_preds/predict/` | pre-trained model's predictions on every test image |
| `outputs/comparison/baseline_grid.png` | 6 example images with baseline detections |
| `outputs/comparison/side_by_side.png` | same 6 images, baseline vs. fine-tuned |
| `outputs/comparison/metrics_comparison.png` | precision / recall / mAP bar chart |
| `outputs/metrics_baseline.json`, `metrics_finetuned.json` | raw numbers |
| `outputs/metrics_comparison.csv` | overall + per-class table |
| `report/report.pdf` | the 3–4 page deliverable |

## Hardware used

MacBook Air M3 (8-core CPU, 16 GB unified memory), device=`mps`.
