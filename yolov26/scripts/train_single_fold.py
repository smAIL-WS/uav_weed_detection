"""
train_single_fold.py
────────────────────────────────────────────────────────────────────────────────
Called as a subprocess by train.py to run one fold on one GPU.
GPU assignment is done before any imports to ensure CUDA_VISIBLE_DEVICES
is set before torch initializes.

Prints RESULT: <map50> <epochs> to stdout for the parent process to parse.

Usage (called internally by train.py):
  python scripts/train_single_fold.py --cfg /tmp/fold_cfg_xxx.yaml --gpu 2
"""

import argparse
import os
import sys

# ── GPU assignment MUST happen before any torch/ultralytics imports ───────────
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", required=True, help="Path to fold config yaml")
parser.add_argument("--gpu", required=True, type=int, help="GPU index to use")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import torch
assert torch.cuda.device_count() == 1, \
    f"Expected 1 GPU, got {torch.cuda.device_count()}"
print(f"[fold gpu={args.gpu}] Using: {torch.cuda.get_device_name(0)}", flush=True)

# ── All other imports after GPU assignment ────────────────────────────────────
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    with open(args.cfg) as f:
        fold_cfg = yaml.safe_load(f)

    data_yaml = fold_cfg["data_yaml"]
    hps       = fold_cfg["hps"]
    cfg       = fold_cfg["cfg"]
    project   = fold_cfg["project"]
    run_name  = fold_cfg["run_name"]

    model = YOLO(cfg["model"]["weights"])

    results = model.train(
        data=data_yaml,
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["imgsz"],
        device=0,              # always 0 — CUDA_VISIBLE_DEVICES remaps to correct physical GPU
        workers=cfg["training"]["workers"],
        patience=cfg["training"]["patience"],
        batch=hps["batch"],
        weight_decay=hps["weight_decay"],
        warmup_epochs=hps["warmup_epochs"],
        lrf=hps["lrf"],
        project=project,
        name=run_name,
        exist_ok=True,
        verbose=False,
        plots=False,
        save_period=-1,
    )

    map50 = float(results.results_dict.get(cfg["optuna"]["metric"], 0.0))

    # Read actual epochs completed from results CSV
    csv_path = Path(project) / run_name / "results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            epochs = sum(1 for _ in csv.DictReader(f))
    else:
        epochs = cfg["training"]["epochs"]

    # Print result for parent process to parse
    print(f"RESULT: {map50} {epochs}", flush=True)


if __name__ == "__main__":
    main()