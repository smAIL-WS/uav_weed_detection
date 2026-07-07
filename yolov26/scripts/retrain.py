"""
retrain.py
────────────────────────────────────────────────────────────────────────────────
Retrains YOLO26 on final_retrain_data using the best HPs from the Optuna study.
Epochs = ceil(mean_epochs across the 4 folds of the best trial).
No validation pass during retraining.

Usage:
  python scripts/retrain.py --variant full_dataset
"""

import argparse
import math
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/pipeline_config.yaml")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert args.variant in cfg["dataset"]["variants"], \
        f"Variant '{args.variant}' not found in config."

    base_dir     = Path(cfg["project"]["base_dir"])
    variant_root = Path(cfg["dataset"]["root"]) / args.variant
    retrain_dir  = variant_root / cfg["dataset"]["retrain_dir"]

    assert retrain_dir.exists(), f"Retrain directory not found: {retrain_dir}"

    # Load best trial
    best_path = base_dir / "results" / args.variant / "optuna" / "best_trial.yaml"
    assert best_path.exists(), \
        f"best_trial.yaml not found. Run train.py first.\n  Expected: {best_path}"

    with open(best_path) as f:
        best = yaml.safe_load(f)

    hps            = best["hps"]
    retrain_epochs = math.ceil(best["mean_epochs"])

    print(f"\n[{args.variant}] Retraining")
    print(f"  Best trial:   #{best['trial_number']}")
    print(f"  Mean mAP50:   {best['mean_mAP50']:.4f}")
    print(f"  Mean epochs:  {best['mean_epochs']:.2f}  →  training for {retrain_epochs} epochs")
    print(f"  HPs:          {hps}")

    # Write data yaml (train only; val key required by Ultralytics but val=False)
    data = {
        "train":      str(retrain_dir / "images"),
        "val":        str(retrain_dir / "images"),  # unused, val=False
        "train_lbls": str(retrain_dir / "labels"),
        "val_lbls":   str(retrain_dir / "labels"),
        "nc":         cfg["dataset"]["nc"],
        "names":      cfg["dataset"]["names"],
    }
    data_yaml_path = base_dir / "results" / args.variant / "retrain_data.yaml"
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    out_dir = base_dir / "results" / args.variant / "retrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(cfg["model"]["weights"])
    model.train(
        data=str(data_yaml_path),
        epochs=retrain_epochs,
        imgsz=cfg["training"]["imgsz"],
        device=cfg["project"]["device"],
        workers=cfg["training"]["workers"],
        patience=cfg["retrain"]["patience"],
        val=cfg["retrain"]["val"],
        batch=hps["batch"],
        weight_decay=hps["weight_decay"],
        warmup_epochs=hps["warmup_epochs"],
        lrf=hps["lrf"],
        project=f"fop_yolo26_{args.variant}_retrain",
        name="final",
        exist_ok=True,
        verbose=True,
        plots=True,
        save_period=-1,
    )

    last_pt = out_dir / "weights" / "last.pt"
    print(f"\n✓ Done.  Checkpoint for inference → {last_pt}")


if __name__ == "__main__":
    main()