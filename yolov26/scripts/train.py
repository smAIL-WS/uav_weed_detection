"""
train.py
────────────────────────────────────────────────────────────────────────────────
Runs Optuna HP optimisation with 4-fold cross validation for a given variant.

For each trial:
  - One HP combination is sampled
  - All 4 folds are trained sequentially
  - Objective = mean mAP50 across the 4 folds

Results saved to:
  results/<variant>/optuna/
      study.db          Optuna SQLite store — resumable if interrupted
      summary.csv       All trials: HPs, per-fold mAP50, mean mAP50, mean epochs
      best_trial.yaml   Best HP combination and mean epoch count

Usage:
  python scripts/train.py --variant full_dataset
  python scripts/train.py --variant full_dataset --config configs/pipeline_config.yaml
"""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import optuna
import yaml
from ultralytics import YOLO

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data yaml (written to a temp file per fold per trial) ─────────────────────

def make_data_yaml(variant_root, fold, nc, names):
    fold_dir = variant_root / fold
    train_img_dir = fold_dir / "images" / "train"
    val_img_dir   = fold_dir / "images" / "val"

    # Write image path list files
    train_txt = fold_dir / "train_paths.txt"
    val_txt   = fold_dir / "val_paths.txt"

    with open(train_txt, "w") as f:
        for p in sorted(train_img_dir.glob("*.png")) + sorted(train_img_dir.glob("*.jpg")):
            f.write(str(p.resolve()) + "\n")

    with open(val_txt, "w") as f:
        for p in sorted(val_img_dir.glob("*.png")) + sorted(val_img_dir.glob("*.jpg")):
            f.write(str(p.resolve()) + "\n")

    data = {
        "train": str(train_txt),
        "val":   str(val_txt),
        "nc":    nc,
        "names": names,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"yolo_{fold}_"
    )
    yaml.dump(data, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


# ── HP suggestion ─────────────────────────────────────────────────────────────

def suggest_hps(trial: optuna.Trial, search_space: dict) -> dict:
    hps = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            hps[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            hps[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            hps[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown HP type '{t}' for parameter '{name}'")
    return hps


# ── Single fold training ──────────────────────────────────────────────────────

def train_fold(
    data_yaml:    str,
    hps:          dict,
    cfg:          dict,
    project:      str,
    run_name:     str,
) -> tuple[float, int]:
    """
    Trains one fold. Returns (mAP50, epochs_completed).
    Ultralytics writes outputs to <project>/<run_name>/
    """
    model = YOLO(cfg["model"]["weights"])

    results = model.train(
        data=data_yaml,
        epochs=cfg["training"]["epochs"],
        imgsz=cfg["training"]["imgsz"],
        device=cfg["project"]["device"],
        workers=cfg["training"]["workers"],
        patience=cfg["training"]["patience"],
        batch=hps["batch"],
        weight_decay=hps["weight_decay"],
        warmup_epochs=hps["warmup_epochs"],
        lrf=hps["lrf"],
        project=project,
        name=run_name,
        exist_ok=True,
        verbose=cfg["training"]["verbose"],
        plots=cfg["training"]["plots"],
        save=False,
        save_period=-1,
    )

    map50 = float(results.results_dict.get(cfg["optuna"]["metric"], 0.0))

    # Actual epochs run — read from results CSV (most reliable)
    csv_path = Path(project) / run_name / "results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            epochs_completed = sum(1 for _ in csv.DictReader(f))
    else:
        epochs_completed = cfg["training"]["epochs"]

    return map50, epochs_completed


# ── Optuna objective ──────────────────────────────────────────────────────────

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def train_fold_subprocess(args):
    data_yaml, hps, cfg, project, run_name, gpu_id = args

    fold_cfg = {
        "data_yaml": data_yaml,
        "hps":       hps,
        "cfg":       cfg,
        "project":   project,
        "run_name":  run_name,
    }
    cfg_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="fold_cfg_"
    )
    yaml.dump(fold_cfg, cfg_tmp)
    cfg_tmp.close()

    # Build a clean environment with only CUDA_VISIBLE_DEVICES set
    env = {}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PATH"] = os.environ.get("PATH", "")
    env["HOME"] = os.environ.get("HOME", "")
    env["CONDA_PREFIX"] = os.environ.get("CONDA_PREFIX", "")
    env["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [sys.executable, "scripts/train_single_fold.py",
         "--cfg", cfg_tmp.name,
         "--gpu", str(gpu_id)],
        env=env,
        capture_output=True,
        text=True,
    )
    os.unlink(cfg_tmp.name)

    if result.returncode != 0:
        print(f"  FAILED (GPU {gpu_id}): {result.stderr[-500:]}")
        return 0.0, cfg["training"]["epochs"]

    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT:"):
            parts = line.split()
            return float(parts[1]), int(parts[2])
    return 0.0, cfg["training"]["epochs"]


def make_objective(cfg, variant_root, results_dir, variant):
    fold_names = cfg["dataset"]["folds"]
    print(f"  Folds loaded: {fold_names}") 
    nc         = cfg["dataset"]["nc"]
    names      = cfg["dataset"]["names"]
    n_gpus     = 4
    n_folds     = len(fold_names)
    max_workers = n_folds   # 5 — fold5 queues on GPU 0 after fold1 starts

    def objective(trial):
        hps = suggest_hps(trial, cfg["optuna"]["search_space"])
        print(f"\n[Trial {trial.number:02d}] {hps}")

        fold_args = []
        for i, fold in enumerate(fold_names):
            data_yaml = make_data_yaml(variant_root, fold, nc, names)
            project   = str(results_dir / f"trial_{trial.number:02d}")
            run_name  = fold
            gpu_id    = i % n_gpus
            fold_args.append((data_yaml, hps, cfg, project, run_name, gpu_id))

        # Run all 4 folds in parallel, one per GPU
        fold_maps, fold_epochs = [], []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(train_fold_subprocess, a): a[4] for a in fold_args}
            for future in as_completed(futures):
                fold_name = futures[future]
                try:
                    map50, epochs = future.result()
                    fold_maps.append(map50)
                    fold_epochs.append(epochs)
                    print(f"  {fold_name}: mAP50={map50:.4f}  epochs={epochs}")
                except Exception as e:
                    print(f"  {fold_name}: FAILED ({e})")
                    return 0.0

        mean_map    = sum(fold_maps) / len(fold_maps)
        mean_epochs = sum(fold_epochs) / len(fold_epochs)

        trial.set_user_attr("fold_maps",    fold_maps)
        trial.set_user_attr("fold_epochs",  fold_epochs)
        trial.set_user_attr("mean_map",     mean_map)
        trial.set_user_attr("mean_epochs",  mean_epochs)

        print(f"  → mean mAP50={mean_map:.4f}  mean_epochs={mean_epochs:.1f}")
        return mean_map

    return objective


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_summary(study: optuna.Study, path: Path, fold_names: list[str]) -> None:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number}
        row.update(t.params)
        fold_maps   = t.user_attrs.get("fold_maps",   [None] * len(fold_names))
        fold_epochs = t.user_attrs.get("fold_epochs", [None] * len(fold_names))
        for fold, m, e in zip(fold_names, fold_maps, fold_epochs):
            row[f"{fold}_mAP50"]  = m
            row[f"{fold}_epochs"] = e
        row["mean_mAP50"]  = t.user_attrs.get("mean_map",    t.value)
        row["mean_epochs"] = t.user_attrs.get("mean_epochs")
        rows.append(row)

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Summary  → {path}")


def save_best(study: optuna.Study, path: Path) -> None:
    bt = study.best_trial
    out = {
        "trial_number": bt.number,
        "mean_mAP50":   bt.user_attrs.get("mean_map",    bt.value),
        "mean_epochs":  bt.user_attrs.get("mean_epochs"),
        "fold_maps":    bt.user_attrs.get("fold_maps"),
        "fold_epochs":  bt.user_attrs.get("fold_epochs"),
        "hps":          bt.params,
    }
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False)
    print(f"  Best trial → {path}")
    print(f"  HPs:          {bt.params}")
    print(f"  mean mAP50:   {out['mean_mAP50']:.4f}")
    print(f"  mean epochs:  {out['mean_epochs']:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/pipeline_config.yaml")
    parser.add_argument("--variant", required=True,
                        help="Dataset variant folder name, e.g. full_dataset")
    args = parser.parse_args()

    cfg = load_config(args.config)
    assert args.variant in cfg["dataset"]["variants"], \
        f"Variant '{args.variant}' not in config. Available: {cfg['dataset']['variants']}"

    base_dir     = Path(cfg["project"]["base_dir"])
    variant_root = Path(cfg["dataset"]["root"]) / args.variant
    assert variant_root.exists(), f"Variant directory not found: {variant_root}"

    results_dir = base_dir / "results" / args.variant / "optuna"
    results_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{results_dir / 'study.db'}"
    sampler = (
        optuna.samplers.TPESampler(seed=cfg["optuna"]["seed"])
        if cfg["optuna"]["sampler"] == "TPE"
        else optuna.samplers.RandomSampler(seed=cfg["optuna"]["seed"])
    )
    study = optuna.create_study(
        study_name=f"{args.variant}_yolo26",
        storage=storage,
        direction=cfg["optuna"]["direction"],
        sampler=sampler,
        load_if_exists=True,
    )

    n_done      = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = cfg["optuna"]["n_trials"] - n_done
    print(f"\n[{args.variant}]  {n_done}/{cfg['optuna']['n_trials']} trials done."
          f"  Running {n_remaining} more.")

    if n_remaining > 0:
        study.optimize(
            make_objective(cfg, variant_root, results_dir, args.variant),
            n_trials=n_remaining,
        )

    fold_names = cfg["dataset"]["folds"]
    save_summary(study, results_dir / "summary.csv", fold_names)
    save_best(study,    results_dir / "best_trial.yaml")


if __name__ == "__main__":
    main()