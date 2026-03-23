import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


# ── User Configuration ─────────────────────────────────────────────────────────
IMAGE_ROOT   = "uav_weed_detection/raw_data/train/images/"
ANNO_ROOT    = "uav_weed_detection/raw_data/train/annotations/"
OUTPUT_ROOT  = "uav_weed_detection/sampled_data/"
SEED         = 50

# Choose which variant to run: "half", "quarter", or "single"
VARIANT = "half"

# CV fold definitions — each fold trains on 3 partfields, validates on the 4th
CV_FOLDS = {
    "fold1": {"train": {"A", "B", "C"}, "val": {"D"}},
    "fold2": {"train": {"A", "B", "D"}, "val": {"C"}},
    "fold3": {"train": {"A", "C", "D"}, "val": {"B"}},
    "fold4": {"train": {"B", "C", "D"}, "val": {"A"}},
}
# ──────────────────────────────────────────────────────────────────────────────


def get_partfield(filename):
    """Extract partfield letter (A, B, C, D) from image filename."""
    return Path(filename).stem.split('_')[2][0]


def sample_images(image_root, variant, seed=SEED):
    """
    Sample images from the dataset according to the chosen variant.

    Sampling rules:
    - half   : ~50% of images per growth stage subfolder
    - quarter: ~25% of images per growth stage subfolder
    - single : exactly 1 image per growth stage subfolder

    Partfield constraint (across ALL growth stages combined):
    - At least 1 image per partfield (A, B, C, D)
    - Max 2 images per partfield total

    Returns:
        List of selected image Paths
    """
    random.seed(seed)
    image_root = Path(image_root)
    subfolders = sorted([f for f in image_root.iterdir() if f.is_dir()])

    # Collect all images grouped by subfolder
    all_images        = []
    folder_to_images  = {}
    for subfolder in subfolders:
        images = list(subfolder.glob("*.png")) + \
                 list(subfolder.glob("*.jpg")) + \
                 list(subfolder.glob("*.jpeg"))
        folder_to_images[subfolder.name] = images
        all_images.extend(images)

    total_images = len(all_images)

    # Determine target sample size based on variant
    if variant == "half":
        target = max(len(subfolders), total_images // 2)
    elif variant == "quarter":
        target = max(len(subfolders), total_images // 4)
    elif variant == "single":
        target = len(subfolders)   # exactly 1 per subfolder
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: half, quarter, single")

    # ── Step 1 — Guarantee at least 1 image per subfolder ─────────────────────
    selected        = []
    selected_names  = set()
    partfield_counts = defaultdict(int)   # tracks count across ALL subfolders

    for subfolder_name, images in folder_to_images.items():
        if not images:
            continue
        random.shuffle(images)

        # Pick one image from this subfolder respecting partfield max=2 constraint
        for img in images:
            pf = get_partfield(img.name)
            if partfield_counts[pf] < 2:
                selected.append(img)
                selected_names.add(img.name)
                partfield_counts[pf] += 1
                break

    if variant == "single":
        # Single mode — one image per subfolder is sufficient, stop here
        return selected

    # ── Step 2 — Fill remaining quota randomly from leftover pool ─────────────
    remaining_needed = target - len(selected)
    remaining_pool   = [img for img in all_images if img.name not in selected_names]

    if remaining_needed > 0 and remaining_pool:
        additional = random.sample(
            remaining_pool,
            min(remaining_needed, len(remaining_pool))
        )
        selected.extend(additional)

    return selected


def copy_images(selected_images, anno_root, output_root):
    """
    Copy selected images and their XML annotations to output_root,
    preserving the original subfolder structure.

    Structure:
        output_root/
        ├── images/train/BBCH_12/
        └── annotations/train/BBCH_12/
    """
    anno_root   = Path(anno_root)
    output_root = Path(output_root)
    missing     = []

    for img_path in selected_images:
        subfolder  = img_path.parent.name
        anno_path  = anno_root / subfolder / img_path.with_suffix(".xml").name

        dest_img   = output_root / "images"      / "train" / subfolder
        dest_anno  = output_root / "annotations" / "train" / subfolder
        dest_img.mkdir(parents=True, exist_ok=True)
        dest_anno.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, dest_img / img_path.name)
        if anno_path.exists():
            shutil.copy(anno_path, dest_anno / anno_path.name)
        else:
            missing.append(img_path.name)

    if missing:
        print(f"[Warning] {len(missing)} missing annotation files:")
        for name in missing:
            print(f"  - {name}")

    print(f"Copied {len(selected_images)} images to: {output_root}")


def copy_fold(selected_images, anno_root, fold_root, partfields):
    """
    Copy selected images into train/ and val/ folders based on
    partfield assignment for a given CV fold.

    Val and test folders receive the same held-out partfield images
    to maintain training pipeline compatibility.
    """
    train_images = [img for img in selected_images
                    if get_partfield(img.name) in partfields["train"]]
    val_images   = [img for img in selected_images
                    if get_partfield(img.name) in partfields["val"]]

    # Train split
    copy_images(
        selected_images=train_images,
        anno_root=anno_root,
        output_root=os.path.join(fold_root, "train/")
    )

    # Val and test — same held-out partfield for pipeline compatibility
    for split in ["val", "test"]:
        copy_images(
            selected_images=val_images,
            anno_root=anno_root,
            output_root=os.path.join(fold_root, f"{split}/")
        )


# ── Main ───────────────────────────────────────────────────────────────────────

print(f"\nRunning variant: {VARIANT}")
print(f"{'='*50}")

# Step 1 — Sample images according to chosen variant
selected_images = sample_images(
    image_root=IMAGE_ROOT,
    variant=VARIANT,
    seed=SEED
)
print(f"Total sampled images: {len(selected_images)}")

# Print partfield distribution for verification
partfield_dist = defaultdict(int)
for img in selected_images:
    partfield_dist[get_partfield(img.name)] += 1
print(f"Partfield distribution: {dict(sorted(partfield_dist.items()))}")

# Step 2 — Copy full sampled dataset (all partfields together)
full_output = os.path.join(OUTPUT_ROOT, VARIANT, "full/")
copy_images(
    selected_images=selected_images,
    anno_root=ANNO_ROOT,
    output_root=full_output
)

# Step 3 — Create CV folds from sampled images
for fold_name, partfields in CV_FOLDS.items():
    print(f"\n  {fold_name} — "
          f"train: {partfields['train']} | val: {partfields['val']}")

    fold_root = os.path.join(OUTPUT_ROOT, VARIANT, fold_name)
    copy_fold(
        selected_images=selected_images,
        anno_root=ANNO_ROOT,
        fold_root=fold_root,
        partfields=partfields
    )

print(f"\nDone. Output saved to: {os.path.join(OUTPUT_ROOT, VARIANT)}")
