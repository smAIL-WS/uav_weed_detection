import os
import cv2
import json
import random
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from glob import glob


# ── User Configuration ─────────────────────────────────────────────────────────
TRAIN_IMAGE_ROOT = "uav_weed_detection/raw_data/train/images/"
TRAIN_ANNO_ROOT  = "uav_weed_detection/raw_data/train/annotations/"
TEST_IMAGE_ROOT  = "uav_weed_detection/raw_data/test/images/"
TEST_ANNO_ROOT   = "uav_weed_detection/raw_data/test/annotations/"
OUTPUT_ROOT      = "uav_weed_detection/ewis_data/"
CROP_SIZE        = 512
SEED             = 50
N_FOLDS          = 5

# Folder of each fold below OUTPUT_ROOT. {pgsr} = 1..6 (PGSR index), {fold} = 1..N_FOLDS.
#   "PGSR_{pgsr}_fold{fold}"   original layout of this script
#   "exp_{pgsr}/fold{fold}"    layout of your current data folders
FOLD_DIR_PATTERN = "PGSR_{pgsr}_fold{fold}"

# What goes into test_images/ of each fold:
#   "val"       a copy of the fold's validation patches (as in your current data)
#   "separate"  the held-out images from TEST_IMAGE_ROOT / TEST_ANNO_ROOT
TEST_SOURCE = "val"

# Balance the folds by growth stage: every fold validates on a similar share of
# every stage. False = plain random K-fold over all images.
STRATIFY_BY_STAGE = True

# ── Patch options (defaults = behaviour of the original script) ───────────────
# Per-patch min-max contrast stretch to 0..255 (original: True). Each patch then
# gets its own brightness/contrast; set False to save the pixels unchanged.
NORMALIZE_PATCHES = True
# Minimum share of a box's area that must lie inside the patch to keep the box
# (original: 0.0 = keep any overlap, including 1-pixel slivers).
MIN_VISIBLE = 0.0

# Progressive Growth Stage Reduction (PGSR):
# Ordered from lowest to highest BBCH stage.
# Each PGSR experiment removes the highest stage from the previous one:
#   PGSR_1 = all stages (BBCH_12 → BBCH_17)
#   PGSR_2 = BBCH_12 → BBCH_16  (BBCH_17 removed)
#   PGSR_3 = BBCH_12 → BBCH_15  (BBCH_17, BBCH_16 removed)
#   PGSR_4 = BBCH_12 → BBCH_14
#   PGSR_5 = BBCH_12 → BBCH_13
#   PGSR_6 = BBCH_12 only       (all higher stages removed)
GROWTH_STAGES = [
    "BBCH_12",
    "BBCH_13",
    "BBCH_14",
    "BBCH_15",
    "BBCH_16",
    "BBCH_17",
]
# ──────────────────────────────────────────────────────────────────────────────

CATEGORIES = [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"}]
UNKNOWN_LABELS = Counter()   # labels other than crop/weed (mapped to weed, as before)


def parse_xml_annotations(xml_path):
    """Parse Pascal VOC XML annotation file and return list of annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        category = obj.find('name').text
        if category not in ("crop", "weed"):
            UNKNOWN_LABELS[category] += 1     # reported at the end
        category_id = 1 if category == "crop" else 2
        bbox  = obj.find('bndbox')
        x_min = int(float(bbox.find('xmin').text))
        y_min = int(float(bbox.find('ymin').text))
        x_max = int(float(bbox.find('xmax').text))
        y_max = int(float(bbox.find('ymax').text))
        annotations.append({
            "category_id": category_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min)
        })
    return annotations


def generate_patches(img, img_path, xml_path, crop_size):
    """
    Generate all patches and their annotations from a single image.
    Returns list of (patch_array, patch_name, patch_annotations) tuples.
    """
    h, w, d          = img.shape
    orig_name        = Path(img_path).stem
    orig_annotations = parse_xml_annotations(xml_path)
    patches          = []

    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):

            actual_crop       = img[i:min(i + crop_size, h), j:min(j + crop_size, w), :]
            curr_h, curr_w, _ = actual_crop.shape
            img_padded        = np.zeros((crop_size, crop_size, d), dtype=img.dtype)
            img_padded[0:curr_h, 0:curr_w, :] = actual_crop

            if NORMALIZE_PATCHES:
                denom     = np.max(img_padded) - np.min(img_padded)
                img_final = (255 * (img_padded - np.min(img_padded)) / denom).astype(np.uint8) \
                            if denom > 0 else img_padded.astype(np.uint8)
            else:
                img_final = img_padded

            savename    = f"{orig_name}_patch_{i // crop_size}_{j // crop_size}.png"
            patch_annos = []

            for ann in orig_annotations:
                x_min, y_min, box_w, box_h = ann["bbox"]
                x_max, y_max = x_min + box_w, y_min + box_h

                if x_min < (j + crop_size) and x_max > j and \
                   y_min < (i + crop_size) and y_max > i:

                    new_x_min = max(0, x_min - j)
                    new_y_min = max(0, y_min - i)
                    new_x_max = min(crop_size, min(x_max - j, curr_w))
                    new_y_max = min(crop_size, min(y_max - i, curr_h))
                    new_w     = new_x_max - new_x_min
                    new_h     = new_y_max - new_y_min

                    visible_ok = ann["area"] <= 0 or (new_w * new_h) / ann["area"] >= MIN_VISIBLE
                    if new_w > 0 and new_h > 0 and visible_ok:
                        patch_annos.append({
                            "category_id": ann["category_id"],
                            "bbox":        [new_x_min, new_y_min, new_w, new_h],
                            "area":        float(new_w * new_h),
                            "iscrowd":     0
                        })

            patches.append((img_final, savename, patch_annos))

    return patches


def process_images(image_paths, anno_root, dst_folder, crop_size, output_json):
    """Cut the given original images into patches, save them, and write a COCO JSON.

    Patches are written image by image, so not all patches are held in memory.
    Each COCO image entry records its original image ("source_image").
    """
    os.makedirs(dst_folder, exist_ok=True)
    coco_data     = {"images": [], "annotations": [], "categories": CATEGORIES}
    annotation_id = 1
    image_id      = 1
    for img_path in image_paths:
        xml_path = get_xml_path(img_path, anno_root)
        img      = cv2.imread(str(img_path))
        if img is None:
            print(f"    [Warning] Cannot read image, skipped: {img_path}")
            continue
        for img_array, savename, patch_annos in generate_patches(img, str(img_path), xml_path, crop_size):
            cv2.imwrite(os.path.join(dst_folder, savename), img_array)
            coco_data["images"].append({
                "id": image_id, "file_name": savename,
                "width": img_array.shape[1], "height": img_array.shape[0],
                "source_image": Path(img_path).name,
                "group": Path(img_path).parent.name,
            })
            for ann in patch_annos:
                ann_entry = {"id": annotation_id, "image_id": image_id}
                ann_entry.update(ann)
                coco_data["annotations"].append(ann_entry)
                annotation_id += 1
            image_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"    Saved {image_id - 1} patches from {len(image_paths)} images to {output_json}")


def save_annotation_txt(image_folder, output_txt):
    """Save image filenames (without extension) to a txt file."""
    img_paths = sorted(glob(os.path.join(image_folder, "*.png")))
    filenames = [os.path.splitext(os.path.basename(p))[0] + "\n" for p in img_paths]
    with open(output_txt, 'w') as f:
        f.writelines(filenames)
    print(f"    Annotation txt saved at {output_txt}")


def get_xml_path(img_path, anno_root):
    """Get corresponding XML annotation path for an image."""
    img_path  = Path(img_path)
    subfolder = img_path.parent.name
    return os.path.join(anno_root, subfolder, img_path.stem + ".xml")


def get_images_for_stages(image_root, stages):
    """Collect all image paths from the specified growth stage subfolders."""
    img_paths = []
    for stage in stages:
        stage_folder = os.path.join(image_root, stage)
        if os.path.exists(stage_folder):
            imgs = sorted(
                glob(os.path.join(stage_folder, "*.png")) +
                glob(os.path.join(stage_folder, "*.jpg")) +
                glob(os.path.join(stage_folder, "*.jpeg"))
            )
            img_paths.extend(imgs)
        else:
            print(f"[Warning] Growth stage folder not found: {stage_folder}")
    return img_paths


def kfold_by_image(img_paths, n_folds, seed, stratify):
    """
    Split ORIGINAL images (not patches) into n_folds groups.
    Every image is in exactly one validation fold, so all patches of an image
    stay together. With stratify=True, images of each growth stage are spread
    evenly over the folds.
    """
    rng    = random.Random(seed)
    groups = defaultdict(list)
    for p in img_paths:
        groups[Path(p).parent.name if stratify else "all"].append(p)

    folds = [[] for _ in range(n_folds)]
    for stage in sorted(groups):
        imgs = sorted(groups[stage])
        rng.shuffle(imgs)
        if len(imgs) < n_folds:
            print(f"  [Warning] {stage}: only {len(imgs)} images for {n_folds} folds; "
                  f"some folds will have none of this stage in validation.")
        stage_count = [0] * n_folds
        for p in imgs:
            # fold with the fewest images of this stage; ties -> smallest fold overall
            f = min(range(n_folds), key=lambda k: (stage_count[k], len(folds[k]), k))
            folds[f].append(p)
            stage_count[f] += 1
    return folds


# ── Main ───────────────────────────────────────────────────────────────────────

if TEST_SOURCE not in ("val", "separate"):
    raise ValueError("TEST_SOURCE must be 'val' or 'separate'")
if TEST_SOURCE == "separate":
    test_img_paths = sorted(glob(os.path.join(TEST_IMAGE_ROOT, "**/*.png"), recursive=True))

n_experiments = len(GROWTH_STAGES)   # 6 PGSR experiments

for pgsr_idx in range(n_experiments):
    pgsr_name = f"PGSR_{pgsr_idx + 1}"

    # PGSR_1: all stages (BBCH_12 → BBCH_17) ... PGSR_6: BBCH_12 only
    n_stages      = n_experiments - pgsr_idx
    active_stages = GROWTH_STAGES[:n_stages]

    print(f"\n{'='*60}")
    print(f"Processing {pgsr_name} — stages: {active_stages}")
    print(f"{'='*60}")

    all_img_paths = get_images_for_stages(TRAIN_IMAGE_ROOT, active_stages)
    if not all_img_paths:
        print(f"[Warning] No images found for {pgsr_name}, skipping.")
        continue
    print(f"Total original images: {len(all_img_paths)}")
    if len(all_img_paths) < N_FOLDS:
        raise ValueError(f"{pgsr_name}: only {len(all_img_paths)} original images for {N_FOLDS} folds; "
                         f"some validation folds would be empty. Reduce N_FOLDS.")

    # Split ORIGINAL IMAGES into folds first; patches are cut afterwards.
    folds = kfold_by_image(all_img_paths, N_FOLDS, SEED, STRATIFY_BY_STAGE)

    for fold_idx in range(N_FOLDS):
        fold_name = f"fold{fold_idx + 1}"
        fold_root = os.path.join(OUTPUT_ROOT, FOLD_DIR_PATTERN.format(pgsr=pgsr_idx + 1, fold=fold_idx + 1))

        val_imgs   = sorted(folds[fold_idx])
        train_imgs = sorted(p for f, imgs in enumerate(folds) if f != fold_idx for p in imgs)
        assert not set(val_imgs) & set(train_imgs)

        stages_val = Counter(Path(p).parent.name for p in val_imgs)
        print(f"\n  {pgsr_name} — {fold_name}")
        print(f"    train images: {len(train_imgs)} | val images: {len(val_imgs)} "
              f"(per stage: {dict(sorted(stages_val.items()))})")

        process_images(train_imgs, TRAIN_ANNO_ROOT, os.path.join(fold_root, "train_images/"),
                       CROP_SIZE, os.path.join(fold_root, "train.json"))
        process_images(val_imgs, TRAIN_ANNO_ROOT, os.path.join(fold_root, "val_images/"),
                       CROP_SIZE, os.path.join(fold_root, "val.json"))

        if TEST_SOURCE == "val":
            # Test = copy of this fold's validation patches (placeholder; real test set is separate)
            shutil.copytree(os.path.join(fold_root, "val_images/"), os.path.join(fold_root, "test_images/"),
                            dirs_exist_ok=True)
            shutil.copyfile(os.path.join(fold_root, "val.json"), os.path.join(fold_root, "test.json"))
            print(f"    test = copy of val ({os.path.join(fold_root, 'test.json')})")
        else:
            process_images(test_img_paths, TEST_ANNO_ROOT, os.path.join(fold_root, "test_images/"),
                           CROP_SIZE, os.path.join(fold_root, "test.json"))

        for split in ["train", "val", "test"]:
            save_annotation_txt(
                image_folder=os.path.join(fold_root, f"{split}_images/"),
                output_txt=os.path.join(fold_root, f"{split}.txt")
            )

if UNKNOWN_LABELS:
    print(f"\n[Warning] Labels other than crop/weed were mapped to weed: {dict(UNKNOWN_LABELS)}")
print("\nAll PGSR experiments processed successfully.")
