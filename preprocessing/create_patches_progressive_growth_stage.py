import os
import cv2
import json
import random
import numpy as np
import xml.etree.ElementTree as ET
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
VAL_RATIO        = 0.2

# Progressive growth stages — each PGS experiment adds the next stage cumulatively
# PGS_1 = BBCH_12 only
# PGS_2 = BBCH_12 + BBCH_13
# PGS_3 = BBCH_12 + BBCH_13 + BBCH_14
# ... and so on
GROWTH_STAGES = [
    "BBCH_12",
    "BBCH_13",
    "BBCH_14",
    "BBCH_15",
    "BBCH_16",
    "BBCH_17",
]
# ──────────────────────────────────────────────────────────────────────────────


def parse_xml_annotations(xml_path):
    """Parse Pascal VOC XML annotation file and return list of annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        category    = obj.find('name').text
        category_id = 1 if category == "crop" else 2
        bbox        = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
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
    No file I/O — patches are held in memory for splitting first.
    """
    h, w, d          = img.shape
    orig_name        = os.path.basename(img_path).replace('.png', '')
    orig_annotations = parse_xml_annotations(xml_path)
    patches          = []

    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):

            actual_crop       = img[i:min(i + crop_size, h), j:min(j + crop_size, w), :]
            curr_h, curr_w, _ = actual_crop.shape
            img_padded        = np.zeros((crop_size, crop_size, d), dtype=img.dtype)
            img_padded[0:curr_h, 0:curr_w, :] = actual_crop

            denom     = np.max(img_padded) - np.min(img_padded)
            img_final = (255 * (img_padded - np.min(img_padded)) / denom).astype(np.uint8) \
                        if denom > 0 else img_padded.astype(np.uint8)

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

                    if new_w > 0 and new_h > 0:
                        patch_annos.append({
                            "category_id": ann["category_id"],
                            "bbox":        [new_x_min, new_y_min, new_w, new_h],
                            "area":        float(new_w * new_h),
                            "iscrowd":     0
                        })

            patches.append((img_final, savename, patch_annos))

    return patches


def save_patches(patches, dst_folder, output_json):
    """Save a list of patches to disk and write COCO-format annotation JSON."""
    os.makedirs(dst_folder, exist_ok=True)
    coco_data = {
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"}]
    }
    annotation_id = 1
    for image_id, (img_array, savename, patch_annos) in enumerate(patches, start=1):
        cv2.imwrite(os.path.join(dst_folder, savename), img_array)
        coco_data["images"].append({
            "id": image_id, "file_name": savename,
            "width": img_array.shape[1], "height": img_array.shape[0]
        })
        for ann in patch_annos:
            ann_entry = {"id": annotation_id, "image_id": image_id}
            ann_entry.update(ann)
            coco_data["annotations"].append(ann_entry)
            annotation_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"    Saved {len(patches)} patches to {output_json}")


def process_images_direct(image_paths, xml_paths, dst_folder, crop_size, output_json):
    """
    Process images directly to disk without in-memory splitting.
    Used for the test set which does not need splitting.
    """
    os.makedirs(dst_folder, exist_ok=True)
    coco_data = {
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"}]
    }
    annotation_id = 1
    image_id      = 1
    for img_path, xml_path in zip(image_paths, xml_paths):
        img     = cv2.imread(str(img_path))
        patches = generate_patches(img, str(img_path), str(xml_path), crop_size)
        for img_array, savename, patch_annos in patches:
            cv2.imwrite(os.path.join(dst_folder, savename), img_array)
            coco_data["images"].append({
                "id": image_id, "file_name": savename,
                "width": img_array.shape[1], "height": img_array.shape[0]
            })
            for ann in patch_annos:
                ann_entry = {"id": annotation_id, "image_id": image_id}
                ann_entry.update(ann)
                coco_data["annotations"].append(ann_entry)
                annotation_id += 1
            image_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"    Saved {image_id - 1} patches to {output_json}")


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


# ── Main ───────────────────────────────────────────────────────────────────────

# Load test set paths once — shared across all PGS experiments and folds
test_img_paths = sorted(glob(os.path.join(TEST_IMAGE_ROOT, "**/*.png"), recursive=True))
test_xml_paths = [get_xml_path(p, TEST_ANNO_ROOT) for p in test_img_paths]

for pgs_idx in range(len(GROWTH_STAGES)):
    pgs_name          = f"PGS_{pgs_idx + 1}"
    cumulative_stages = GROWTH_STAGES[:pgs_idx + 1]

    print(f"\n{'='*60}")
    print(f"Processing {pgs_name} — stages: {cumulative_stages}")
    print(f"{'='*60}")

    # Step 1 — Collect all original images for cumulative stages
    all_img_paths = get_images_for_stages(TRAIN_IMAGE_ROOT, cumulative_stages)
    if not all_img_paths:
        print(f"[Warning] No images found for {pgs_name}, skipping.")
        continue

    print(f"Total original images: {len(all_img_paths)}")

    # Step 2 — Generate ALL patches in memory first, then split 80/20
    all_patches = []
    for img_path in all_img_paths:
        xml_path = get_xml_path(img_path, TRAIN_ANNO_ROOT)
        img      = cv2.imread(img_path)
        patches  = generate_patches(img, img_path, xml_path, CROP_SIZE)
        all_patches.extend(patches)

    print(f"Total patches generated: {len(all_patches)}")

    # Step 3 — Create 5 independent 80/20 random splits on patches
    for fold_idx in range(N_FOLDS):
        fold_name = f"fold{fold_idx + 1}"
        fold_root = os.path.join(OUTPUT_ROOT, f"{pgs_name}_{fold_name}")

        print(f"\n  {pgs_name} — {fold_name}")

        # Independent random split per fold using seed + fold_idx
        random.seed(SEED + fold_idx)
        shuffled_patches = all_patches.copy()
        random.shuffle(shuffled_patches)

        n_val         = max(1, int(len(shuffled_patches) * VAL_RATIO))
        val_patches   = shuffled_patches[:n_val]
        train_patches = shuffled_patches[n_val:]

        print(f"    train patches: {len(train_patches)} | "
              f"val patches: {len(val_patches)}")

        # Save train patches
        save_patches(
            patches=train_patches,
            dst_folder=os.path.join(fold_root, "train_images/"),
            output_json=os.path.join(fold_root, "train.json")
        )

        # Save val patches
        save_patches(
            patches=val_patches,
            dst_folder=os.path.join(fold_root, "val_images/"),
            output_json=os.path.join(fold_root, "val.json")
        )

        # Save test patches — same held-out test set for all experiments
        process_images_direct(
            test_img_paths, test_xml_paths,
            dst_folder=os.path.join(fold_root, "test_images/"),
            crop_size=CROP_SIZE,
            output_json=os.path.join(fold_root, "test.json")
        )

        # Generate annotation txt files
        for split in ["train", "val", "test"]:
            save_annotation_txt(
                image_folder=os.path.join(fold_root, f"{split}_images/"),
                output_txt=os.path.join(fold_root, f"{split}.txt")
            )

print("\nAll PGS experiments processed successfully.")
