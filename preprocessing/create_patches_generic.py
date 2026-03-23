import os
import cv2
import json
import numpy as np
import xml.etree.ElementTree as ET
from glob import glob


# ── User Configuration ─────────────────────────────────────────────────────────
TRAIN_IMAGE_ROOT = "uav_weed_detection/raw_data/train/images/"
TRAIN_ANNO_ROOT  = "uav_weed_detection/raw_data/train/annotations/"
TEST_IMAGE_ROOT  = "uav_weed_detection/raw_data/test/images/"
TEST_ANNO_ROOT   = "uav_weed_detection/raw_data/test/annotations/"
OUTPUT_ROOT      = "uav_weed_detection/ewis_data/"
CROP_SIZE        = 512
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


def crop_and_annotate(img, img_path, xml_path, dst_folder, crop_size, coco_data, annotation_id, image_id):
    """Crop image into patches of crop_size x crop_size with zero padding at edges."""
    h, w, d   = img.shape
    orig_name = os.path.basename(img_path).replace('.png', '')
    orig_annotations = parse_xml_annotations(xml_path)

    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):

            actual_crop       = img[i:min(i + crop_size, h), j:min(j + crop_size, w), :]
            curr_h, curr_w, _ = actual_crop.shape
            img_padded        = np.zeros((crop_size, crop_size, d), dtype=img.dtype)
            img_padded[0:curr_h, 0:curr_w, :] = actual_crop

            savename  = f"{orig_name}_patch_{i // crop_size}_{j // crop_size}.png"
            savepath  = os.path.join(dst_folder, savename)
            denom     = np.max(img_padded) - np.min(img_padded)
            img_final = (255 * (img_padded - np.min(img_padded)) / denom).astype(np.uint8) \
                        if denom > 0 else img_padded.astype(np.uint8)
            cv2.imwrite(savepath, img_final)

            coco_data["images"].append({
                "id": image_id, "file_name": savename,
                "width": crop_size, "height": crop_size
            })

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
                        coco_data["annotations"].append({
                            "id":          annotation_id,
                            "image_id":    image_id,
                            "category_id": ann["category_id"],
                            "bbox":        [new_x_min, new_y_min, new_w, new_h],
                            "area":        float(new_w * new_h),
                            "iscrowd":     0
                        })
                        annotation_id += 1

            image_id += 1

    return annotation_id, image_id


def process_images(image_paths, xml_paths, dst_folder, crop_size, output_json):
    """Process a list of images and save patches with COCO-format annotations."""
    os.makedirs(dst_folder, exist_ok=True)
    coco_data = {
        "images": [], "annotations": [],
        "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"}]
    }
    annotation_id = 1
    image_id      = 1
    for img_path, xml_path in zip(image_paths, xml_paths):
        img = cv2.imread(img_path)
        annotation_id, image_id = crop_and_annotate(
            img, img_path, xml_path, dst_folder,
            crop_size, coco_data, annotation_id, image_id
        )
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"Saved {image_id - 1} patches and annotations to {output_json}")


def save_annotation_txt(image_folder, output_txt):
    """Save image filenames (without extension) to a txt file."""
    img_paths = sorted(glob(os.path.join(image_folder, "*.png")))
    filenames = [os.path.splitext(os.path.basename(p))[0] + "\n" for p in img_paths]
    with open(output_txt, 'w') as f:
        f.writelines(filenames)
    print(f"Annotation txt saved at {output_txt}")


# ── Main ───────────────────────────────────────────────────────────────────────

# Train — all training images
train_img_paths = sorted(glob(os.path.join(TRAIN_IMAGE_ROOT, "**/*.png"), recursive=True))
train_xml_paths = sorted(glob(os.path.join(TRAIN_ANNO_ROOT,  "**/*.xml"), recursive=True))

process_images(
    train_img_paths, train_xml_paths,
    dst_folder=os.path.join(OUTPUT_ROOT, "train_images/"),
    crop_size=CROP_SIZE,
    output_json=os.path.join(OUTPUT_ROOT, "train.json")
)

# Test and Val — test images used for both to maintain training pipeline compatibility
test_img_paths = sorted(glob(os.path.join(TEST_IMAGE_ROOT, "**/*.png"), recursive=True))
test_xml_paths = sorted(glob(os.path.join(TEST_ANNO_ROOT,  "**/*.xml"), recursive=True))

for split in ["val", "test"]:
    process_images(
        test_img_paths, test_xml_paths,
        dst_folder=os.path.join(OUTPUT_ROOT, f"{split}_images/"),
        crop_size=CROP_SIZE,
        output_json=os.path.join(OUTPUT_ROOT, f"{split}.json")
    )

# ── Generate Annotation txt Files ─────────────────────────────────────────────
for split in ["train", "val", "test"]:
    save_annotation_txt(
        image_folder=os.path.join(OUTPUT_ROOT, f"{split}_images/"),
        output_txt=os.path.join(OUTPUT_ROOT, f"{split}.txt")
    )