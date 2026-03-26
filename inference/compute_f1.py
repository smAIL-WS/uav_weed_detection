import os
import csv
import numpy as np
import torch
import xml.etree.ElementTree as ET
from glob import glob


# ── User Configuration ─────────────────────────────────────────────────────────
GT_XML_ROOT  = "<specify_path_to>/annotations/test/"   # folder containing XML files
PRED_ROOT    = "<specify_path_to>/saved_predictions/"  # folder containing .pt files
OUTPUT_CSV   = "csv_metrics/f1_metrics.csv"

EXPERIMENT   = "<specify_experiment_name>"             # e.g. 'pgs_1', 'full_dataset'
MODEL        = "<specify_model_name>"                  # e.g. 'gdino', 'retinanet'
CLASS_NAMES  = ["crop", "weed"]
IOU_THRESH   = 0.5
SCORE_THRESH = 0.5

# NOTE: Model prediction bounding boxes must be in [xmin, ymin, xmax, ymax] format.
# Ground truth XML annotations are read directly and converted to
# [xmin, ymin, xmax, ymax] format automatically.
# ──────────────────────────────────────────────────────────────────────────────


def load_ground_truth_from_xml(xml_root):
    """
    Load ground truth bounding boxes and labels directly from XML files.
    Boxes are returned in [xmin, ymin, xmax, ymax] format.

    Args:
        xml_root : Path to folder containing XML annotation files

    Returns:
        gt_bboxes : List[List] — boxes per image
        gt_labels : List[List] — class ids per image (0=crop, 1=weed)
    """
    category_map = {"crop": 0, "weed": 1}
    xml_files    = sorted(glob(os.path.join(xml_root, "**/*.xml"), recursive=True))

    gt_bboxes = []
    gt_labels = []

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_boxes  = []
        img_labels = []

        for obj in root.findall("object"):
            category = obj.find("name").text
            if category not in category_map:
                continue

            bndbox = obj.find("bndbox")
            xmin   = int(float(bndbox.find("xmin").text))
            ymin   = int(float(bndbox.find("ymin").text))
            xmax   = int(float(bndbox.find("xmax").text))
            ymax   = int(float(bndbox.find("ymax").text))

            img_boxes.append([xmin, ymin, xmax, ymax])
            img_labels.append(category_map[category])

        gt_bboxes.append(img_boxes)
        gt_labels.append(img_labels)

    print(f"Loaded ground truth from {len(xml_files)} XML files")
    return gt_bboxes, gt_labels


def get_iou(a, b, epsilon=1e-5):
    """Compute IoU between two boxes in [xmin, ymin, xmax, ymax] format."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    w  = x2 - x1
    h  = y2 - y1

    if w < 0 or h < 0:
        return 0.0

    intersection = w * h
    area_a       = (a[2] - a[0]) * (a[3] - a[1])
    area_b       = (b[2] - b[0]) * (b[3] - b[1])

    return intersection / (area_a + area_b - intersection + epsilon)


def calc_tp_fp_fn(gt_boxes, pred_boxes):
    """Compute TP, FP, FN for a single image."""
    gt_matched   = np.zeros(len(gt_boxes))
    pred_matched = np.zeros(len(pred_boxes))

    for i, gt_box in enumerate(gt_boxes):
        iou_scores = [get_iou(gt_box, pred_box) for pred_box in pred_boxes]
        if iou_scores:
            best_idx = int(np.argmax(iou_scores))
            if iou_scores[best_idx] >= IOU_THRESH:
                gt_matched[i]          = 1
                pred_matched[best_idx] = 1

    tp = int(np.sum(gt_matched))
    fn = len(gt_boxes)   - tp
    fp = len(pred_boxes) - int(np.sum(pred_matched))

    return tp, fp, fn


def compute_f1(gt_bboxes, gt_labels, pred_bboxes, pred_scores, pred_labels):
    """
    Compute per-class precision, recall and F1 score over all images.

    Args:
        gt_bboxes   : List[List] — GT boxes per image in [xmin, ymin, xmax, ymax]
        gt_labels   : List[List] — GT class ids per image (0=crop, 1=weed)
        pred_bboxes : List[List] — predicted boxes per image
        pred_scores : List[List] — predicted confidence scores per image
        pred_labels : List[List] — predicted class ids per image

    Returns:
        dict with precision, recall, f1 per class
    """
    results = {}

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        total_tp, total_fp, total_fn = 0, 0, 0

        for i in range(len(gt_bboxes)):

            filtered_preds = [
                pred_bboxes[i][j]
                for j in range(len(pred_bboxes[i]))
                if pred_scores[i][j] > SCORE_THRESH
                and pred_labels[i][j] == cls_id
            ]
            filtered_gt = [
                gt_bboxes[i][j]
                for j in range(len(gt_bboxes[i]))
                if gt_labels[i][j] == cls_id
            ]

            tp, fp, fn = calc_tp_fp_fn(filtered_gt, filtered_preds)
            total_tp  += tp
            total_fp  += fp
            total_fn  += fn

        precision = round(total_tp / (total_tp + total_fp), 3) \
                    if (total_tp + total_fp) > 0 else 0.0
        recall    = round(total_tp / (total_tp + total_fn), 3) \
                    if (total_tp + total_fn) > 0 else 0.0
        f1        = round(2 * precision * recall / (precision + recall), 3) \
                    if (precision + recall) > 0 else 0.0

        results[cls_name] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1
        }

    return results


def save_to_csv(csv_path, experiment, model, metrics):
    """Save metrics to CSV file."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "experiment", "model",
                "precision_crop", "recall_crop", "f1_crop",
                "precision_weed", "recall_weed", "f1_weed"
            ])
        writer.writerow([
            experiment, model,
            metrics["crop"]["precision"],
            metrics["crop"]["recall"],
            metrics["crop"]["f1"],
            metrics["weed"]["precision"],
            metrics["weed"]["recall"],
            metrics["weed"]["f1"],
        ])


# ── Main ───────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Load ground truth directly from XML
gt_bboxes, gt_labels = load_ground_truth_from_xml(GT_XML_ROOT)

# Load predictions from .pt files
print(f"\nProcessing: {EXPERIMENT} | {MODEL}")
pred_bboxes = torch.load(os.path.join(PRED_ROOT, "full_pred_nms_boxes.pt"))
pred_scores = torch.load(os.path.join(PRED_ROOT, "full_pred_scores.pt"))
pred_labels = torch.load(os.path.join(PRED_ROOT, "full_pred_labels.pt"))

# Compute F1 over all test images
metrics = compute_f1(gt_bboxes, gt_labels, pred_bboxes, pred_scores, pred_labels)

# Print results
print(f"\n  crop — P={metrics['crop']['precision']:.3f} "
      f"R={metrics['crop']['recall']:.3f} "
      f"F1={metrics['crop']['f1']:.3f}")
print(f"  weed — P={metrics['weed']['precision']:.3f} "
      f"R={metrics['weed']['recall']:.3f} "
      f"F1={metrics['weed']['f1']:.3f}")

# Save to CSV
save_to_csv(OUTPUT_CSV, EXPERIMENT, MODEL, metrics)
print(f"\nMetrics saved to {OUTPUT_CSV}")