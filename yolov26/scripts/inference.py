"""
inference.py
────────────────────────────────────────────────────────────────────────────────
Sliding window inference on full drone images using YOLO26.

Two window sizes (512, 1024) are run over each image using edge-snapping
to ensure full image coverage. All detections are merged with a single
global NMS pass.

Output per image:
  results/<variant>/inference/<image_stem>.pt

Each .pt file is a dict:
  {
    "image":   str,               original image filename
    "boxes":   FloatTensor[N, 4], absolute pixel coords (x1, y1, x2, y2)
    "scores":  FloatTensor[N],    confidence scores
    "labels":  LongTensor[N],     class indices
    "names":   list[str],         class names
  }

Usage:
  python scripts/inference.py --variant full_dataset --image_dir /path/to/images
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
import yaml
from ultralytics import YOLO


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sliding_window_inference(
    model,
    image:              np.ndarray,
    image_name:         str,
    window_sizes:       tuple = (512,1024),
    stride:             int   = 256,
    nms_iou_threshold:  float = 0.3,
    conf:               float = 0.05,
    device:             int   = 0,
):
    """
    Perform sliding window inference on a drone image using two window sizes.

    Args:
        model:             Initialised YOLO26 model.
        image:             Input image as numpy array (H, W, C).
        image_name:        Name of the input image.
        window_sizes:      Tuple of window sizes for multi-scale detection.
        stride:            Step size for sliding windows.
        nms_iou_threshold: IoU threshold for NMS.
        conf:              Confidence threshold for detections.
        device:            GPU device id.

    Returns:
        Tuple of (boxes, scores, labels) as CPU tensors.
        boxes:  FloatTensor[N, 4]  absolute pixel coords (x1, y1, x2, y2)
        scores: FloatTensor[N]
        labels: LongTensor[N]
    """
    h, w, _ = image.shape
    pred_bbox_coll, score_coll, label_coll = [], [], []

    # 1. Multi-scale tiling
    for win_size in window_sizes:

        # Edge-snapping: ensure last window covers image edge exactly
        y_coords = list(range(0, h - win_size + 1, stride))
        if h > win_size and (not y_coords or y_coords[-1] != h - win_size):
            y_coords.append(h - win_size)

        x_coords = list(range(0, w - win_size + 1, stride))
        if w > win_size and (not x_coords or x_coords[-1] != w - win_size):
            x_coords.append(w - win_size)

        # Handle images smaller than window size
        if h <= win_size:
            y_coords = [0]
        if w <= win_size:
            x_coords = [0]

        for y in y_coords:
            for x in x_coords:
                tile = image[y: y + win_size, x: x + win_size, :]

                results = model.predict(
                    source=tile,
                    imgsz=win_size,
                    conf=conf,
                    verbose=False,
                    device=device,
                )

                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue

                    # 2. Vectorised local-to-global coordinate mapping
                    bboxes = r.boxes.xyxy.cpu()                          # [N, 4]
                    offset = torch.tensor([x, y, x, y], dtype=torch.float32)
                    global_bboxes = bboxes + offset

                    pred_bbox_coll.append(global_bboxes)
                    score_coll.append(r.boxes.conf.cpu())
                    label_coll.append(r.boxes.cls.cpu().long())

    if not pred_bbox_coll:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    # 3. Consolidate detections
    all_bboxes = torch.cat(pred_bbox_coll, dim=0)
    all_scores = torch.cat(score_coll,     dim=0)
    all_labels = torch.cat(label_coll,     dim=0)

    # 4. Global NMS
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), nms_iou_threshold).cpu()

    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_bboxes, final_scores, final_labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/pipeline_config.yaml")
    parser.add_argument("--variant",   required=True)
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing full drone images")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    base_dir = Path(cfg["project"]["base_dir"])

    ckpt = base_dir / "results" / args.variant / "retrain" / "weights" / "last.pt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}\nRun retrain.py first."

    window_sizes = tuple(cfg["inference"]["window_sizes"])
    stride       = cfg["inference"]["stride"]
    conf         = cfg["inference"]["conf"]
    iou          = cfg["inference"]["iou"]
    device       = cfg["project"]["device"]
    names        = cfg["dataset"]["names"]

    out_dir = base_dir / "results" / args.variant / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.variant}] Sliding window inference")
    print(f"  Checkpoint:   {ckpt}")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Stride:       {stride}")
    print(f"  conf={conf}  iou={iou}\n")

    model = YOLO(str(ckpt))

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        for p in Path(args.image_dir).glob(ext)
    )
    assert image_paths, f"No images found in {args.image_dir}"
    print(f"  Images: {len(image_paths)}\n")

    all_boxes_list  = []   # list of FloatTensor[N, 4], one entry per image
    all_scores_list = []   # list of FloatTensor[N],    one entry per image
    all_labels_list = []   # list of LongTensor[N],     one entry per image
    image_names     = []   # image filenames, preserves order

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  WARNING: could not read {img_path.name}, skipping.")
            continue

        boxes, scores, labels = sliding_window_inference(
            model=model,
            image=image,
            image_name=img_path.name,
            window_sizes=window_sizes,
            stride=stride,
            nms_iou_threshold=iou,
            conf=conf,
            device=device,
        )

        all_boxes_list.append(boxes)
        all_scores_list.append(scores)
        all_labels_list.append(labels)
        image_names.append(img_path.name)

        print(f"  {img_path.name:<40}  detections={len(boxes):4d}")

    # Save 3 .pt files — each is a list with one entry per image
    torch.save(all_boxes_list,  out_dir / "pred_nms_boxes.pt")
    torch.save(all_scores_list, out_dir / "pred_scores.pt")
    torch.save(all_labels_list, out_dir / "pred_labels.pt")

    # Save image name order for reference
    torch.save(image_names, out_dir / "image_names.pt")

    print(f"\n✓ Done. Saved:")
    print(f"  {out_dir}/boxes.pt   — {len(all_boxes_list)} images")
    print(f"  {out_dir}/scores.pt  — {len(all_scores_list)} images")
    print(f"  {out_dir}/labels.pt  — {len(all_labels_list)} images")
    print(f"  {out_dir}/image_names.pt")


if __name__ == "__main__":
    main()