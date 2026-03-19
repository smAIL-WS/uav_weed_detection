import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from torchvision.ops import nms
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from glob import glob
import os
import shutil
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from mmdet.structures import DetDataSample

# Initialize the model
def load_model(config_path, checkpoint_path, device='cuda:0'):
    """
    Load the finetuned Grounding DINO model using a configuration file from MMDetection's GitHub repository.

    Args:
        config_path: Path to the model configuration file.
        checkpoint_path: Path to the model checkpoint file.
        device: Device to load the model on (e.g., 'cuda:0').

    Returns:
        model: The initialized model.
    """
    model = init_detector(config_path, checkpoint_path, device=device)
    return model


def sliding_window_inference_grounding_dino(model, image, image_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3):
    """
    Perform sliding window inference on an image using the provided model with two window sizes.

    Args:
        model: The initialized MMDetection model.
        image: The input image as a numpy array (H, W, C).
        image_name: Name of the input image.
        window_sizes: Tuple of two window sizes for detecting smaller and larger objects.
        stride: Step size for sliding windows.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Aggregated detections (list of bounding box coordinates, scores, and labels).
    """
    h, w, _ = image.shape
    pred_bbox_coll, score_coll, label_coll = [], [], []

    # 1. Multi-scale Tiling 
    for win_size in window_sizes:
        # Generate coordinates with 'Edge-Snapping' to cover every pixel
        y_coords = list(range(0, h - win_size + 1, stride))
        if h > win_size and (not y_coords or y_coords[-1] != h - win_size):
            y_coords.append(h - win_size)
            
        x_coords = list(range(0, w - win_size + 1, stride))
        if w > win_size and (not x_coords or x_coords[-1] != w - win_size):
            x_coords.append(w - win_size)

        for y in y_coords:
            for x in x_coords:
                tile = image[y:y + win_size, x:x + win_size, :]
                
                # Inference (Grounding DINO specific prompt format)
                result = inference_detector(model, tile, text_prompt='crop . weed')
                
                instances = result.pred_instances
                if len(instances) == 0:
                    continue

                # 2. Vectorized Local-to-Global Coordinate Mapping
                bboxes = instances.bboxes.cpu() # [N, 4]
                offset = torch.tensor([x, y, x, y], dtype=torch.float32)
                global_bboxes = bboxes + offset
                
                pred_bbox_coll.append(global_bboxes)
                score_coll.append(instances.scores.cpu())
                label_coll.append(instances.labels.cpu())

    if not pred_bbox_coll:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))

    # 3. Consolidate Detections
    all_bboxes = torch.cat(pred_bbox_coll, dim=0)
    all_scores = torch.cat(score_coll, dim=0)
    all_labels = torch.cat(label_coll, dim=0)

    # 4. Standard NMS
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), nms_iou_threshold).cpu()
    
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_bboxes, final_scores, final_labels


def sliding_window_inference_dino(model, image, image_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3):
    """
    Perform sliding window inference on an image using the provided model with two window sizes.

    Args:
        model: The initialized MMDetection model.
        image: The input image as a numpy array (H, W, C).
        image_name: Name of the input image.
        window_sizes: Tuple of two window sizes for detecting smaller and larger objects.
        stride: Step size for sliding windows.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Aggregated detections (list of bounding box coordinates, scores, and labels).
    """
    h, w, _ = image.shape
    pred_bbox_coll, score_coll, label_coll = [], [], []

    # 1. Multi-scale Tiling 
    for win_size in window_sizes:
        # Generate coordinates with 'Edge-Snapping' to cover every pixel
        y_coords = list(range(0, h - win_size + 1, stride))
        if h > win_size and (not y_coords or y_coords[-1] != h - win_size):
            y_coords.append(h - win_size)
            
        x_coords = list(range(0, w - win_size + 1, stride))
        if w > win_size and (not x_coords or x_coords[-1] != w - win_size):
            x_coords.append(w - win_size)

        for y in y_coords:
            for x in x_coords:
                tile = image[y:y + win_size, x:x + win_size, :]
                
                # Inference
                result = inference_detector(model, tile)
                
                if isinstance(result, DetDataSample):
                    instances = result.pred_instances
                    if len(instances) == 0:
                        continue

                    # 2. Vectorized Local-to-Global Coordinate Mapping
                    bboxes = instances.bboxes.cpu() # [N, 4]
                    offset = torch.tensor([x, y, x, y], dtype=torch.float32)
                    global_bboxes = bboxes + offset
                    
                    pred_bbox_coll.append(global_bboxes)
                    score_coll.append(instances.scores.cpu())
                    label_coll.append(instances.labels.cpu())

    if not pred_bbox_coll:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))

    # 3. Consolidate Detections
    all_bboxes = torch.cat(pred_bbox_coll, dim=0)
    all_scores = torch.cat(score_coll, dim=0)
    all_labels = torch.cat(label_coll, dim=0)

    # 4. Standard NMS
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), nms_iou_threshold).cpu()
    
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]


    return final_bboxes, final_scores, final_labels

def sliding_window_inference_retinanet(model, image, image_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3):
    """
    Perform sliding window inference on an image using the provided model with two window sizes.

    Args:
        model: The initialized MMDetection model.
        image: The input image as a numpy array (H, W, C).
        image_name: Name of the input image.
        window_sizes: Tuple of two window sizes for detecting smaller and larger objects.
        stride: Step size for sliding windows.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Aggregated detections (list of bounding box coordinates, scores, and labels).
    """
    h, w, _ = image.shape
    pred_bbox_coll, score_coll, label_coll = [], [], []

    # 1. Multi-scale Tiling 
    for win_size in window_sizes:
        # Generate coordinates with 'Edge-Snapping' to cover every pixel
        y_coords = list(range(0, h - win_size + 1, stride))
        if h > win_size and (not y_coords or y_coords[-1] != h - win_size):
            y_coords.append(h - win_size)
            
        x_coords = list(range(0, w - win_size + 1, stride))
        if w > win_size and (not x_coords or x_coords[-1] != w - win_size):
            x_coords.append(w - win_size)

        for y in y_coords:
            for x in x_coords:
                tile = image[y:y + win_size, x:x + win_size, :]
                
                # Inference
                result = inference_detector(model, tile)
                
                if isinstance(result, DetDataSample):
                    instances = result.pred_instances
                    if len(instances) == 0:
                        continue

                    # 2. Vectorized Local-to-Global Coordinate Mapping
                    bboxes = instances.bboxes.cpu() # [N, 4]
                    offset = torch.tensor([x, y, x, y], dtype=torch.float32)
                    global_bboxes = bboxes + offset
                    
                    pred_bbox_coll.append(global_bboxes)
                    score_coll.append(instances.scores.cpu())
                    label_coll.append(instances.labels.cpu())

    if not pred_bbox_coll:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))

    # 3. Consolidate Detections
    all_bboxes = torch.cat(pred_bbox_coll, dim=0)
    all_scores = torch.cat(score_coll, dim=0)
    all_labels = torch.cat(label_coll, dim=0)

    # 4. Standard NMS
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), nms_iou_threshold).cpu()
    
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_bboxes, final_scores, final_labels

def sliding_window_inference_yolov8(model, image, image_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3):
    """
    Perform sliding window inference on an image using the provided model with two window sizes.

    Args:
        model: The initialized MMDetection model.
        image: The input image as a numpy array (H, W, C).
        image_name: Name of the input image.
        window_sizes: Tuple of two window sizes for detecting smaller and larger objects.
        stride: Step size for sliding windows.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Aggregated detections (list of bounding box coordinates, scores, and labels).
    """
    h, w, _ = image.shape
    pred_bbox_coll, score_coll, label_coll = [], [], []

    # 1. Multi-scale Tiling 
    for win_size in window_sizes:
        # Generate coordinates with 'Edge-Snapping' to cover every pixel
        y_coords = list(range(0, h - win_size + 1, stride))
        if h > win_size and (not y_coords or y_coords[-1] != h - win_size):
            y_coords.append(h - win_size)
            
        x_coords = list(range(0, w - win_size + 1, stride))
        if w > win_size and (not x_coords or x_coords[-1] != w - win_size):
            x_coords.append(w - win_size)

        for y in y_coords:
            for x in x_coords:
                tile = image[y:y + win_size, x:x + win_size, :]
                
                # Inference (Grounding DINO specific prompt format)
                result = inference_detector(model, tile)
                
                if isinstance(result, DetDataSample):
                    instances = result.pred_instances
                    if len(instances) == 0:
                        continue

                    # 2. Vectorized Local-to-Global Coordinate Mapping
                    bboxes = instances.bboxes.cpu() # [N, 4]
                    offset = torch.tensor([x, y, x, y], dtype=torch.float32)
                    global_bboxes = bboxes + offset
                    
                    pred_bbox_coll.append(global_bboxes)
                    score_coll.append(instances.scores.cpu())
                    label_coll.append(instances.labels.cpu())

    if not pred_bbox_coll:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,))

    # 3. Consolidate Detections
    all_bboxes = torch.cat(pred_bbox_coll, dim=0)
    all_scores = torch.cat(score_coll, dim=0)
    all_labels = torch.cat(label_coll, dim=0)

    # 4. Standard NMS
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), nms_iou_threshold).cpu()
    
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_bboxes, final_scores, final_labels


def load_groundtruth_from_xml(xml_file, class_map={"crop": 0, "weed": 1}):
    """
    Load ground truth annotations from Pascal VOC-style XML file.
    Returns: gt_bboxes, gt_labels
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes, labels = [], []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in class_map:
            continue  # skip unknown classes

        cls_id = class_map[cls_name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls_id)

    return bboxes, labels  # wrapped in list → per-image format




def plot_predictions(
    image_path,
    pred_bboxes, pred_labels, pred_scores,
    gt_bboxes, gt_labels,
    score_thresh=0.5,
    class_map={0: "crop", 1: "weed"},
    save_dir="visualizations"
):
    """
    Plot ground truth and predictions side by side with a shared legend.
    Saves the figure in high resolution.
    """

    img = Image.open(image_path)

    # Create two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))  # large figsize for drone images

    # --- Ground Truth Plot ---
    axes[0].imshow(img)
    axes[0].set_title("Ground Truth", fontsize=20)
    for bbox, label in zip(gt_bboxes, gt_labels):
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax - xmin, ymax - ymin
        cls_name = class_map[label]
        color = "blue" if cls_name == "crop" else "red"

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        axes[0].add_patch(rect)
        # axes[0].text(
        #     xmin, ymin - 5, f"{cls_name}",
        #     fontsize=12, color=color, weight="bold"
        # )
    axes[0].axis("off")

    # --- Prediction Plot ---
    axes[1].imshow(img)
    axes[1].set_title("Predictions", fontsize=20)
    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if score <= score_thresh:
            continue

        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = xmin.cpu().numpy(), ymin.cpu().numpy(), xmax.cpu().numpy(), ymax.cpu().numpy()
        width, height = xmax - xmin, ymax - ymin
        cls_name = class_map[label.cpu().item()]
        color = "blue" if cls_name == "crop" else "red"

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        axes[1].add_patch(rect)
        # axes[1].text(
        #     xmin, ymin - 5, f"{cls_name} ({score:.2f})",
        #     fontsize=12, color=color, weight="bold"
        # )
    axes[1].axis("off")

    # --- Shared Legend ---
    handles = [
        patches.Patch(color="blue", label="Crop"),
        patches.Patch(color="red", label="Weed"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=16,
        frameon=False
    )

    # --- Save the figure ---
    save_path = f"{save_dir}/gt_pred_{image_path.split('/')[-1]}"
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend at bottom
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, format="png", dpi=600)  # high resolution
    plt.show()
    plt.close(fig)