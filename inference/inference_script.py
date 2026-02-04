import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from torchvision.ops import nms
import cv2
import json
import matplotlib.pyplot as plt
import torchvision
from glob import glob
import torch
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample

# Paths to config and checkpoint
config_path = 'mmdetection/configs/grounding_dino/dino_swin-t_finetune_8xb2_20e_crop_weed.py'
checkpoint_path = 'mmdetection/work_dirs/<checkpoint>.pth'

test_image_path = "./groundtruths/images/test_image.png"
pred_save_path = "./predictions/"

nms_iou_threshold = 0.5

# Inference on single image
img = cv2.imread(test_image_path)
img_name = test_image_path.split('/')[-1].split('.')[0]

# Initialize the model
def load_model(config_path, checkpoint_path, device='cuda:0'):
    """
    Load the finetuned RetinaNet model using a configuration file from MMDetection's GitHub repository.

    Args:
        config_path: Path to the model configuration file.
        checkpoint_path: Path to the model checkpoint file.
        device: Device to load the model on (e.g., 'cuda:0').

    Returns:
        model: The initialized model.
    """
    
    model = init_detector(config_path, checkpoint_path, device=device)
    return model


def sliding_window_inference_dino_detr(model, image, image_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3):
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

def sliding_window_inference_retinanet(model, image, image_name, window_sizes=(512, 1024), stride=256, iou_threshold=0.3):
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
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), iou_threshold).cpu()
    
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_bboxes, final_scores, final_labels

def sliding_window_inference_yolov8(model, image, image_name, window_sizes=(512, 1024), stride=256, iou_threshold=0.3):
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
    keep_indices = nms(all_bboxes.cuda(), all_scores.cuda(), iou_threshold).cpu()
    
    final_bboxes = all_bboxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_labels = all_labels[keep_indices]

    return final_bboxes, final_scores, final_labels

model = load_model(config_path, checkpoint_path, device='cuda:0')

# Perform inference on Groudning Dino
bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_grounding_dino(model, img, img_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3)

# # Perform inference on DINO-DETR
# bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_dino_detr(model, img, img_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3)

# # Perform inference on DINO-DETR
# bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_dino_detr(model, img, img_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3)

# # Perform inference on RetinaNet
# bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_retinanet(model, img, img_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3)

# # Perform inference on YOLOv8
# bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_yolov8(model, img, img_name, window_sizes=(512, 1024), stride=256, nms_iou_threshold=0.3)

# Save predictions in a single tensor   
torch.save(bbox_preds_kept,f'{pred_save_path}/full_pred_boxes.pt')
torch.save(scores_preds, f'{pred_save_path}/full_pred_scores.pt')
torch.save(labels_preds, f'{pred_save_path}/full_pred_labels.pt')

