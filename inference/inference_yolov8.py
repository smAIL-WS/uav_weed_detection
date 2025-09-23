from mmdet.apis import init_detector, inference_detector
import torch
import numpy as np
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules
from torchvision.ops import nms
import cv2
import json
import matplotlib.pyplot as plt
from glob import glob

# Paths to config and checkpoint
config_path = 'mmyolo/configs/yolov8/yolov8_s_fast_1xb12-40e_crop_weed.py'
checkpoint_path = 'mmyolo/work_dirs/<checkpoint>.pth'

test_image_path = "./groundtruths/images/test_image.png"
pred_save_path = "./predictions/"

def load_yolov8_model(config_path, checkpoint_path, device='cuda:0'):
    """
    Load a YOLOv8 model from MMDetection framework.

    Args:
        config_path: Path to the YOLOv8 model config file.
        checkpoint_path: Path to the trained checkpoint file.
        device: Device to load the model on (e.g., 'cuda:0').

    Returns:
        model: The initialized model.
    """

    register_all_modules()  # Ensure MMDetection modules are registered
    model = init_detector(config_path, checkpoint_path, device=device)
    return model


def sliding_window_inference_yolov8(model, image, image_name, window_sizes=(512, 256), stride=256, iou_threshold=0.5):
    """
    Perform sliding window inference on an image using YOLOv8 model.

    Args:
        model: The initialized YOLOv8 MMDetection model.
        image: The input image as a numpy array (H, W, C).
        image_name: Name of the image being processed.
        window_sizes: Tuple of two window sizes for detecting objects.
        stride: Step size for sliding windows.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Aggregated detections (list of bounding box coordinates, scores, and labels).
    """
    
    h, w, c = image.shape
    assert c == 3, "The input image must have 3 channels (RGB)."
    
    pred_bbox_coll = []
    score_coll = []
    label_coll = []

    # Iterate over each window size
    for window_size in window_sizes:
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = image[y:y + window_size, x:x + window_size, :]

                # Perform inference on the window
                result = inference_detector(model, window)
                
                if isinstance(result, DetDataSample):  # MMDetection's output structure
                    pred_instances = result.pred_instances
                elif isinstance(result, InstanceData):
                    pred_instances = result
                else:
                    raise TypeError("Unexpected model output format.")

                bbox_results = pred_instances.bboxes.cpu().numpy()

                # Adjust coordinates based on window position
                temp_coll = np.zeros_like(bbox_results)
                for i in range(bbox_results.shape[0]):
                    xmin, ymin, xmax, ymax = bbox_results[i]
                    temp_coll[i] = [xmin + x, ymin + y, xmax + x, ymax + y]

                # Convert to torch tensors for NMS processing
                temp_coll = torch.Tensor(temp_coll).to(device='cuda')
                pred_bbox_coll.append(temp_coll)
                score_coll.append(pred_instances.scores.to(device='cuda'))
                label_coll.append(pred_instances.labels.to(device='cuda'))

    # Stack predictions
    if len(pred_bbox_coll) == 0:
        return [], [], []

    pred_bboxes = torch.vstack(pred_bbox_coll)
    scores = torch.cat(score_coll)
    labels = torch.cat(label_coll)

    # Apply Non-Maximum Suppression (NMS)
    keep_indices = nms(pred_bboxes, scores, iou_threshold)
    bbox_preds_kept = pred_bboxes[keep_indices]
    scores_preds = scores[keep_indices]
    labels_preds = labels[keep_indices]

    return bbox_preds_kept, scores_preds, labels_preds


# Load model
model = load_yolov8_model(config_path, checkpoint_path, device='cuda:0')

iou_threshold = 0.5
img = cv2.imread(test_image_path)
img_name = test_image_path.split('/')[-1].split('.')[0]

bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_yolov8(model, img, img_name, window_sizes=(512, 1024), stride=256, iou_threshold=0.5)

    
# Save predictions of all images in testset in a single tensor
    
torch.save(bbox_preds_kept,f'{pred_save_path}/full_pred_boxes_yolo_best_config.pt')
torch.save(scores_preds, f'{pred_save_path}/full_pred_scores_yolo_best_config.pt')
torch.save(labels_preds, f'{pred_save_path}/full_pred_labels_yolo_best_config.pt')