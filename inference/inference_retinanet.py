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
config_path = '../mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco_crop_weed.py'
checkpoint_path = '../work_dirs/retinanet_r50_fpn_2x_coco_crop_weed/epoch_3.pth'

test_image_path = "groundtruths/images/Platte_20220520_Maize_048.png"
pred_save_path = "predictions/"


def load_retinanet_model(config_path, checkpoint_path, device='cuda:0'):
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



def sliding_window_inference_retinanet(model, image, image_name, window_sizes=(512, 256), stride=256, iou_threshold=0.5):
    """
    Perform sliding window inference on an image using the provided RetinaNet model with two window sizes.

    Args:
        model: The initialized MMDetection RetinaNet model.
        image: The input image as a numpy array (H, W, C).
        window_sizes: Tuple of two window sizes for detecting smaller and larger objects.
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

    # Process each window size
    for window_size in window_sizes:
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = image[y:y + window_size, x:x + window_size, :]

                # Perform inference on the window
                result = inference_detector(model, window)

                if isinstance(result, DetDataSample):
                    instances = result.pred_instances  # Extract InstanceData
                    
                    bbox_results = instances.bboxes.cpu().numpy()  # Bounding boxes
                    scores = instances.scores.cpu().numpy()  # Confidence scores
                    labels = instances.labels.cpu().numpy()  # Class labels

                    # Adjust bounding box coordinates based on the current window position
                    if len(bbox_results) > 0:
                        temp_coll = np.zeros_like(bbox_results)
                        for i in range(len(bbox_results)):
                            xmin, ymin, xmax, ymax = bbox_results[i]
                            temp_coll[i] = [xmin + x, ymin + y, xmax + x, ymax + y]

                        # Convert to tensors for later processing
                        pred_bbox_coll.append(torch.tensor(temp_coll, dtype=torch.float32))
                        score_coll.append(torch.tensor(scores, dtype=torch.float32))
                        label_coll.append(torch.tensor(labels, dtype=torch.int64))

    if not pred_bbox_coll:
        return torch.empty(0), torch.empty(0), torch.empty(0)  # No detections

    # Convert lists to tensors for NMS
    pred_bboxes = torch.cat(pred_bbox_coll).to(device='cuda')
    scores = torch.cat(score_coll).to(device='cuda')
    labels = torch.cat(label_coll).to(device='cuda')

    # Perform NMS
    keep_indices = nms(pred_bboxes, scores, iou_threshold)

    # Apply NMS results
    bbox_preds_kept = pred_bboxes[keep_indices]
    scores_preds = scores[keep_indices]
    labels_preds = labels[keep_indices]

    return bbox_preds_kept, scores_preds, labels_preds



# Load model
model = load_retinanet_model(config_path, checkpoint_path, device='cuda:0')

iou_threshold = 0.5
img = cv2.imread(test_image_path)
img_name = test_image_path.split('/')[-1].split('.')[0]

bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference_retinanet(model, img, img_name, window_sizes=(512, 1024), stride=256, iou_threshold=0.5)
    
# Save predictions of all images in testset in a single tensor
    
torch.save(bbox_preds_kept,f'{pred_save_path}/full_pred_boxes_retinanet_best_config.pt')
torch.save(scores_preds, f'{pred_save_path}/full_pred_scores_retinanet_best_config.pt')
torch.save(labels_preds, f'{pred_save_path}/full_pred_labels_retinanet_best_config.pt')

