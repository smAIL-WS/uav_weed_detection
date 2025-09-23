import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from torchvision.ops import nms
import cv2
import json
import matplotlib.pyplot as plt
import torchvision
from glob import glob
import cv2
import os
import shutil
import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Paths to config and checkpoint
config_path = 'mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_crop_weed.py'
checkpoint_path = 'mmdetection/work_dirs/<checkpoint>.pth'

test_image_path = "./groundtruths/images/test_image.png"
pred_save_path = "./predictions/"

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


def sliding_window_inference(model, image, image_name, window_sizes=(512, 256), stride=256, iou_threshold=0.5):
    """
    Perform sliding window inference on an image using the provided model with two window sizes.

    Args:
        model: The initialized MMDetection model.
        image: The input image as a numpy array (H, W, C).
        window_sizes: Tuple of two window sizes for detecting smaller and larger objects.
        stride: Step size for sliding windows.
        conf_threshold: Confidence threshold for filtering predictions.
        iou_threshold: IoU threshold for NMS.
        output_json_path: Path to save the predictions in JSON format.

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
                result = inference_detector(model, window, text_prompt=['crop','weed'])
                bbox_results = result.pred_instances.bboxes.cpu().numpy()
                # print(bbox_results.shape)
                # Filter by confidence threshold and adjust coordinates
                temp_coll = np.zeros(bbox_results.shape)
                for i in range(bbox_results.shape[0]):
                    xmin, ymin, xmax, ymax = bbox_results[i,:]
                    temp_coll[i,:] = xmin + x, ymin + y, xmax + x, ymax + y
                temp_coll = torch.Tensor(temp_coll)
                pred_bbox_coll.append(temp_coll)
                score_coll.append(result.pred_instances.scores)
                label_coll.append(result.pred_instances.labels)


    pred_bboxes = torch.vstack(pred_bbox_coll)
    pred_bboxes = pred_bboxes.to(device='cuda')
    scores = torch.vstack([torch.as_tensor(i) for i in score_coll])
    scores = torch.flatten(scores)
    labels = torch.vstack([torch.as_tensor(i) for i in label_coll])
    labels = torch.flatten(labels)

    # print("Len of pred boxes before nms", len(pred_bboxes))

    # Perform NMS
    keep_indices = nms(pred_bboxes, scores, iou_threshold)

    bbox_preds_kept = pred_bboxes[keep_indices]
    scores_preds = scores[keep_indices]
    labels_preds = labels[keep_indices]

    # print("Len of pred boxes after nms", len(bbox_preds_kept))

                

    return bbox_preds_kept, scores_preds, labels_preds

# Load model
model = load_model(config_path, checkpoint_path, device='cuda:0')



iou_threshold = 0.5
img = cv2.imread(test_image_path)
img_name = test_image_path.split('/')[-1].split('.')[0]

bbox_preds_kept, scores_preds, labels_preds = sliding_window_inference(model, img, img_name, window_sizes=(512, 1024), stride=256, iou_threshold=0.5)
    
# Save predictions of all images in testset in a single tensor
    
torch.save(bbox_preds_kept,f'{pred_save_path}/full_pred_boxes_gd_best_config.pt')
torch.save(scores_preds, f'{pred_save_path}/full_pred_scores_gd_best_config.pt')
torch.save(labels_preds, f'{pred_save_path}/full_pred_labels_gd_best_config.pt')