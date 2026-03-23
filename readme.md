# Repository Overview
This repository contains the codebase for our research on data-efficient weed detection using Grounding DINO, a fine-tuned foundation model designed for open-set object detection in agricultural environments. The project evaluates the model’s performance under varying levels of training data availability and compares it against state-of-the-art detectors such as DINO, RetinaNet and YOLOv8. Using expert-annotated UAV imagery from sorghum and maize fields, the study demonstrates that Grounding DINO can achieve robust and accurate weed detection even when fine-tuned with only a small number of representative images per crop growth stage. The repository includes all training pipelines, experimental configurations, and evaluation scripts to enable full reproducibility and support further research in precision agriculture.

Figure 1: Comparative performance for *crop* and *weed* detection for all models and training dataset variants we consider. The values indicate the percentage performance decrease in terms of F1-score and AP relative to the best-performing model (denoted as REF) , where a darker color intensity represents a more significant performance decline.
![](readme_images/data_efficiency.png)

Figure 2: F1 and AP comparison for the class: *crop* and *weed* of fine-tuned Grounding DINO model across Progressive growth stage experiments on the held-out testset.
![](readme_images/progressive_growth_stage.png)

Figure 3. Qualitative comparison of object detection performance across different architectures on a representative test sample from BBCH 13 crop growth stage. The first column illustrates the Ground Truth annotations. Subsequent columns display predictions from Grounding DINO, DINO, RetinaNet, and YOLOv8. The top panel depicts results for models trained on the *full_dataset*, while the bottom panel shows performance for the *quarter_dataset* variant. Blue bounding boxes denote *crop* instances, and red bounding boxes indicate *weed* species. 
![](readme_images/qa_1.png)

Figure 4. Qualitative comparison of object detection performance across different architectures on a representative test sample from BBCH 15 crop growth stage. The first column illustrates the Ground Truth annotations. Subsequent columns display predictions from Grounding DINO, DINO, RetinaNet, and YOLOv8. The top panel depicts results for models trained on the *full_dataset*, while the bottom panel shows performance for the *quarter_dataset* variant. Blue bounding boxes denote *crop* instances, and red bounding boxes indicate *weed* species.
![](readme_images/qa_2.png)

## Installation and Environment Setup

We provide two pre-built Docker images on Docker Hub to reproduce all experiments.

| Image | Models |
|---|---|
| `hswt555har/mmdetection-models:v1.1` | Grounding DINO, RetinaNet, DINO |
| `hswt555har/mmyolo-models:v1.1` | YOLOv8 |

### Pull Docker Images
```bash
docker pull hswt555har/mmdetection-models:v1.1
docker pull hswt555har/mmyolo-models:v1.1
```

### Verify Images
```bash
# Verify mmdetection image
docker run --gpus all hswt555har/mmdetection-models:v1.1 python -c "
import torch, mmcv, mmdet, transformers
from mmcv.ops import MultiScaleDeformableAttention
print('PyTorch  :', torch.__version__)
print('CUDA     :', torch.cuda.is_available())
print('GPU      :', torch.cuda.get_device_name(0))
print('mmcv     :', mmcv.__version__)
print('mmdet    :', mmdet.__version__)
print('CUDA ops : OK')
"

# Verify mmyolo image
docker run --gpus all hswt555har/mmyolo-models:v1.1 python -c "
import torch, mmcv, mmyolo
print('PyTorch  :', torch.__version__)
print('CUDA     :', torch.cuda.is_available())
print('GPU      :', torch.cuda.get_device_name(0))
print('mmcv     :', mmcv.__version__)
print('mmyolo   : OK')
"
```

## Clone the repository

Clone the repository and navigate to the root before running any command:
```bash
git clone https://github.com/smAIL-WS/uav_weed_detection.git
cd uav_weed_detection
```

## Dataset

The EWIS dataset used in this paper is publicly available on Mendeley Data: https://data.mendeley.com/datasets/6j5pxgf437/1. The dataset as published does not include train/test splits or growth stage stratification. Follow the steps below to prepare the dataset for training.

---

### Step 1 — Download the Dataset

Download the dataset from Mendeley Data and place it in your local machine.

---

### Step 2 — Categorize the Dataset

The downloaded dataset is not categorized into train/test splits or stratified by growth stages. Refer to `preprocessing/readme.md` in the repository for the categorization details used in this project and organize the data into the respective folders accordingly.
```
uav_weed_detection/
└── raw_data/
    └── train
        └── images
            ├── BBCH 12/
            ├── BBCH 13/
            └── ...
        └── annotations
            ├── BBCH 12/
            ├── BBCH 13/
            └── ...
    └── test
        └── images
            ├── BBCH 12/
            ├── BBCH 13/
            └── ...
        └── annotations
            ├── BBCH 12/
            ├── BBCH 13/
            └── ...
```

---

### Step 3 — Preprocess the Dataset

Run the preprocessing script to generate 512×512 patches from the original drone images, stratified into train and test splits. The script saves the patches in mmdetection-compatible format inside the repository under `ewis_data/`:
```bash
python some_script.py
```

After running the script, the following structure will be created:
```
your-paper-repo/
└── ewis_data/
    ├── train_images/
    ├── val_images/
    ├── test_images/
    └── ...
```

---

### Step 4 — Create Annotation Files

Run the annotation script to generate the annotation files required by the mmdetection toolbox:
```bash
python another_script.py
```

---

### Step 5 — Update Config Files

Once the dataset is prepared, replace the sample dataset path with the full dataset path in all configuration files:
```python
# Replace this (sample dataset path)
data_root = '/workspace/sample_ewis_data/'

# With this (full preprocessed dataset path)
data_root = '/workspace/ewis_data/'
```

This change needs to be made in the following config files:
- `mmdetection/configs/grounding_dino/gd_full_dataset.py`
- `mmdetection/configs/retinanet/rn_full_dataset.py`
- `mmdetection/configs/dino/dino_config.py`
- `mmyolo/configs/yolov8/yolov8_config.py`

---

### Sample Dataset

A small sample of the dataset is provided in `sample_ewis_data/` in the repository root. This can be used to verify your setup and test the training pipeline before running on the full dataset. The sample data is already in mmdetection-compatible format and the path is pre-configured in all config files.

---

## Dataset Preparation

A sample dataset is provided in `sample_ewis_data/` in the repository root to help verify your setup. This folder contains sample patches in mmdetection-compatible format and can be used to test your training pipeline before running on the full dataset.

To train on your own data, preprocess your original drone images into patches and format them according to the mmdetection framework. Once prepared, update the `data_root` path in the respective config files:
```python
# In mmdetection/configs/grounding_dino/gd_full_dataset.py
# In mmdetection/configs/retinanet/rn_full_dataset.py
# In mmdetection/configs/dino/dino_config.py
# In mmyolo/configs/yolov8/yolov8_config.py

# Replace this with the path to your own dataset
data_root = '/workspace/sample_ewis_data/'

# Example
data_root = '/workspace/path_to_your_data/'
```

---

---

## Running Experiments

### Grounding DINO
```bash
docker run --gpus all \
    --shm-size=8g \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/mmdetection/tools/train.py \
           /workspace/mmdetection/configs/grounding_dino/gd_full_dataset.py
```

### RetinaNet
```bash
docker run --gpus all \
    --shm-size=8g \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/mmdetection/tools/train.py \
           /workspace/mmdetection/configs/retinanet/retinanet_config.py
```

### DINO
```bash
docker run --gpus all \
    --shm-size=8g \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/mmdetection/tools/train.py \
           /workspace/mmdetection/configs/dino/dino_config.py
```

### YOLOv8
```bash
docker run --gpus all \
    --shm-size=8g \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmyolo-models:v1.1 \
    python /workspace/mmyolo/tools/train.py \
           /workspace/mmyolo/configs/yolov8/yolov8_config.py
```

> **Note:** All training outputs including checkpoints and logs are saved to `work_dirs/` in the repository root.

## Inference on Held-out Testset

Before running inference, update the following variables in the respective inference scripts:
- `config_path` — path to the model config file
- `checkpoint_path` — path to the best model checkpoint
- `test_image_path` — path to the test image
- `pred_save_path` — path to save predictions

### Grounding DINO
```bash
docker run --gpus all \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/inference/inference_groundingDino.py
```

### RetinaNet
```bash
docker run --gpus all \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/inference/inference_retinanet.py
```

### YOLOv8
```bash
docker run --gpus all \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmyolo-models:v1.1 \
    python /workspace/inference/inference_yolov8.py
```

The predictions for the test image are saved in `inference/predictions` in `.pt` format.

---

## Visualization

Before running visualization, update the following variables in `inference/prediction_visualization.py`:
- `pred_bboxes` — predicted bounding boxes
- `pred_scores` — predicted confidence scores
- `pred_labels` — predicted labels
- `gt_file` — path to ground truth annotations
- `test_image_path` — path to the test image
```bash
docker run --gpus all \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/inference/prediction_visualization.py
```

The plots are saved in `inference/visualization`.