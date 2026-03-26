The original paper can be accessed here: [link].

# Repository Overview

# Repository Overview

This repository contains the codebase for our research on data-efficient weed detection using Grounding DINO, a fine-tuned foundation model designed for open-set object detection in agricultural environments. The project evaluates the model's performance under varying levels of training data availability and compares it against state-of-the-art detectors such as DINO, RetinaNet and YOLOv8. Using expert-annotated UAV imagery from sorghum and maize fields, the study demonstrates that Grounding DINO can achieve robust and accurate weed detection even when fine-tuned with only a small number of representative images per crop growth stage.

Given the breadth of experiments conducted in the paper, spanning multiple dataset variants, cross-validation strategies, and inference scenarios, this repository focuses on providing the core training and inference framework necessary to reproduce the key results. Specifically, it includes the preprocessing pipeline to generate 512×512 patches from the original drone imagery following the experimental setup described in the paper, as well as the installation procedure and config files to reproduce the training of Grounding DINO, DINO, RetinaNet and YOLOv8. The config files contain hyperparameters optimized using all available training data following a rigorous cross-validation strategy as described in the paper. For a comprehensive understanding of the full experimental setup, inference across multiple scenarios, and in-depth quantitative analysis, please refer to the paper directly.

A demo inference notebook is also provided, which performs inference on a sample test image using pretrained model checkpoints available on Hugging Face, along with a step-by-step visualization of predictions against ground truth annotations. For the in-depth inference on the full held-out test set and its subsets, two evaluation metrics are used: AP is computed using [Padilla's Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) repository and F1 score is computed using custom-defined functions provided in `inference/compute_f1.py`.


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

## Dataset Preparation

The EWIS dataset used in this paper is publicly available on Mendeley Data: https://data.mendeley.com/datasets/6j5pxgf437/1. The dataset as published does not include train/test splits or growth stage stratification. Follow the steps below to prepare the dataset for training.

### Sample Dataset

A small sample of the dataset is provided in `sample_ewis_data/` in the repository root. This can be used to verify your setup and test the training pipeline before running on the full dataset. The sample data is already in mmdetection-compatible format and the paths are pre-configured in all config files.

---

### Step 1 — Download the Dataset

Download the dataset from Mendeley Data and place it in your local machine.

---

### Step 2 — Categorize the Dataset

The downloaded dataset is not categorized into train/test splits or stratified by growth stages. Refer to `preprocessing/readme.md` in the repository for the categorization details used in this project and organize the data into the respective folders accordingly.
```
uav_weed_detection/
└── raw_data/
    ├── train/
    │   ├── images/
    │   │   ├── BBCH_12/
    │   │   ├── BBCH_13/
    │   │   └── ...
    │   └── annotations/
    │       ├── BBCH_12/
    │       ├── BBCH_13/
    │       └── ...
    └── test/
        ├── images/
        │   ├── BBCH_13/
        │   ├── BBCH_14/
        │   └── ...
        └── annotations/
            ├── BBCH_13/
            ├── BBCH_14/
            └── ...
```

> **Note:** The annotations for 10 additional test set images are not included in the Mendeley Data publication. These annotations are available in the repository under `annotations_additional_images/`.

---

### Step 3 — Preprocess the Full Dataset

Run `create_patches_generic.py` to generate 512×512 patches from the original drone images. The script splits the data into train, val and test sets and saves the patches in mmdetection-compatible format under `uav_weed_detection/ewis_data/`. Before running, update the path variables at the top of the script to point to your local `raw_data/` directory.
```bash
python preprocessing/create_patches_generic.py
```
After running the script, the following structure will be created:
```
uav_weed_detection/
└── ewis_data/
    ├── train_images/
    ├── val_images/
    ├── test_images/
    ├── train.json
    ├── val.json
    ├── test.json
    ├── train.txt
    ├── val.txt
    └── test.txt
```


> **Note:** This preprocessing setup corresponds to the final retraining of the model as described in the paper, performed after finding the best hyperparameters via cross-validation. The config files in the respective folders contain the optimized hyperparameters and a fixed number of training epochs - there is no validation set as training runs for a fixed number of epochs. To maintain training pipeline compatibility, the test set is also copied to the `val_images/` folder. The annotation txt files are generated automatically at the end of the script.

---

### Step 4 — Reproduce Cross-Validation Experiments

To reproduce the cross-validation strategies described in the paper, the following scripts are provided. Update the path variables at the top of each script before running.

**Data Efficiency Experiment** — `create_patches_data_efficiency.py` was used to create patches for the full, half, quarter and single image per growth stage training dataset variants following the same 4-fold CV protocol as described in the paper. For the half, quarter and single variants, `sample_dataset.py` was first used to sample the original images before patching:
```bash
python preprocessing/sample_dataset.py        # set VARIANT = "half", "quarter" or "single"
python preprocessing/create_patches_data_efficiency.py
```

**Progressive Growth Stage Experiment** — `create_patches_progressive_growth_stage.py` stratifies the patches based on the progressive growth stage experimental setup described in the paper:
```bash
python preprocessing/create_patches_progressive_growth_stage.py
```

Refer to the paper for a detailed explanation of the stratification strategy used in each cross-validation experiment.

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

To perform inference on a sample image from the testset, use the demo inference notebook available at `inference/demo_inference_notebook`. The notebook provides step-by-step instructions to generate and visualize predictions using pretrained model checkpoints, which can be downloaded from Hugging Face.


If you encounter any issues with the code or reproducibility, please open a [GitHub issue](https://github.com/smAIL-WS/uav_weed_detection/issues).

## Citing this work
This work is currently under review:
```
Towards Data-efficient Weed Detection via Fine-Tuning Grounding DINO
Harshavardhan Subramanian, Nikita Genze, Heinz Bernhardt, Dominik G. Grimm, Florian Haselbeck
```