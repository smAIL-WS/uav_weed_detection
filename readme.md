The original paper can be accessed here: [link].

# Repository Overview

This repository contains the codebase for our research on **Assessing Training Data Efficiency of Fine-Tuned Object Detection Models for Drone-Based Crop and Weed Detection**. This study assesses the training data efficiency of four state-of-the-art object detection architectures (RetinaNet, YOLOv26, DINO, and Grounding DINO) from two perspectives: data volume efficiency, by progressively reducing annotated training images, and growth stage distribution efficiency, by progressively reducing the growth stages present in training.

Given the breadth of experiments conducted in the paper, spanning multiple dataset variants, cross-validation strategies, and inference scenarios, this repository focuses on providing the core training and inference framework. Specifically, it includes the preprocessing pipeline to generate 512×512 patches from the original drone imagery following the experimental setup described in the paper, as well as the installation procedure and config files to reproduce the training of RetinaNet, YOLOv26, DINO and Grounding DINO. The config files contain hyperparameters optimized using all available training data following a rigorous cross-validation strategy as described in the paper. For a comprehensive understanding of the full experimental setup, inference across multiple scenarios, and in-depth quantitative analysis, please refer to the paper directly.

A demo inference notebook is also provided, which performs inference on a sample test image using pretrained model checkpoints available on Hugging Face, along with a step-by-step visualization of predictions against ground truth annotations. For the in-depth inference on the full held-out test set and its subsets, two evaluation metrics are used: AP is computed using [Padilla's Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) repository and F1 score is computed using custom-defined functions provided in `inference/compute_f1.py`.


## Installation and Environment Setup

### RetinaNet, DINO, and Grounding DINO — Docker

We provide a pre-built Docker image on Docker Hub to reproduce experiments for RetinaNet, DINO, and Grounding DINO.

| Image | Models |
|---|---|
| `hswt555har/mmdetection-models:v1.1` | Grounding DINO, RetinaNet, DINO |

#### Pull Docker Image
```bash
docker pull hswt555har/mmdetection-models:v1.1
```

#### Verify Image
```bash
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
```

---

### YOLOv26 — Conda Environment

YOLOv26 is implemented via the official [Ultralytics](https://docs.ultralytics.com) framework and does not require Docker. Set up a dedicated conda environment as follows:

```bash
conda create -n yolo26_env python=3.11 -y
conda activate yolo26_env
pip install ultralytics optuna pyyaml
```

#### Verify Installation
```bash
python -c "
import torch
from ultralytics import YOLO
print('PyTorch  :', torch.__version__)
print('CUDA     :', torch.cuda.is_available())
print('GPU      :', torch.cuda.get_device_name(0))
model = YOLO('yolo26l.pt')
print('YOLOv26l : OK')
"
```

---

## Clone the Repository

Clone the repository and navigate to the root before running any command:
```bash
git clone https://github.com/smAIL-WS/uav_weed_detection.git
cd uav_weed_detection
```

---

## Dataset Preparation

The EWIS dataset used in this paper is publicly available on Mendeley Data: https://data.mendeley.com/datasets/6j5pxgf437/1. The dataset as published does not include train/test splits or growth stage stratification. Follow the steps below to prepare the dataset for training.

### Sample Dataset

A small sample of the dataset is provided in `sample_ewis_data/` in the repository root. This can be used to verify your setup and test the training pipeline before running on the full dataset. The sample data is already in the required format and the paths are pre-configured in all config files.

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

Run `create_patches_generic.py` to generate 512×512 patches from the original drone images. The script splits the data into train, val and test sets and saves the patches in the required format under `uav_weed_detection/ewis_data/`. Before running, update the path variables at the top of the script to point to your local `raw_data/` directory.
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

> **Note:** This preprocessing setup corresponds to the final retraining of the model as described in the paper, performed after finding the best hyperparameters via cross-validation. The config files in the respective folders contain the optimized hyperparameters and a fixed number of training epochs — there is no validation set as training runs for a fixed number of epochs. To maintain training pipeline compatibility, the test set is also copied to the `val_images/` folder. The annotation txt files are generated automatically at the end of the script.

---

### Step 4 — Reproduce Cross-Validation Experiments

To reproduce the cross-validation strategies described in the paper, the following scripts are provided. Update the path variables at the top of each script before running.

**Progressive Training Data Reduction Experiment** — `create_patches_PTDR.py` was used to create patches for the full, half, quarter and single image per growth stage training dataset variants following the same 4-fold CV protocol as described in the paper. For the half, quarter and single variants, `sample_dataset.py` was first used to sample the original images before patching:
```bash
python preprocessing/sample_dataset.py        # set VARIANT = "half", "quarter" or "single"
python preprocessing/create_patches_PTDR.py
```

**Progressive Growth Stage Reduction Experiment** — `create_patches_PGSR.py` stratifies the patches based on the progressive growth stage experimental setup described in the paper:
```bash
python preprocessing/create_patches_PGSR.py
```

Refer to the paper for a detailed explanation of the stratification strategy used in each cross-validation experiment.

---

### Step 5 — Update Config Files

Once the dataset is prepared, replace the sample dataset path with the full dataset path in the configuration files.

**MMDetection models** — update `data_root` in:
- `mmdetection/configs/grounding_dino/gd_full_dataset.py`
- `mmdetection/configs/retinanet/rn_full_dataset.py`
- `mmdetection/configs/dino/dino__full_dataset.py`

> **Note:** All YOLOv26-related files (training scripts, config, preprocessing) are located under the `yolov26/` directory in the repository root.

```python
# Replace this (sample dataset path)
data_root = '/workspace/sample_ewis_data/'

# With this (full preprocessed dataset path)
data_root = '/workspace/ewis_data/'
```

**YOLOv26** — update `base_dir` and `root` in `yolov26/configs/pipeline_config.yaml`:
```yaml
project:
  base_dir: "/path/to/your/project"

dataset:
  root: "/path/to/ewis_data"
```

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
           /workspace/mmdetection/configs/retinanet/rn_full_dataset.py
```

### DINO
```bash
docker run --gpus all \
    --shm-size=8g \
    -e WANDB_MODE=disabled \
    -v $(pwd):/workspace \
    hswt555har/mmdetection-models:v1.1 \
    python /workspace/mmdetection/tools/train.py \
           /workspace/mmdetection/configs/dino/dino_full_dataset.py
```

### YOLOv26

All YOLOv26 files are located under the `yolov26/` directory. Activate the conda environment before running any command.

#### Dataset Format

YOLOv26 expects data in **YOLO label format** (not COCO JSON). Each fold directory must follow this structure:

```
yolov26/data/<variant>/
├── fold_1/
│   ├── images/
│   │   ├── train/        ← 512×512 patch images (.png)
│   │   └── val/          ← 512×512 patch images (.png)
│   └── labels/
│       ├── train/        ← YOLO .txt labels (class cx cy w h, normalised)
│       └── val/
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/
```

The training script (`yolov26/scripts/train.py`) automatically generates `train_paths.txt` and `val_paths.txt` inside each fold directory at runtime — these are temporary path list files used to construct the Ultralytics data YAML and do not need to be created manually.

> **Note:** Patch images are normalised per-tile (min-max to [0, 255]) and saved as RGB. During inference on full drone images, the same per-tile normalisation and BGR→RGB conversion must be applied before passing tiles to the model, since Ultralytics reads saved patches via PIL (RGB) but numpy arrays via OpenCV (BGR).

#### Hyperparameter Optimisation (Cross-Validation)
```bash
conda activate yolo26_env
cd yolov26
python scripts/train.py \
    --variant full_dataset \
    --config  configs/pipeline_config.yaml
```

Replace `full_dataset` with any dataset variant defined in `pipeline_config.yaml`:
```bash
python scripts/train.py --variant half_dataset_v1    --config configs/pipeline_config.yaml
python scripts/train.py --variant quarter_dataset_v1 --config configs/pipeline_config.yaml
python scripts/train.py --variant single_image_per_growth_stage_v1 --config configs/pipeline_config.yaml
```

The Optuna study is saved to `results/<variant>/optuna/study.db` and is resumable if interrupted.

#### Final Retraining with Best Hyperparameters
Once hyperparameter optimisation is complete, retrain the model on the combined training and validation data:
```bash
python scripts/retrain.py \
    --variant full_dataset \
    --config  configs/pipeline_config.yaml
```

The retrained checkpoint is saved to `results/<variant>/final_retrain/`.

> **Note:** All YOLOv26 training outputs including checkpoints, Optuna studies, and logs are saved to `yolov26/results/`.

---

## Inference on Held-out Testset

To perform inference on a sample image from the testset, use the demo inference notebook available at `inference/demo_inference_notebook`. The notebook provides step-by-step instructions to generate and visualize predictions using pretrained model checkpoints, which can be downloaded from Hugging Face.

Run the notebook inside the Docker container (MMDetection models):
```bash
docker run --gpus all -p 8888:8888 \
  -v $(pwd):/workspace \
  hswt555har/mmdetection-models:v1.1 \
  jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

For YOLOv26 inference, activate the conda environment and run the notebook directly:
```bash
conda activate yolo26_env
jupyter lab inference/demo_inference_notebook
```

---

If you encounter any issues with the code or reproducibility, please open a [GitHub issue](https://github.com/smAIL-WS/uav_weed_detection/issues).

## Citing this work
This work is currently under review:
```
Assessing Training Data Efficiency of Fine-Tuned Object Detection Models for Drone-Based Crop and Weed Detection
Harshavardhan Subramanian, Nikita Genze, Heinz Bernhardt, Dominik G. Grimm, Florian Haselbeck
```