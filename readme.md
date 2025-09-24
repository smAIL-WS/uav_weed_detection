# Repository Overview
This repository presents implementations for fine-tuning **Grounding DINO**, and training **Retinanet** and **Yolov8** on a custom dataset, based on the **MMDetection** framework.

## Installation

### Grounding DINO / Retinanet (MMDetection)
```
conda create -n mmdet_env python=3.10
conda activate mmdet_env

conda install pytorch torchvision -c pytorch

pip install -U openmim
pip install sympy==1.13.1 fsspec wandb optuna scikit-image
mim install mmengine
mim install "mmcv==2.1.0"


cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.

# GDino specific installations
pip install -r requirements/multimodal.txt
```


### Yolov8
```
conda create -n mmyolo_env python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo_env


pip install openmim
pip install albumentations==1.3.1 wandb regex optuna
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"

cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

## Dataset preparation

1. Ensure that the cropped dataset for training, validation and testing is present in  `{mmdetection,mmyolo}/data/ewis/{train_images, val_images, test_images}/` respectively. 
2. Ensure that annotations for the images in training, validation and testset is present in .json format. For example, `{mmdetection,mmyolo}/data/ewis/{train,val,test}.json`. 
3. Additionally, include three `.txt` files listing image names for each split. For example, `{mmdetection,mmyolo}/data/ewis/{train,val,test}.txt`
4. The sample training, validation and testset required to run the training process can be found in `sample_ewis_data/`.


## Training models

1. Activate the appropriate conda environment.

```
# For Grounding DINO or Retinanet
conda activate mmdet_env

# For Yolov8
conda activate mmyolo
```

2. Train models
```
# Fine-tune Grounding DINO
python mmdetection/tools/train.py mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_crop_weed.py

# Train Retinanet from scratch
python mmdetection/tools/train.py mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco_crop_weed.py

# Train Yolov8 from scratch
python mmyolo/tools/train.py mmyolo/configs/yolov8/yolov8_s_fast_1xb12-40e_crop_weed.py
```

## Inference on held-out testset

1. To perform inference on an image from held-out testset based on best model, update the following variables `config_path, checkpoint_path, test_image_path, pred_save_path` in the appropriate inference scripts `inference/inference_groundingDino.py` and run the below command,

```
# For inference based on fine-tuned Grounding DINO
python inference_groundingDino.py

# For inference based on trained Retinanet
python inference_retinaney.py

# For inference based on trained Yolov8
python inference_yolov8.py
```

The predictions for the test image based on all the three models is now saved in `inference/predictions` in .pt format. 

## Visualization
To visualize along with respective ground truth annotations and models for the same image, update the following variables `pred_bboxes, pred_scores, pred_labels, gt_file, test_image_path` in the script `inference/prediction_visualization.py` and run the below command,

```
python prediction_visualization.py
```
The plots are saved in `inference/visualization`
