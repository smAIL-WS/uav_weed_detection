This repository includes implementation of finetuning Grounding Dino, training retinanet and yolov8 on custom dataset based on MMDetection framework.

### Installation procedure
```
conda create -n mmdet_env python=3.10
conda activate mmdet_env

conda install pytorch torchvision -c pytorch

pip install sympy==1.13.1 fsspec wandb optuna
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"


git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.

# GDino specific installations
pip install -r requirements/multimodal.txt
```


#### Installation procedure for YOLOv8
```
conda create -n mmyolo_env python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo_env

pip install albumentations==1.3.1 wandb regex optuna
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

#### Preprocessing steps

1. Ensure that the cropped dataset for training, validation and testing is present in the folder `mmdetection/data/ewis/train_images/`,  `mmdetection/data/ewis/val_images/` and `mmdetection/data/ewis/test_images/` respectively. 
2. Ensure that annotations for the images in training, validation and testset is also present in the folder `mmdetection/data/ewis` in .json format. For example, `mmdetection/data/ewis/{train,val,test}.json`. It is also important to have three more .txt files which contains the image names of the all the three datasets. For example, `mmdetection/data/ewis/{train,val,test}.txt`
3. Image pre-processing scripts from `preprocessing/` folder can be used which creates cropped datasets based on CV splits used in the original paper (`preprocessing/create_cropped_dataset.py`), create samples of training set for data efficiency experiments (`preprocessing/create_samples.py`) and creates image names for .txt files (`preprocessing/create_annotation_txt_file.py`). 

### Training models

1. Activate the respective conda environment.

```
# To fine-tune Grounding DINO or to train Retinanet from scratch
conda activate mmdet_env

# To train Yolov8 from scratch
conda activate mmyolo
```

2. To fine-tune Grounding DINO
```
python mmdetection/tools/train.py mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_crop_weed.py

```

3. To train Retinanet from scratch
```
python mmdetection/tools/train.py mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco_crop_weed.py

```

4. To train Yolov8 from scratch
```
python mmyolo/tools/train.py mmyolo/configs/yolov8/yolov8_s_fast_1xb12-40e_crop_weed.py

```

### Inference on held-out testset
