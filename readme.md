This repository includes implementation of finetuning Grounding Dino, training retinanet and yolov8 on custom dataset based on MMDetection framework.

### Installation procedure
```
mkdir groundingDino
conda create -n groundingDino_env python=3.10
conda activate groundingDino_env

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