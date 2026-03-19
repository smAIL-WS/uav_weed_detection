# docker/mmdet/Dockerfile
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget ninja-build \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1-mesa-glx ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# 1. Core torch stack
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 2. Core mm stack
RUN pip install --no-cache-dir \
    mmengine==0.10.7 \
    mmcv==2.1.0

# 3. mmdet pinned to exact dev commit
RUN pip install --no-cache-dir \
    "git+https://github.com/open-mmlab/mmdetection.git@cfd5d3a985b0249de009b67d04f37263e11cdf3d#egg=mmdet"

# 4. Grounding DINO specific deps
RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    torchinfo==1.8.0 \
    einops \
    timm

# 5. Remaining cleaned dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace