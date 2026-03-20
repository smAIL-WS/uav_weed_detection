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

# 2. Install mmengine first
RUN pip install --no-cache-dir mmengine==0.10.7

# 3. Install mmcv FROM SOURCE — compiles CUDA kernels for your GPU
# This replaces: pip install mmcv==2.1.0
RUN pip install --no-cache-dir -U openmim && \
    mim install mmcv==2.1.0

# OR alternatively build from source manually:
# RUN git clone https://github.com/open-mmlab/mmcv.git && \
#     cd mmcv && git checkout v2.1.0 && \
#     MMCV_WITH_OPS=1 pip install -e . && \
#     cd .. && rm -rf mmcv

# 4. mmdet pinned to exact dev commit
RUN pip install --no-cache-dir \
    "git+https://github.com/open-mmlab/mmdetection.git@cfd5d3a985b0249de009b67d04f37263e11cdf3d#egg=mmdet"

# 5. Grounding DINO specific deps
RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    torchinfo==1.8.0 \
    einops \
    timm \
    fairscale

# 6. Remaining cleaned dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 7. Fix git safe directory warning
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace