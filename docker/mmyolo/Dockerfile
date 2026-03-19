# docker/mmyolo/Dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget ninja-build \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1-mesa-glx ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Fix: pin typing-extensions BEFORE torch install
# Latest typing-extensions requires Python >=3.9, but base image has 3.8.10
RUN pip install --no-cache-dir "typing-extensions>=3.10,<4.6"

# 1. Core torch stack
RUN pip install --no-cache-dir \
    torch==1.10.1+cu113 \
    torchvision==0.11.2+cu113 \
    --index-url https://download.pytorch.org/whl/cu113

# 2. Core mm stack
RUN pip install --no-cache-dir \
    mmengine==0.10.7 \
    mmcv==2.0.1 \
    mmdet==3.3.0

# 3. Custom mmyolo fork — updated to latest commit
RUN pip install --no-cache-dir \
    "git+https://github.com/smAIL-WS/uav_weed_detection.git@942c9bf446edec41373d45b670ad27a88f755ce0#egg=mmyolo&subdirectory=mmyolo"

# 4. Remaining cleaned dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace