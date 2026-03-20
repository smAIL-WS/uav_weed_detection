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

# 2. Install mmengine
RUN pip install --no-cache-dir mmengine==0.10.7

# 3. Set GPU architectures for all common GPUs
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1

# 4. Build mmcv from source
# Split into separate RUN steps so each step fails loudly if something goes wrong
RUN git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    git checkout v2.1.0

RUN cd mmcv && \
    pip install --no-cache-dir -r requirements/optional.txt

RUN cd mmcv && \
    pip install --no-cache-dir .

# 5. Verify mmcv installed correctly before proceeding
# This will cause the build to FAIL LOUDLY if mmcv is not installed
RUN python -c "import mmcv; print('mmcv installed successfully:', mmcv.__version__)"

# 6. Clean up source only after verifying installation
RUN rm -rf mmcv

# 7. mmdet pinned to exact dev commit
RUN pip install --no-cache-dir \
    "git+https://github.com/open-mmlab/mmdetection.git@cfd5d3a985b0249de009b67d04f37263e11cdf3d#egg=mmdet"

# 8. Grounding DINO specific deps
RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    torchinfo==1.8.0 \
    einops \
    timm \
    fairscale

# 9. Remaining cleaned dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 10. Fix git safe directory warning
RUN git config --global --add safe.directory /workspace

WORKDIR /workspace