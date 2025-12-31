FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.1
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from main branch (required for Flux2Pipeline)
RUN pip3 install git+https://github.com/huggingface/diffusers.git

# Install other dependencies
RUN pip3 install \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    bitsandbytes \
    huggingface_hub \
    runpod \
    Pillow \
    requests

WORKDIR /

# Copy handler
COPY rp_handler.py /rp_handler.py

# Start the handler
CMD ["python3", "-u", "/rp_handler.py"]
