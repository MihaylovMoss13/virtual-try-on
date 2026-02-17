# FASHN VTON v1.5 Docker Image (GPU Optimized)
# Uses NVIDIA CUDA base image to ensure library compatibility

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python and system dependencies
# libgl1/libglib2.0 are for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python if not present
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Install dependencies
# 1. Upgrade pip
# 2. Install PyTorch with CUDA support (default from PyPI is usually CUDA 12.1 compatible)
# 3. Install project (will leverage onnxruntime-gpu from pyproject.toml)
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision && \
    pip3 install -e .

# Create directories for weights and outputs
RUN mkdir -p /app/weights /app/outputs /app/hf_cache

# Expose port for potential API server
EXPOSE 8000

# Default command
CMD ["python3", "examples/basic_inference.py", "--weights-dir", "/app/weights", "--help"]
