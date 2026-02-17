# FASHN VTON v1.5 Docker Image
# Supports both GPU (NVIDIA) and CPU inference

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache

WORKDIR /app

# Install system dependencies (libgl1 replaces deprecated libgl1-mesa-glx in Debian Trixie)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Install PyTorch CPU version first, then modify pyproject.toml to use CPU onnxruntime
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    sed -i 's/onnxruntime-gpu/onnxruntime/g' pyproject.toml && \
    pip install -e .

# Create directories for weights and outputs
RUN mkdir -p /app/weights /app/outputs /app/hf_cache

# Download weights at build time (optional - can be mounted instead)
# RUN python scripts/download_weights.py --weights-dir /app/weights

# Expose port for potential API server
EXPOSE 8000

# Default command - run inference example
CMD ["python", "examples/basic_inference.py", "--weights-dir", "/app/weights", "--help"]
