FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    ffmpeg git wget curl && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

# Install PyTorch 2.7 with CUDA 12.8
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install LTX packages
RUN pip install --no-cache-dir ltx-core ltx-pipelines

# Install RunPod SDK and other deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY src/ /app/src/

# Environment for model caching
ENV HF_HOME=/runpod-volume/huggingface_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface_cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python", "-u", "/app/src/handler.py"]
