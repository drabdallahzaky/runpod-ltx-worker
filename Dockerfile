FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

ENV HF_HOME=/runpod-volume/huggingface_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface_cache
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/app/src/handler.py"]
