# Dockerfile — Booker GPU backend for RunPod Serverless
# CUDA 12.1 + cuDNN 8 (matches torch 2.4.0 cu121), Python 3.11.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf-cache \
    OMNIVOICE_CACHE_DIR=/runpod-volume/hf-cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --upgrade pip

WORKDIR /app

# Torch + cuda 12.1 wheels
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchaudio==2.4.0

# RunPod SDK + helpers
RUN pip install runpod==1.7.0 numpy

# OmniVoice + omnivoice-server (variant B uses OmniVoice directly; server
# package included so its services.inference adapter is importable as a
# fallback if you ever want to switch back to variant A)
RUN pip install \
        "git+https://github.com/k2-fsa/OmniVoice.git" \
        "git+https://github.com/maemreyo/omnivoice-server.git@main"

# Handler
COPY handler.py /app/handler.py

# Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
