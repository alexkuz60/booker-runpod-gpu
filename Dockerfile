# Dockerfile — Booker GPU backend on RunPod Serverless
# ============================================================================
# Base: NVIDIA CUDA 12.1 + cuDNN 8 (matches torch 2.4.0 cu121 wheels).
# Installs Python 3.11, OmniVoice + omnivoice-server, runpod SDK.
# HF weights are cached at /runpod-volume/hf-cache (Network Volume in RunPod).
# ============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/hf-cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-cache \
    TORCH_HOME=/runpod-volume/torch-cache

# ── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git ffmpeg libsndfile1 ca-certificates curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ─────────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# OmniVoice + server (from upstream git, vanilla — PCM_16 fix is in main)
RUN pip install \
        "git+https://github.com/k2-fsa/OmniVoice.git" \
        "git+https://github.com/maemreyo/omnivoice-server.git@main"

# ── Handler ─────────────────────────────────────────────────────────────────
COPY handler.py /app/handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
