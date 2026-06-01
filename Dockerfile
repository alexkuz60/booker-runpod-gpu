# Dockerfile v2 — Booker GPU backend for RunPod Serverless
# CUDA 12.1 + cuDNN 8 (matches torch 2.4.0 cu121), Python 3.11.
#
# FIX v2: pin transformers<5.
# OmniVoice/omnivoice-server were written against transformers 4.x. In 4.46+
# AutoFeatureExtractor lives in transformers.models.auto.feature_extraction_auto
# and is re-exported from the top-level package. In transformers 5.x the lazy
# import map was reorganized and the bare string "AutoFeatureExtractor" no
# longer resolves through the legacy path that OmniVoice uses, producing:
#   ModuleNotFoundError: Could not import module 'AutoFeatureExtractor'.
# Installing transformers==4.46.3 AFTER OmniVoice forces a clean downgrade.

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

# OmniVoice + omnivoice-server
RUN pip install \
        "git+https://github.com/k2-fsa/OmniVoice.git" \
        "git+https://github.com/maemreyo/omnivoice-server.git@main"

# CRITICAL: downgrade transformers to a 4.x release compatible with OmniVoice.
# Must run AFTER OmniVoice install (which pulls latest transformers 5.x as a dep).
RUN pip install --force-reinstall --no-deps "transformers==4.46.3" "tokenizers>=0.20,<0.21"

# Sanity check: fail the build early if the import OmniVoice does is broken.
RUN python -c "from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel; print('transformers ok:', __import__('transformers').__version__)"

# Handler
COPY handler.py /app/handler.py

# Serverless entrypoint
CMD ["python", "-u", "/app/handler.py"]
