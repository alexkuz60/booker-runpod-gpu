# ============================================================================
# Booker GPU Server — RunPod Serverless image (v6)
# ----------------------------------------------------------------------------
# v6 changes vs v5:
#   - Base image FIX: runpod/base:0.6.2-cuda12.8.1 НЕ существует на Docker Hub.
#     Берём канонический nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 — гарантированно
#     существует, поддерживает RTX 6000 Pro (Blackwell, sm_120), даёт стабильный
#     Python 3.11.x из deadsnakes (не rc1).
#   - Реальные доступные runpod/base с cuda12.8 — только 0.7.2-dev-* (6 GB),
#     слишком тяжёлые; nvidia/cuda devel = ~3 GB и контролируемая среда.
# ============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Python 3.11 stable (deadsnakes), git, build essentials, ffmpeg ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        build-essential ffmpeg libsndfile1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ---- Sanity check: Python version stable, not rc ----
RUN python --version && python -c "import sys; assert sys.version_info >= (3,11,5), f'Python too old: {sys.version}'; print('Python OK:', sys.version)"

WORKDIR /app

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

# ---- Sanity checks: torch+cuda, transformers (Higgs), runpod ----
RUN python -c "import torch; print('torch:', torch.__version__, 'cuda build:', torch.version.cuda)"
RUN python -c "from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel, HiggsAudioV2TokenizerModel; import transformers, huggingface_hub, tokenizers; print('transformers:', transformers.__version__, 'hub:', huggingface_hub.__version__, 'tokenizers:', tokenizers.__version__)"
RUN python -c "import runpod; print('runpod:', runpod.__version__)"

# ---- App code ----
COPY . /app

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
