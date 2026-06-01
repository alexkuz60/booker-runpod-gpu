# Dockerfile v3 — Booker GPU backend for RunPod Serverless
# CUDA 12.1 + cuDNN 8, Python 3.11.
#
# FIX v3: при --force-reinstall --no-deps transformers 4.46.3 остаётся с
# huggingface_hub==1.17.0 (его подтянул transformers 5.x). 4.46 несовместим
# с hub>=1.0 — AutoFeatureExtractor падает ещё на импорте. Принудительно
# откатываем huggingface_hub до 0.26.x и tokenizers до 0.20.x В ОДНОЙ
# pip-команде, чтобы резолвер увидел совместимый набор.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf-cache \
    OMNIVOICE_CACHE_DIR=/runpod-volume/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python -m pip install --upgrade pip

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0 torchaudio==2.4.0

RUN pip install runpod==1.7.0 numpy

# OmniVoice + omnivoice-server (тянут transformers 5.x и hub 1.x как deps)
RUN pip install \
        "git+https://github.com/k2-fsa/OmniVoice.git" \
        "git+https://github.com/maemreyo/omnivoice-server.git@main"

# КРИТИЧНО: откатываем весь HF-стек к 4.x-совместимому набору ОДНОЙ командой.
# huggingface_hub<1.0 обязателен — иначе transformers 4.46 не стартует.
RUN pip install --force-reinstall --no-deps \
        "transformers==4.46.3" \
        "tokenizers>=0.20,<0.21" \
        "huggingface_hub>=0.26,<1.0" \
        "safetensors>=0.4.5"

# Sanity check
RUN python -c "from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel; import transformers, huggingface_hub; print('transformers', transformers.__version__, 'hub', huggingface_hub.__version__)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
