# booker-runpod-gpu — Dockerfile v7 (CUDA 12.4, RTX 4090 compatible)
#
# Изменения v6 → v7:
#   • Базовый образ понижен с CUDA 12.8 → 12.4
#     Причина: nvidia-container-cli на хостах RunPod (US-NC-1, RTX 4090)
#     отбраковывает контейнеры с cuda>=12.8 — драйвер хоста ещё не обновлён.
#     В логах v6: "unsatisfied condition: cuda>=12.8, please update your driver".
#     CUDA 12.4 поддерживает RTX 4090 (sm_89) и совместима со всеми текущими
#     драйверами RunPod. Для RTX 6000 Pro Blackwell вернёмся к 12.8 позже,
#     когда RunPod раскатит новые драйверы.
#   • torch 2.6.0 + cu124 (вместо cu128).
#   • Python 3.11.x stable из deadsnakes PPA — без изменений.
#   • Sanity-checks — без изменений (torch/transformers/runpod).

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/runpod-volume/hf-cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache

# Python 3.11 stable (НЕ rc1) из deadsnakes
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        curl git build-essential && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала requirements — лучше кэшируется
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Sanity: torch + CUDA visible (build-time check)
RUN python -c "import sys; assert sys.version_info >= (3,11,5), f'Python too old: {sys.version}'; print(f'[sanity] python={sys.version.split()[0]}')"
RUN python -c "import torch; print(f'[sanity] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_version={torch.version.cuda}')"
RUN python -c "from transformers import HiggsAudioV2TokenizerModel; print('[sanity] transformers HiggsAudioV2TokenizerModel import OK')"
RUN python -c "import runpod; print(f'[sanity] runpod=={runpod.__version__}')"

# Handler и всё остальное
COPY . .

CMD ["python", "-u", "handler.py"]

# Rebuild trigger: 2026-06-01 (force CUDA 12.4 base)
