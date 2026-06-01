# Dockerfile v5 — Booker GPU backend for RunPod Serverless
# Base: RunPod official image (CUDA 12.8 + cuDNN 9 + stable Python 3.11)
#
# Зачем v5:
#  - CUDA 12.8 (вместо 12.1) — поддержка Blackwell (RTX 5090 / RTX 6000 Pro, sm_120).
#  - PyTorch 2.6 + cu128 wheels (под 12.1 был torch 2.4, на Blackwell не работает).
#  - Стабильный Python 3.11.x из runpod/base (а не 3.11.0rc1 из ubuntu22.04 nvidia-image).
#  - runpod SDK явно зафиксирован на 1.9.0 (раньше pip ставил 1.7).
#  - transformers 5.8.1 оставляем — содержит HiggsAudioV2TokenizerModel.

FROM runpod/base:0.6.2-cuda12.8.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf-cache \
    OMNIVOICE_CACHE_DIR=/runpod-volume/hf-cache

# runpod/base уже содержит python3.11 + pip + ffmpeg + git, но проверим/обновим
RUN python3.11 -m pip install --upgrade pip \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /app

# --- Torch stack под CUDA 12.8 (Blackwell-ready) ---
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.6.0 torchaudio==2.6.0

# --- RunPod SDK + численная база ---
RUN pip install runpod==1.9.0 numpy scipy soundfile httpx

# --- OmniVoice + omnivoice-server (из git, без кэша) ---
RUN pip install \
        "git+https://github.com/k2-fsa/OmniVoice.git" \
        "git+https://github.com/maemreyo/omnivoice-server.git@main"

# --- Жёстко фиксируем transformers 5.8.1 (HiggsAudioV2TokenizerModel внутри).
#     Без --no-deps: pip сам подтянет совместимые hub/tokenizers/safetensors.
RUN pip install --upgrade "transformers==5.8.1"

# --- Sanity checks: CUDA доступна, импорты не падают, версии совпадают ---
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"
RUN python -c "from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel, HiggsAudioV2TokenizerModel; import transformers, huggingface_hub, tokenizers; print('transformers', transformers.__version__, 'hub', huggingface_hub.__version__, 'tokenizers', tokenizers.__version__)"
RUN python -c "import runpod; print('runpod', runpod.__version__)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
