# Booker RunPod GPU — minimal diagnostic image (Step 1)
# Goal: prove that the image builds, OmniVoice imports, GPU is visible.
# No TTS logic yet — that comes in Step 4.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# git + ca-certificates нужны для pip install из GitHub-репозиториев.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — это позволяет Docker'у кэшировать слой при правках handler.py.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Smoke-test прямо в билде: если omnivoice не импортируется — билд падает СРАЗУ,
# а не молча выкатывается сломанный образ.
RUN python -c "import omnivoice; print('[BUILD] omnivoice import OK')"
RUN python -c "import torch; print('[BUILD] torch', torch.__version__, 'cuda', torch.version.cuda)"

# Handler и всё остальное
COPY . .

CMD ["python", "-u", "handler.py"]
