FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Системные зависимости для аудио
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build-time smoke test — если omnivoice не импортируется, билд падает СРАЗУ
RUN python -c "import omnivoice; import torch; print('OK omnivoice + torch', torch.__version__)"

# Handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
