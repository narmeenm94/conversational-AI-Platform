FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    git+https://github.com/Deathdadev/Orpheus-Speech-PyPi \
    accelerate

COPY . .

# Model caches — mount persistent volumes here so models survive restarts
ENV HF_HOME=/models/huggingface
VOLUME ["/models/huggingface"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import websockets, asyncio; asyncio.run(websockets.connect('ws://localhost:8765').__aenter__())" || exit 1

CMD ["python", "main.py"]
