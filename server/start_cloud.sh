#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Quick-start script for RunPod — survives pod restarts.
# Everything persistent lives under /workspace/.
#
# First run:  takes ~5-10 min (installs packages)
# Restarts:   takes ~30 seconds (skips installs)
#
# Usage:  bash /workspace/conversational-AI-Platform/server/start_cloud.sh
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

REPO_DIR="/workspace/conversational-AI-Platform"
VENV_DIR="/workspace/venv"
SCRIPT_DIR="$REPO_DIR/server"

# HF_TOKEN is read from /workspace/.hf_token (create once: echo "your_token" > /workspace/.hf_token)
if [[ -f /workspace/.hf_token ]]; then
    export HF_TOKEN=$(cat /workspace/.hf_token | tr -d '[:space:]')
elif [[ -n "${HF_TOKEN:-}" ]]; then
    echo "$HF_TOKEN" > /workspace/.hf_token
else
    echo "ERROR: No HF_TOKEN found. Run:  echo 'your_token' > /workspace/.hf_token"
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/workspace/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub
export OLLAMA_MODELS=/workspace/ollama_models

log() { echo -e "\n\033[1;36m>>> $*\033[0m"; }

mkdir -p "$HF_HOME" "$OLLAMA_MODELS"

# ── 1. System deps (always needed after restart, very fast) ──
log "Installing system deps..."
apt-get update -qq && apt-get install -y -qq zstd curl ffmpeg > /dev/null 2>&1 || true

# ── 2. Ollama (binary doesn't persist, but install is fast) ──
if ! command -v ollama &> /dev/null; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    log "Ollama already installed."
fi

# ── 3. Python venv on persistent volume (skip if exists) ──
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "Creating Python venv on persistent volume (first time only)..."
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    log "Installing Python packages (this takes ~5 min first time)..."
    pip install --upgrade pip
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r "$SCRIPT_DIR/requirements.txt"
    pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi accelerate
    pip install python-dotenv sounddevice numpy
    pip install vllm snac openai
else
    log "Python venv found — skipping base pip installs."
    source "$VENV_DIR/bin/activate"

    # Always ensure critical TTS packages are present (handles upgrades)
    log "Ensuring vllm + SNAC packages are installed..."
    pip install --quiet vllm snac openai 2>/dev/null || pip install vllm snac openai
fi

# ── 4. HuggingFace authentication (required for gated Orpheus model) ──
log "Authenticating with HuggingFace..."
# Save token to ALL locations transformers/huggingface_hub might check
mkdir -p "$HF_HOME"
echo "$HF_TOKEN" > "$HF_HOME/token"
mkdir -p ~/.cache/huggingface
echo "$HF_TOKEN" > ~/.cache/huggingface/token
mkdir -p ~/.huggingface
echo "$HF_TOKEN" > ~/.huggingface/token

python -c "
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=False)
print('  HuggingFace login successful')
" || echo "  WARNING: huggingface_hub login failed"

# Pre-download the Orpheus TTS model so vllm finds it in cache (skip training artifacts)
ORPHEUS_MODEL="canopylabs/orpheus-3b-0.1-ft"
log "Ensuring Orpheus TTS model is downloaded (inference files only)..."
python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    '$ORPHEUS_MODEL',
    token=os.environ.get('HF_TOKEN'),
    cache_dir='$HUGGINGFACE_HUB_CACHE',
    ignore_patterns=['optimizer*', 'training_args*', 'global_step*', 'rng_state*', 'scheduler*', '*.pt'],
)
print(f'  Model cached at: {path}')
"

# ── 5. .env setup (always refresh from template to pick up changes) ──
log "Setting up .env..."
cp "$SCRIPT_DIR/.env.cloud" "$SCRIPT_DIR/.env"
sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" "$SCRIPT_DIR/.env"
sed -i "s|^HF_HOME=.*|HF_HOME=$HF_HOME|" "$SCRIPT_DIR/.env"

# ── 5. Start Ollama daemon FIRST (so it claims GPU memory before TTS) ──
if ! pgrep -x ollama > /dev/null; then
    log "Starting Ollama daemon..."
    ollama serve &
    sleep 5
fi

# ── 6. Pull LLM model if needed (persists on /workspace) ──
if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    log "Pulling Llama 3.1 8B model..."
    ollama pull llama3.1:8b
else
    log "Llama model already downloaded."
fi

# Warm up Ollama so the model is loaded into GPU VRAM before vllm starts
log "Warming up LLM (loading into GPU)..."
curl -s http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","prompt":"hi","stream":false}' > /dev/null 2>&1 || true
log "LLM warm-up complete. GPU after Ollama:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

# ── 7. Kill any leftover vllm processes from prior runs ──
VLLM_PORT=${VLLM_PORT:-8000}
ORPHEUS_MODEL="canopylabs/orpheus-3b-0.1-ft"

if [[ -f /workspace/vllm.pid ]]; then
    OLD_PID=$(cat /workspace/vllm.pid)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        # Check if it's actually responsive
        if curl -s --max-time 5 "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
            log "vllm TTS server already running and responsive on port $VLLM_PORT."
            VLLM_RUNNING=1
        else
            log "Stale vllm process found (pid=$OLD_PID), killing it..."
            kill "$OLD_PID" 2>/dev/null || true
            sleep 3
            kill -9 "$OLD_PID" 2>/dev/null || true
            VLLM_RUNNING=0
        fi
    else
        VLLM_RUNNING=0
    fi
else
    VLLM_RUNNING=0
fi

if [[ "$VLLM_RUNNING" == "0" ]]; then
    log "Starting vllm TTS server on port $VLLM_PORT..."
    VLLM_USE_V1=0 \
    HF_TOKEN="$HF_TOKEN" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    HF_HOME="$HF_HOME" \
    HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
        "$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$ORPHEUS_MODEL" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.35 \
        --max-model-len 8192 \
        --enforce-eager \
        --port "$VLLM_PORT" \
        > /workspace/vllm.log 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" > /workspace/vllm.pid

    log "Waiting for vllm server to be ready (first load can take 3-5 min)..."
    VLLM_READY=0
    for i in $(seq 1 360); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log "ERROR: vllm server died. Last 40 lines of /workspace/vllm.log:"
            tail -40 /workspace/vllm.log
            exit 1
        fi
        # Test with an actual /v1/models request, not just /health
        if curl -s --max-time 3 "http://localhost:$VLLM_PORT/v1/models" 2>/dev/null | grep -q "$ORPHEUS_MODEL"; then
            log "vllm TTS server ready and serving model (took ${i}s)"
            VLLM_READY=1
            break
        fi
        if (( i % 15 == 0 )); then
            log "  Still waiting... (${i}s elapsed)"
            tail -3 /workspace/vllm.log 2>/dev/null || true
        fi
        sleep 1
    done

    if [[ "$VLLM_READY" == "0" ]]; then
        log "ERROR: vllm server did not become ready within 360s."
        tail -40 /workspace/vllm.log
        exit 1
    fi

    # Final verification: send a tiny test request to confirm the model actually works
    log "Verifying vllm can generate tokens..."
    TEST_RESULT=$(curl -s --max-time 30 "http://localhost:$VLLM_PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$ORPHEUS_MODEL\",\"prompt\":\"test\",\"max_tokens\":1}" 2>&1)
    if echo "$TEST_RESULT" | grep -q '"choices"'; then
        log "vllm verification passed — model is generating tokens."
    else
        log "WARNING: vllm test request did not return expected response:"
        echo "$TEST_RESULT" | head -5
        log "Continuing anyway — the model may need more warm-up time."
    fi
fi

# ── 8. GPU check ──
log "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

# ── 9. Start server ──
cd "$SCRIPT_DIR"
log "Starting Conversational AI Avatar Server..."
echo "═══════════════════════════════════════════════════════"
echo "  WebSocket on port 8765"
echo "  vllm TTS on port $VLLM_PORT"
echo "═══════════════════════════════════════════════════════"
python main.py
