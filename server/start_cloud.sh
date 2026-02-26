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
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
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
else
    log "Python venv found — skipping pip installs."
    source "$VENV_DIR/bin/activate"
fi

# ── 4. .env setup ──
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
    log "Creating .env from template..."
    cp "$SCRIPT_DIR/.env.cloud" "$SCRIPT_DIR/.env"
    sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" "$SCRIPT_DIR/.env"
    sed -i "s|^HF_HOME=.*|HF_HOME=$HF_HOME|" "$SCRIPT_DIR/.env"
fi

# ── 5. Start Ollama daemon ──
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

# ── 7. Start server ──
cd "$SCRIPT_DIR"
log "Starting Conversational AI Avatar Server..."
echo "═══════════════════════════════════════════════════════"
echo "  WebSocket on port 8765"
echo "═══════════════════════════════════════════════════════"
python main.py
