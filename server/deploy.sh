#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Conversational AI Avatar Server — Cloud GPU Deploy Script
# Run this on a RunPod pod (or any Linux GPU server) to set up
# and start the full pipeline in one shot.
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh            # first-time setup + start
#   ./deploy.sh --start    # skip installs, just start the server
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

START_ONLY=false
if [[ "${1:-}" == "--start" ]]; then
    START_ONLY=true
fi

log() { echo -e "\n\033[1;36m>>> $*\033[0m"; }

# ── Skip installs if --start ──
if [[ "$START_ONLY" == false ]]; then

    # ── System deps ──
    log "Installing system dependencies..."
    apt-get update -qq && apt-get install -y -qq curl git > /dev/null 2>&1 || true

    # ── Python deps ──
    log "Installing Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --quiet -r requirements.txt
    pip install --quiet git+https://github.com/Deathdadev/Orpheus-Speech-PyPi accelerate
    pip install --quiet python-dotenv sounddevice numpy

    # ── Ollama ──
    if ! command -v ollama &> /dev/null; then
        log "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        log "Ollama already installed."
    fi

    # Start Ollama daemon if not running
    if ! pgrep -x ollama > /dev/null; then
        log "Starting Ollama daemon..."
        ollama serve &
        sleep 5
    fi

    # Pull LLM model
    log "Pulling LLM model (llama3.1:8b)..."
    ollama pull llama3.1:8b

    # ── .env setup ──
    if [[ ! -f .env ]]; then
        log "Creating .env from .env.cloud template..."
        cp .env.cloud .env
        echo ""
        echo "╔═══════════════════════════════════════════════════════╗"
        echo "║  IMPORTANT: Edit .env and set your HF_TOKEN value!   ║"
        echo "║  nano .env  (or vi .env)                             ║"
        echo "╚═══════════════════════════════════════════════════════╝"
        echo ""
        read -p "Press ENTER after you've set HF_TOKEN in .env..."
    fi

    log "Setup complete!"
fi

# ── Ensure Ollama is running ──
if ! pgrep -x ollama > /dev/null; then
    log "Starting Ollama daemon..."
    ollama serve &
    sleep 3
fi

# ── Start the AI server ──
log "Starting Conversational AI Avatar Server..."
echo "═══════════════════════════════════════════════════════"
echo "  WebSocket will be available on port 8765"
echo "  Connect Unity client to: ws://<your-pod-ip>:8765"
echo "═══════════════════════════════════════════════════════"
python main.py
