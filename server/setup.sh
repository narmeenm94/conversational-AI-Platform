#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Conversational AI Avatar Server — Setup (Linux/macOS)"
echo "============================================================"
echo

# ── Check Python ──
if ! command -v python3.10 &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3.10+ is not installed."
    echo "        Install with: sudo apt install python3.10 python3.10-venv"
    exit 1
fi
PYTHON=$(command -v python3.10 || command -v python3)
echo "[OK] Python found: $($PYTHON --version)"
echo

# ── Check NVIDIA GPU ──
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "[WARNING] nvidia-smi not found. CUDA GPU may not be available."
    echo "          Install NVIDIA drivers for GPU acceleration."
fi
echo

# ── Check Ollama ──
if ! command -v ollama &>/dev/null; then
    echo "[INFO] Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "[OK] Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
echo

# ── Create virtual environment ──
if [ ! -d ".venv" ]; then
    echo "[STEP 1/5] Creating Python virtual environment..."
    $PYTHON -m venv .venv
else
    echo "[STEP 1/5] Virtual environment already exists."
fi
echo

# ── Activate venv ──
echo "[STEP 2/5] Activating virtual environment..."
source .venv/bin/activate
echo

# ── Install PyTorch with CUDA ──
echo "[STEP 3/5] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo

# ── Install pip requirements ──
echo "[STEP 4/5] Installing Python dependencies..."
pip install -r requirements.txt
echo

# ── Install Orpheus TTS from GitHub ──
echo "[STEP 5/5] Installing Orpheus TTS..."
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi
pip install accelerate
echo

# ── Create .env if needed ──
if [ ! -f ".env" ]; then
    echo "[CONFIG] Creating .env from .env.example..."
    cp .env.example .env
    echo "         Edit .env to customize settings."
else
    echo "[CONFIG] .env already exists, skipping."
fi
echo

# ── Pull Ollama model ──
echo "[MODEL] Checking Ollama model..."
if ! ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
    echo "         Pulling llama3.1:8b (~4.7 GB download)..."
    ollama pull llama3.1:8b
else
    echo "         llama3.1:8b already available."
fi
echo

echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo
echo "  NOTE: The first time you run the server, these models will"
echo "  auto-download from HuggingFace (one-time, ~10 GB total):"
echo "    - Faster Whisper large-v3  (~3 GB)"
echo "    - Orpheus TTS medium-3b   (~6 GB)"
echo "    - BGE embedding model     (~1.3 GB)"
echo
echo "  To start the server, run:"
echo "    source .venv/bin/activate"
echo "    python main.py"
echo
echo "  The server will show 'Waiting for Unity client connection...'"
echo "  when it's ready."
echo "============================================================"
