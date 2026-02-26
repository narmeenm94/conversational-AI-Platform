@echo off
echo ============================================================
echo   Conversational AI Avatar Server — Setup (Windows)
echo ============================================================
echo.

:: ── Check Python ──
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

:: ── Check NVIDIA GPU ──
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] nvidia-smi not found. You need an NVIDIA GPU with CUDA.
    echo           Download drivers from: https://www.nvidia.com/drivers
    echo.
)
echo [OK] NVIDIA GPU detected.
echo.

:: ── Check Ollama ──
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed.
    echo         Download from: https://ollama.com/download
    echo         After installing, run: ollama pull llama3.1:8b
    pause
    exit /b 1
)
echo [OK] Ollama found.
echo.

:: ── Create virtual environment ──
if not exist ".venv" (
    echo [STEP 1/5] Creating Python virtual environment...
    python -m venv .venv
    echo           Created .venv
) else (
    echo [STEP 1/5] Virtual environment already exists.
)
echo.

:: ── Activate venv ──
echo [STEP 2/5] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

:: ── Install PyTorch with CUDA ──
echo [STEP 3/5] Installing PyTorch with CUDA 12.1 support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

:: ── Install pip requirements ──
echo [STEP 4/5] Installing Python dependencies...
pip install -r requirements.txt
pip install chromadb sentence-transformers pypdf python-docx python-dotenv
echo.

:: ── Install Orpheus TTS from GitHub ──
echo [STEP 5/5] Installing Orpheus TTS...
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi
pip install accelerate
echo.

:: ── Create .env if it doesn't exist ──
if not exist ".env" (
    echo [CONFIG] Creating .env from .env.example...
    copy .env.example .env
    echo          Edit .env to customize settings.
) else (
    echo [CONFIG] .env already exists, skipping.
)
echo.

:: ── Pull Ollama model ──
echo [MODEL] Pulling llama3.1:8b into Ollama (skips if already present)...
ollama pull llama3.1:8b
echo.

echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   First run will auto-download ~10 GB of AI models.
echo.
echo   To start the server:
echo     .venv\Scripts\activate
echo     python main.py
echo ============================================================
pause
