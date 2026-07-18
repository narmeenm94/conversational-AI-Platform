# ──────────────────────────────────────────────────────────────
# Native Windows local runner (Chatterbox/Kokoro + Ollama + Faster Whisper).
# Cloud / pod path is unchanged: use server\start_cloud.sh on RunPod.
#
# First time:
#   cd server
#   powershell -ExecutionPolicy Bypass -File .\setup_local.ps1
# ──────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ServerDir

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.local.example") {
        Copy-Item ".env.local.example" ".env"
        Write-Host "Created .env from .env.local.example. Edit it if needed, then run this script again." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Missing .env. Copy .env.local.example to .env first." -ForegroundColor Red
    exit 1
}

$env:HF_HUB_ENABLE_HF_TRANSFER = if ($env:HF_HUB_ENABLE_HF_TRANSFER) { $env:HF_HUB_ENABLE_HF_TRANSFER } else { "0" }
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Force HF cache onto C:\ (NTFS) so model downloads don't hit symlink errors on D:.
# Cloud / pod deploy uses /workspace/huggingface and is untouched.
$LocalHfHome = Join-Path $env:USERPROFILE ".cache\huggingface"
if (-not (Test-Path $LocalHfHome)) { New-Item -Path $LocalHfHome -ItemType Directory -Force | Out-Null }
$env:HF_HOME = $LocalHfHome
$env:HUGGINGFACE_HUB_CACHE = (Join-Path $LocalHfHome "hub")
Write-Host "HF cache: $LocalHfHome" -ForegroundColor DarkGray

$venvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "No usable .venv found. Run: .\setup_local.ps1" -ForegroundColor Yellow
    exit 1
}
& $venvPython --version | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "The existing .venv points to a Python installation that no longer exists." -ForegroundColor Red
    Write-Host "Rename server\.venv, install Python 3.11, then run .\setup_local.ps1." -ForegroundColor Yellow
    exit 1
}

# Kokoro's misaki phonemizer needs spaCy en_core_web_sm. Install once so we don't
# hit the race where misaki installs it mid-run and the parent process can't import it.
$spacyCheck = & $venvPython -c "import importlib.util as u; print(bool(u.find_spec('en_core_web_sm')))"
if ($spacyCheck -ne "True") {
    Write-Host "Installing spaCy en_core_web_sm (one-time, ~12MB)..." -ForegroundColor DarkGray
    & $venvPython -m spacy download en_core_web_sm
}

# Friendly Ollama check (non-fatal — the user may run it themselves).
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null
    Write-Host "Ollama responding on http://localhost:11434" -ForegroundColor Green
} catch {
    Write-Host "Ollama is not responding on :11434. Open another terminal and run:  ollama serve" -ForegroundColor Yellow
}

Write-Host @"

Starting Conversational AI server (local profile from .env)
  WebSocket:  ws://0.0.0.0:8765
  Mic test:   $venvPython ..\tools\mic_test_client.py --url ws://127.0.0.1:8765

"@ -ForegroundColor Cyan

& $venvPython main.py
