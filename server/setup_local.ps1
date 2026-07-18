param(
    [ValidateSet("pocket", "kokoro", "chatterbox")]
    [string]$Tts = "pocket"
)

$ErrorActionPreference = "Stop"
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ServerDir

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python 3.11 is required. Install it from python.org, enable Add to PATH, then rerun." -ForegroundColor Red
    exit 1
}
$version = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($version -ne "3.11") {
    Write-Host "Found Python $version, but this audio stack is tested on Python 3.11." -ForegroundColor Red
    exit 1
}

$venvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython --version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Existing .venv is broken. Rename it to .venv.old, then rerun this script." -ForegroundColor Red
        exit 1
    }
} else {
    & $python.Source -m venv .venv
}

& $venvPython -m pip install --upgrade pip wheel setuptools
& $venvPython -m pip install -r requirements.txt
if ($Tts -eq "kokoro") {
    & $venvPython -m pip install kokoro soundfile
    & $venvPython -m spacy download en_core_web_sm
} elseif ($Tts -eq "chatterbox") {
    & $venvPython -m pip install chatterbox-tts
    # PyPI's default torch wheel is CPU-only on Windows. Restore the matching
    # CUDA build after Chatterbox's pinned dependencies so Turbo uses NVIDIA.
    & $venvPython -m pip install --upgrade --force-reinstall `
        torch==2.6.0 torchaudio==2.6.0 `
        --index-url https://download.pytorch.org/whl/cu124
}

# Pipecat's sentence aggregator uses NLTK to release complete sentences to TTS.
# Without these tables, transcripts work but speech is delayed until a much
# larger text block is available.
& $venvPython -m nltk.downloader -d ".venv\nltk_data" punkt punkt_tab

# Keep streaming Moonshine in a small isolated environment so its ONNX/numpy
# dependencies cannot disturb the main Pipecat, TTS, or CUDA environment.
$moonshinePython = Join-Path $ServerDir ".venv-moonshine\Scripts\python.exe"
if (-not (Test-Path $moonshinePython)) {
    & $venvPython -m venv ".venv-moonshine"
}
& $moonshinePython -m pip install --upgrade pip
& $moonshinePython -m pip install moonshine-voice==0.0.69 fastapi uvicorn

# Pocket TTS is the current low-latency production voice. Keep it isolated so
# its dependencies cannot disturb the main pipeline environment.
$pocketPython = Join-Path $ServerDir ".venv-pocket\Scripts\python.exe"
if (-not (Test-Path $pocketPython)) {
    & $venvPython -m venv ".venv-pocket"
}
& $pocketPython -m pip install --upgrade pip
& $pocketPython -m pip install pocket-tts==2.1.0 fastapi uvicorn

if (-not (Test-Path ".env")) {
    Copy-Item ".env.local.example" ".env"
    $envContents = Get-Content ".env"
    $envContents -replace '^TTS_BACKEND=.*$', "TTS_BACKEND=$Tts" |
        Set-Content ".env" -Encoding utf8
}

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama pull llama3.2:3b
} else {
    Write-Host "Ollama is not installed. Install it, then run: ollama pull llama3.2:3b" -ForegroundColor Yellow
}

Write-Host "Local environment ready. Start with .\start_local.ps1" -ForegroundColor Green
