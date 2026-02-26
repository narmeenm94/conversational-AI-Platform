# Conversational AI Avatar Platform

A fully self-hosted, subscription-free conversational AI platform that powers realistic 3D avatars on Meta Quest 3 (standalone VR) and Unity desktop. The user speaks naturally to an AI avatar that listens, thinks, and responds with a human-sounding emotional voice — all powered by open-source models running on your own PC.

## How It Works

```
Quest 3 (on Wi-Fi)  ←── WebSocket ──→  Your PC (GPU)

Captures mic                            Whisper (STT)
Renders avatar                          LLM + RAG (brain)
Plays audio                             Orpheus TTS (voice)
Lip sync                                Pipecat (pipeline)
```

The Quest 3 headset is a thin client — it only captures your voice, renders the avatar, and plays audio. All AI processing happens on your PC (or any server with a GPU).

## Hardware Requirements

### Minimum (12 GB VRAM)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4070 (12 GB VRAM) |
| CPU | 8-core |
| RAM | 32 GB |
| Storage | 50 GB SSD |
| OS | Windows 11 + WSL2, or Ubuntu 22.04+ |

### Recommended (24 GB VRAM)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| CPU | 12-core |
| RAM | 64 GB |
| Storage | 100 GB NVMe SSD |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- [Ollama](https://ollama.com/download) installed and running

### 2. Install Ollama and Pull a Model

```bash
# Windows: download from https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.1:8b
```

### 3. Set Up the Server

```bash
cd server
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install Orpheus TTS (from GitHub — not on PyPI)
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi
pip install accelerate

cp .env.example .env
# Edit .env as needed
```

### 4. Index Your Knowledge Base (Optional)

Place your training documents (`.txt`, `.md`, `.pdf`, `.docx`) in `server/knowledge/documents/`, then run:

```bash
python knowledge/ingest.py --docs-dir knowledge/documents/ --db-dir knowledge/db/
```

### 5. Start the Server

```bash
python main.py
# → WebSocket server listening on 0.0.0.0:8765
```

### 6. Desktop Testing (No Quest 3 Needed)

1. Open the Unity project in `unity-client/`
2. Open `DesktopTestScene`
3. Set ConversationManager → Server Address = `localhost`
4. Press Play and speak into your mic

### 7. Deploy to Quest 3

1. Find your PC's IP: `ipconfig` (Windows) or `ifconfig` (Linux/macOS)
2. Open firewall port: `netsh advfirewall firewall add rule name="AI" dir=in action=allow protocol=tcp localport=8765`
3. In Unity: Set ConversationManager → Server Address = your PC's IP
4. File → Build Settings → Android → Build and Run
5. Put on Quest 3 headset and start talking

## Project Structure

```
server/                     Python AI backend (runs on your PC)
  main.py                   Entry point — Pipecat pipeline + WebSocket
  config.py                 Configuration loader
  pipeline/                 AI service modules
    stt_service.py          Faster Whisper speech-to-text
    llm_service.py          Ollama LLM + system prompt + RAG
    tts_service.py          Orpheus TTS (custom Pipecat service)
    rag_service.py          ChromaDB retrieval
    emotion_processor.py    Optional sentiment analysis
  knowledge/                Knowledge base
    ingest.py               Document indexing CLI
    documents/              Your training docs go here
    db/                     ChromaDB storage (auto-created)
  voices/reference_clips/   Audio clips for voice cloning

unity-client/               Unity project (Quest 3 + Desktop)
  Assets/Scripts/           C# client scripts
  Assets/Plugins/Android/   Android manifest for Quest 3

tools/                      Utility scripts
  find_my_ip.py             Find your LAN IP for Quest 3
  test_connection.py        Test WebSocket from CLI
  benchmark_latency.py      Measure end-to-end latency

docs/                       Documentation
```

## Technology Stack

| Component | Technology | License |
|-----------|-----------|---------|
| STT | Faster Whisper | MIT |
| LLM | Ollama + Llama 3.1 | MIT |
| TTS | Orpheus TTS | Apache 2.0 |
| RAG | ChromaDB + Sentence Transformers | Apache 2.0 |
| Pipeline | Pipecat | BSD-2-Clause |
| VR Client | Unity + Meta XR SDK | Meta License |
| Lip Sync | uLipSync | MIT |
| WebSocket | NativeWebSocket | Apache 2.0 |

## Documentation

- [Quest 3 Setup Guide](docs/quest3_setup.md)
- [Knowledge Base Guide](docs/knowledge_base_guide.md)
- [Voice Customization](docs/voice_customization.md)
- [Cloud Deployment](docs/cloud_deployment.md)

## License

This project uses exclusively open-source components. See individual component licenses above.
