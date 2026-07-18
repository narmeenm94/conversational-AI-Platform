# Conversational AI Avatar Platform

A fully self-hosted conversational AI platform for realistic Unity and Meta Quest avatars. It supports continuous microphone streaming, natural turn detection, interruption, character profiles, local knowledge retrieval, expressive local TTS, and realtime facial reactions without a paid inference service.

The production target is **Quest as a thin XR client plus a local PC inference host**. A Quest-only inference profile is experimental and intentionally lower quality. See [the 2026 architecture decision](docs/architecture_2026.md).

## How It Works

```
Quest 3 (on Wi-Fi)  ←── WebSocket ──→  Your PC (GPU)

Captures mic                            Whisper (STT)
Renders avatar                          LLM + RAG (brain)
Plays audio                             Chatterbox / Kokoro / Orpheus
Lip sync + expressions                  Pipecat (pipeline)
```

The Quest 3 headset is a thin client — it only captures your voice, renders the avatar, and plays audio. All AI processing happens on your PC (or any server with a GPU).

## Hardware Requirements

### Minimum local profile (8 GB VRAM)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4070 Laptop (8 GB VRAM) |
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

- Python 3.11
- NVIDIA GPU with CUDA support
- [Ollama](https://ollama.com/download) installed and running

### 2. Install Ollama and Pull a Model

```bash
# Windows: download from https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2:3b
```

### 3. Set Up the Server

Windows:

```powershell
cd server
.\setup_local.ps1
# Fast neutral fallback: .\setup_local.ps1 -Tts kokoro
.\start_local.ps1
```

Manual/Linux:

```bash
cd server
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# Choose a local TTS backend
pip install kokoro soundfile
# or: pip install chatterbox-tts
# Windows NVIDIA after Chatterbox: restore its pinned CUDA wheels
# pip install --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

cp .env.example .env
# Edit .env as needed
```

### 4. Open the Character Platform

Start the server and open [http://127.0.0.1:8766](http://127.0.0.1:8766). The Metropolia-branded editor controls each character's identity, backstory, traits, speaking style, spoken language, avatar asset, expressive voice, and isolated knowledge documents. Activating a character updates the live Unity client and resets conversation memory between identities.

Each character also stores its social relationship, conversational initiative,
canon-versus-improvisation level, **Talkativeness**, **Follow-up tendency**,
natural sentence range, semantic Animator state mappings, whether it speaks
first on connection, and a runtime preset. Highly talkative characters develop
an idea, contribute an opinion or brainstorm, and hand the topic back with a
specific question instead of behaving like support assistants. Brain and voice
engines can be selected independently. Changing engines takes effect on server
restart because only the chosen models are loaded into memory; voice identity,
personality, conversational style, and knowledge changes within the running
engine apply live.

Each character can select one of 23 local output languages. The recommended
English laptop profile uses Pocket TTS for genuine CPU audio streaming, with
Kokoro, Chatterbox, and Orpheus retained as selectable alternatives. Chatterbox
Multilingual provides the broadest language path and acoustic emotion control.
The first language-family switch warms the required model, then reuses its local
cache. See [Voice Customization](docs/voice_customization.md) for the complete
language list and microphone-language setting.

### 5. Index Your Knowledge Base (Optional)

The recommended path is to upload `.txt`, `.md`, `.pdf`, or `.docx` files in the character editor and press **Reindex knowledge**. Every character gets a separate Chroma collection.

The legacy command-line path is also available:

Place your training documents (`.txt`, `.md`, `.pdf`, `.docx`) in `server/knowledge/documents/`, then run:

```bash
python knowledge/ingest.py --docs-dir knowledge/documents/ --db-dir knowledge/db/
```

### 6. Start the Server

```bash
python main.py
# → WebSocket server listening on 0.0.0.0:8765
```

### 7. Desktop Testing (No Quest 3 Needed)

1. Open the Unity project in `unity-client/`
2. Open `Assets/Scenes/ConversationalAvatarDemo.unity`
3. Set ConversationManager → Server Address = `localhost`
4. Press Play and speak into your mic

### 8. Deploy to Quest 3

1. Find your PC's IP: `ipconfig` (Windows) or `ifconfig` (Linux/macOS)
2. Open LAN ports 8765 (conversation) and 8766 (character control, if used from the headset)
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
  control/                  Character registry, API, and branded editor
  characters/               Persistent multi-character profiles
  knowledge/                Knowledge base
    ingest.py               Document indexing CLI
    documents/              Your training docs go here
    db/                     ChromaDB storage (auto-created)
  voices/                   Per-character Chatterbox reference WAVs

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
| LLM | Ollama + Llama 3.2 3B; optional LFM2.5 / Qwen 3.5 / Granite | Llama Community / LFM Open / Apache 2.0 |
| TTS | Chatterbox Turbo + Multilingual / Kokoro / Orpheus CPP | MIT / Apache 2.0 / Apache 2.0 |
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
- [2026 Architecture and Model Decisions](docs/architecture_2026.md)
- [Unity WebSocket Protocol](docs/websocket_protocol.md)

## License

This project uses exclusively open-source components. See individual component licenses above.
