# Fresh PC Setup

This repository contains the complete Python server, character platform, Unity
6 project, desktop and Quest scenes, avatar assets, animation setup, embedded
Unity packages, character profile, voice reference, and source knowledge files.
The steps below recreate the tested local installation without changing its
runtime behavior.

## 1. Install prerequisites

- Git for Windows
- Python 3.11 x64 with **Add Python to PATH** enabled
- Ollama for Windows
- Unity Hub and Unity **6000.5.4f1**
- Unity Android Build Support, including its SDK/NDK tools and OpenJDK, for Quest builds
- An NVIDIA GPU is recommended, although the Pocket voice and Moonshine worker
  are designed to keep the low-latency path lightweight

## 2. Clone and prepare the server

```powershell
git clone --branch feature/vr-avatar-platform https://github.com/narmeenm94/conversational-AI-Platform.git
cd conversational-AI-Platform\server
Set-ExecutionPolicy -Scope Process Bypass
.\setup_local.ps1 -Tts pocket
```

The setup creates isolated Python environments for the main pipeline,
streaming speech recognition, and Pocket TTS. It also installs the pinned
Pocket worker and downloads the configured Ollama model. Model downloads can
take time on the first installation; later starts use the local caches.

Start the system:

```powershell
.\start_local.ps1
```

Keep that terminal open. The conversation WebSocket listens on port 8765 and
the character platform is available at <http://127.0.0.1:8766>.

## 3. Rebuild the included knowledge index

Generated Chroma database files are deliberately not stored in Git because
they are machine-generated and can be recreated. The HXRC source documents are
included under `server/knowledge/characters/alex/documents`.

1. Open the character platform at <http://127.0.0.1:8766>.
2. Select Narm.
3. Press **Reindex knowledge** once.

## 4. Open the included Unity project

1. In Unity Hub, add the repository's `unity-client` folder.
2. Open it with Unity **6000.5.4f1**.
3. Let Unity restore packages and finish importing.
4. For desktop testing, open `Assets/Scenes/ConversationalAvatarDemo.unity`,
   confirm the ConversationManager server address is `localhost`, and press Play.

Do not create a new Unity project or reinstall NativeWebSocket/uLipSync. Their
tested copies and all relevant project settings are included.

## 5. Configure a Quest build

The headset runs the Unity client while the AI models run on the PC.

1. Connect the PC and Quest 3 to the same low-latency Wi-Fi network.
2. Find the PC's current LAN IPv4 address with `ipconfig`.
3. In the Quest scene, set **ConversationSystem > ConversationManager > Server
   Address** to that LAN address. A cloned PC will usually have a different IP.
4. Allow inbound TCP ports 8765 and 8766 through the PC firewall.
5. Follow [Quest 3 Setup](quest3_setup.md) to verify OpenXR and build for Android.

## Files intentionally generated locally

The following are excluded from Git and are recreated by setup, Unity, or the
character editor:

- Python virtual environments (`server/.venv*`)
- downloaded Ollama, Hugging Face, STT, and TTS model caches
- `server/.env` (created from the safe example file)
- Chroma vector indexes under character `db` folders
- server logs and runtime worker output
- Unity `Library`, `Temp`, `Logs`, `obj`, and per-user settings

Excluding these files avoids committing machine-specific paths, caches,
credentials, and many gigabytes of reproducible output. No paid or hosted API
credentials are required for the default local profile.
