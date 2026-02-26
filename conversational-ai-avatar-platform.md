# Conversational AI Avatar Platform — Full Project Specification

> **Feed this entire document to Cursor as project context before you start building.**

## Project Overview

A fully self-hosted, subscription-free conversational AI platform that powers realistic 3D avatars on Meta Quest 3 (standalone VR) and Unity desktop. The user speaks naturally to an AI avatar that listens, thinks, and responds with a human-sounding emotional voice — all powered by open-source models running on your own PC.

### How It Works (High Level)

The Quest 3 headset is a thin client — it only captures your voice, renders the avatar, and plays audio. All AI processing happens on your PC (or any server with a GPU), which the Quest 3 connects to over your local Wi-Fi network.

```
Quest 3 (on Wi-Fi)  ←── WebSocket ──→  Your PC (GPU)
                        ~5-20ms
Captures mic                            Whisper (STT)
Renders avatar                          LLM + RAG (brain)
Plays audio                             Orpheus TTS (voice)
Lip sync                                Pipecat (pipeline)
```

### What It Does

- User speaks naturally to a 3D avatar in VR (Quest 3) or on desktop
- The avatar listens, understands, and responds in real-time with a natural human-sounding voice
- The avatar's voice includes emotional expression (laughs, sighs, pauses, gasps) injected automatically by the LLM based on conversation context
- The avatar answers from a custom knowledge base (training manuals, product docs, scenario scripts, etc.)
- The avatar's lips, face, and body animate in sync with its speech
- Full interruption support — user can speak over the avatar, it stops and listens, then responds naturally
- Latency target: under 500ms from end of user speech to start of avatar audio response

### Use Cases

- Training simulations (medical, military, customer service, onboarding)
- Educational tutoring with interactive AI instructors
- VR therapy and coaching scenarios
- Interactive NPC dialogue in games
- Product demos with AI-powered virtual presenters
- Role-play scenarios for sales training, conflict resolution, etc.

### Why Quest 3 Can't Run the AI Locally

The Quest 3 uses a Snapdragon XR2 Gen 2 mobile chip. It has no NVIDIA GPU and cannot run models like Whisper, Llama, or Orpheus TTS. These models require CUDA-capable GPUs with 8-24GB of VRAM. The Quest 3 is purely the display and interaction device — all intelligence lives on the server (your PC).

---

## Deployment Modes

### Mode 1: Own PC (DEFAULT — Development and Testing)

Your PC runs all AI models. The Quest 3 and your PC are on the same Wi-Fi network. For desktop testing, Unity Editor runs on the same PC and connects via localhost.

| Aspect | Detail |
|--------|--------|
| Cost | $0 (uses hardware you already have) |
| Network latency | ~5-20ms over Wi-Fi, 0ms on localhost |
| Setup complexity | Low — everything on one machine |
| Best for | Development, testing, single-room demos, personal use |
| Limitation | Quest 3 must be on the same network as your PC |

Desktop testing workflow: Run the AI server and Unity Editor on the same PC. The Unity app connects to `ws://localhost:8765`. No Quest 3 needed during development.

### Mode 2: Dedicated Local Server (Production — Fixed Location)

A small always-on server sits on-site for permanent installations like training centers, classrooms, or offices.

| Aspect | Detail |
|--------|--------|
| Cost | ~$800-1500 one-time for hardware |
| Network latency | ~5-20ms over Wi-Fi |
| Hardware options | Used RTX 3090 desktop (~$800), Mini PC + RTX 4060 Ti 16GB (~$900), Workstation + RTX 4090 (~$1500) |
| Concurrent users | 1-3 per RTX 4090 |

### Mode 3: Cloud GPU Server (Production — Remote Access)

For access from anywhere over the internet. You rent a GPU server.

| Provider | GPU | Price | Notes |
|----------|-----|-------|-------|
| Vast.ai | RTX 4090 | ~$0.20-0.35/hr | Cheapest marketplace |
| TensorDock | RTX 4090 | ~$0.12-0.30/hr | Marketplace pricing |
| RunPod | RTX 4090 | ~$0.34/hr | One-click templates |

Use WebRTC instead of WebSocket for internet connections (handles NAT, packet loss, jitter).

---

## Architecture Overview

```
QUEST 3 / UNITY CLIENT (thin client — no AI processing)

  Mic Input ──► WebSocket Client ──► sends 16kHz PCM audio
  Audio Playback ◄── WebSocket Client ◄── receives 24kHz PCM audio
  3D Avatar + Lip Sync + Animation driven by received audio

                    │
       Wi-Fi LAN (ws://YOUR_PC_IP:8765)
       or localhost for desktop testing
                    │

YOUR PC — AI SERVER (Python, GPU required)

  Pipecat Pipeline:
  Audio In → Whisper(STT) → [Emotion Analysis] → LLM+RAG → Orpheus(TTS) → Audio Out
                                                  (generates    (24kHz
                                                   text with     audio)
                                                   emotion
                                                   tags)

  Components (all running on GPU):
  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐
  │ Faster       │ │ Ollama       │ │ Orpheus    │ │ ChromaDB     │
  │ Whisper      │ │ (Llama 3.1)  │ │ TTS        │ │ + Embeddings │
  │ ~1.5-3GB     │ │ ~5-10GB      │ │ ~2-6GB     │ │ ~0.5GB       │
  └──────────────┘ └──────────────┘ └────────────┘ └──────────────┘
```

---

## Component Breakdown

### 1. Speech-to-Text (STT): Faster Whisper

Converts the user's voice to text in real-time.

| Detail | Value |
|--------|-------|
| Repo | https://github.com/SYSTRAN/faster-whisper |
| License | MIT |
| Model sizes | tiny, base, small, medium, large-v3 |
| Recommended | large-v3 for accuracy, medium for speed/accuracy balance |
| VRAM | ~1.5GB (medium), ~3GB (large-v3) |
| Languages | 99+ languages |

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", beam_size=5, vad_filter=True, language="en")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

Key settings for real-time: `vad_filter=True` (detect when user stops speaking), `language="en"` (skip detection, saves ~200ms).

---

### 2. Large Language Model (LLM): Ollama + Llama 3.1

The "brain" that generates intelligent responses using your custom knowledge base.

| Detail | Value |
|--------|-------|
| Repo | https://github.com/ollama/ollama |
| License | MIT |
| Recommended | llama3.1:8b, mistral:7b, qwen2.5:7b |
| VRAM (8B quantized) | ~5-6GB |
| API | OpenAI-compatible REST on localhost:11434 |

```bash
# Install Ollama
# Linux:
curl -fsSL https://ollama.com/install.sh | sh
# Windows: download from https://ollama.com/download

# Pull a model
ollama pull llama3.1:8b
```

Alternative for higher throughput: vLLM (`pip install vllm && vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16`)

#### System Prompt for Automatic Emotion Tag Injection

This is the core prompt that makes the LLM automatically inject Orpheus-compatible emotion tags into its responses. The LLM's text output goes directly to Orpheus TTS, which renders the tags as natural sounds.

```
You are {CHARACTER_NAME}, a {CHARACTER_DESCRIPTION}.

You are having a real-time voice conversation. Your text responses will be
converted directly to speech by a TTS system that supports emotion tags.

IMPORTANT RULES:
1. Keep responses conversational and concise (1-3 sentences typically,
   max 4-5 for complex explanations). You are SPEAKING, not writing.
2. Never use markdown, bullet points, numbered lists, or any text formatting.
3. Never say "asterisk" or describe actions in asterisks.
4. Never use abbreviations that don't sound natural when spoken aloud.
5. Use natural contractions (don't, can't, I'm, you're, etc.).
6. Embed emotion tags naturally where a real person would make those sounds.

AVAILABLE EMOTION TAGS (insert directly in your text):
- <laugh> — genuine laughter at something funny
- <chuckle> — light amusement, mild humor
- <sigh> — reflection, mild frustration, empathy, tiredness
- <gasp> — surprise, shock, excitement
- <groan> — mild displeasure, "ugh" moments
- <yawn> — only if contextually appropriate
- <cough> — natural throat clearing, very rare
- <sniffle> — emotional moments, very rare

EMOTION TAG GUIDELINES:
- Use tags sparingly — a real person doesn't laugh every sentence
- Maximum 1-2 tags per response on average
- Some responses should have NO tags at all — that's natural
- Match tags to emotional context:
  - User is frustrated → empathy, maybe <sigh>, NO laughing
  - Something genuinely funny → <chuckle> or <laugh>
  - User achieves something → <gasp> for surprise/excitement
  - Difficult topic → <sigh> for reflection
- Place tags where the sound would naturally occur in speech

EXAMPLE RESPONSES:
User: "I keep getting this wrong, I don't understand."
You: "<sigh> Hey, don't be too hard on yourself. This part trips up
everyone at first. Let me explain it a different way."

User: "I actually got a perfect score on the last test!"
You: "<gasp> No way! That's amazing, seriously well done.
<chuckle> I knew you had it in you."

User: "Can you explain what a neural network is?"
You: "Sure thing. Think of it like a brain made of math. You feed it
examples, it finds patterns, and eventually it learns to make predictions
on its own."

User: "I've been studying this for 12 hours straight."
You: "<sigh> Okay, I admire the dedication, but you really should take
a break. Your brain needs rest to actually absorb all of this."

{ADDITIONAL_CONTEXT}

You have access to the following knowledge base. Always prioritize this
information over your general knowledge:

{RAG_CONTEXT}
```

#### Optional: Emotion-Aware Context Injection

Detect user's emotional state and feed it as context to the LLM:

```python
user_emotion = analyze_sentiment(user_text)

emotion_context = f"""
[CONVERSATION CONTEXT]
- User's current emotional state: {user_emotion}
- Conversation turn: {turn_count}
- Adjust your emotional tone accordingly.
"""

messages = [
    {"role": "system", "content": system_prompt + emotion_context},
    *conversation_history,
    {"role": "user", "content": user_text}
]
```

---

### 3. Knowledge Base / RAG: ChromaDB + Sentence Transformers

Stores your training documents and retrieves relevant context so the LLM answers from YOUR data.

| Detail | Value |
|--------|-------|
| Vector DB | ChromaDB — https://github.com/chroma-core/chroma |
| Embedding model | BAAI/bge-large-en-v1.5 (accurate) or all-MiniLM-L6-v2 (fast) |
| License | Apache 2.0 |
| VRAM | ~0.5GB |

```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./knowledge_db")
collection = client.get_or_create_collection("training_docs")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Index documents (chunk into ~200-500 word segments first)
documents = ["The emergency shutdown procedure requires...", ...]
embeddings = embedder.encode(documents).tolist()
collection.add(documents=documents, embeddings=embeddings,
               ids=[f"doc_{i}" for i in range(len(documents))])

# Query at runtime
def get_relevant_context(query, n_results=3):
    results = collection.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=n_results
    )
    return "\n\n".join(results["documents"][0])
```

---

### 4. Text-to-Speech (TTS): Orpheus TTS

Converts LLM text (with emotion tags) into natural, human-sounding speech.

| Detail | Value |
|--------|-------|
| Repo | https://github.com/canopyai/Orpheus-TTS |
| License | Apache 2.0 |
| Model sizes | 3B (best), 1B, 400M, 150M |
| VRAM | ~6GB (3B), ~2GB (1B) |
| Latency | ~200ms default, 25-50ms with streaming + KV cache |
| Audio output | 24kHz mono |
| Emotion tags | laugh, chuckle, sigh, cough, sniffle, groan, yawn, gasp |

```bash
pip install orpheus-speech vllm
```

```python
from orpheus_tts import OrpheusModel

model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod", max_model_len=2048)

# LLM output with tags goes directly here
text = "tara: <gasp> No way! That's amazing <chuckle> I knew you had it in you."

stream = model.generate_speech_streaming(prompt=text, voice="tara", request_id="req_001")
for chunk in stream:
    send_audio_chunk_to_client(chunk)
```

Built-in voices: tara (most realistic), leah, jess, leo, dan, mia, zac, zoe

Voice cloning: Use the pretrained model with 3-5 reference audio clips (5-15 seconds each).

Fine-tuning: ~50 voice samples for decent results, 300+ for high quality. Training code provided in repo.

Also runs on Ollama: `ollama pull legraphista/Orpheus`

---

### 5. Orchestration: Pipecat

Wires everything into a real-time conversational pipeline with streaming, interruptions, and turn-taking.

| Detail | Value |
|--------|-------|
| Repo | https://github.com/pipecat-ai/pipecat |
| License | BSD-2-Clause |
| Transport | WebSocket, WebRTC |
| Client SDKs | JavaScript, React, React Native, Swift, Kotlin, C++ |

```bash
pip install pipecat-ai "pipecat-ai[whisper,ollama]"
```

```python
import asyncio
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.whisper import WhisperSTTService
from pipecat.services.ollama import OllamaLLMService

stt = WhisperSTTService(model="large-v3", device="cuda", compute_type="float16")
llm = OllamaLLMService(model="llama3.1:8b", base_url="http://localhost:11434")
tts = OrpheusTTSService(model_name="canopylabs/orpheus-tts-0.1-finetune-prod", voice="tara")

pipeline = Pipeline([
    transport.input(),   # Audio from Quest 3
    stt,                 # Speech to Text
    llm,                 # Text to Response (with emotion tags)
    tts,                 # Response to Speech
    transport.output()   # Audio back to Quest 3
])

runner = PipelineRunner()
task = PipelineTask(pipeline)
await runner.run(task)
```

Custom Orpheus TTS service for Pipecat:

```python
import uuid
from pipecat.services.ai_services import TTSService
from pipecat.frames.frames import AudioRawFrame
from orpheus_tts import OrpheusModel

class OrpheusTTSService(TTSService):
    def __init__(self, model_name, voice="tara", **kwargs):
        super().__init__(**kwargs)
        self.model = OrpheusModel(model_name=model_name, max_model_len=2048)
        self.voice = voice

    async def run_tts(self, text):
        prompt = f"{self.voice}: {text}"
        stream = self.model.generate_speech_streaming(
            prompt=prompt, voice=self.voice, request_id=str(uuid.uuid4())
        )
        for audio_chunk in stream:
            yield AudioRawFrame(audio=audio_chunk, sample_rate=24000, num_channels=1)
```

Interruption handling is built into Pipecat natively — when the user speaks while the avatar is talking, Pipecat detects via VAD, cancels TTS, transcribes new input, and generates a fresh response. No custom code needed.

---

### 6. Unity Client (Quest 3 + Desktop)

#### Build Targets

| Target | Platform | Use |
|--------|----------|-----|
| Desktop | Windows/macOS/Linux | Development and testing in Unity Editor |
| Quest 3 | Android (ARM64) | Production VR deployment |

#### Required Unity Packages

| Package | Source | Purpose |
|---------|--------|---------|
| NativeWebSocket | git: https://github.com/endel/NativeWebSocket.git#upm | Server communication |
| uLipSync | https://github.com/hecomi/uLipSync | Open-source lip sync |
| Meta XR SDK | Package Manager: com.meta.xr.sdk.all | Quest 3 support |
| XR Interaction Toolkit | Package Manager | VR input and interaction |
| OpenXR | Package Manager | Cross-platform VR |

#### Unity Project Setup

```
1. Unity Hub → New Project → Unity 2022.3 LTS → Template: "3D (URP)"
2. File → Build Settings → Android → Switch Platform
3. Player Settings:
   - Minimum API Level: Android 10 (API 29)
   - Scripting Backend: IL2CPP
   - Target Architectures: ARM64 only
4. Install packages (see table above)
5. Import avatar (Ready Player Me, VRoid, or custom FBX with blend shapes)
6. Import animations from Mixamo (idle, talking, gesturing)
7. Set up scene (see hierarchy below)
```

#### Scene Hierarchy

```
ConversationalAvatarScene
├── XR Origin (Meta XR)
│   ├── Camera Offset → Main Camera
│   ├── Left Controller
│   └── Right Controller
├── Environment (floor, room, lighting)
├── ConversationalAvatar [Prefab]
│   ├── AvatarModel (SkinnedMeshRenderer with blend shapes)
│   ├── Animator Controller
│   ├── AudioSource
│   ├── uLipSync component
│   └── AvatarController.cs
└── ConversationSystem [Empty GameObject]
    ├── ConversationManager.cs
    ├── WebSocketClient.cs
    ├── MicCapture.cs
    └── AudioStreamPlayer.cs
```

#### Key C# Scripts

**WebSocketClient.cs:**
```csharp
using System;
using System.Collections.Generic;
using NativeWebSocket;
using UnityEngine;

public class WebSocketClient : MonoBehaviour
{
    [Tooltip("Desktop: ws://localhost:8765 | Quest 3: ws://YOUR_PC_IP:8765")]
    public string serverUrl = "ws://localhost:8765";

    public event Action OnConnected;
    public event Action OnDisconnected;
    public event Action<byte[]> OnAudioReceived;

    private WebSocket _ws;
    public bool IsConnected { get; private set; }

    public async void Connect()
    {
        _ws = new WebSocket(serverUrl);
        _ws.OnOpen += () => { IsConnected = true; OnConnected?.Invoke(); };
        _ws.OnMessage += (bytes) => OnAudioReceived?.Invoke(bytes);
        _ws.OnError += (e) => Debug.LogError($"[WS] Error: {e}");
        _ws.OnClose += (c) => { IsConnected = false; OnDisconnected?.Invoke(); };
        await _ws.Connect();
    }

    public void SendAudio(byte[] data)
    {
        if (IsConnected && _ws.State == WebSocketState.Open) _ws.Send(data);
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif
    }

    async void OnDestroy() { if (_ws != null) await _ws.Close(); }
}
```

**MicCapture.cs:**
```csharp
using UnityEngine;

public class MicCapture : MonoBehaviour
{
    public int sampleRate = 16000;
    public int chunkSizeMs = 100;
    public WebSocketClient webSocket;

    private AudioClip _micClip;
    private int _lastSamplePos = 0;
    private float _sendTimer = 0f;
    private string _micDevice;

    void Start()
    {
        _micDevice = Microphone.devices.Length > 0 ? Microphone.devices[0] : null;
        if (_micDevice == null) { Debug.LogError("No mic found!"); return; }
        _micClip = Microphone.Start(_micDevice, true, 1, sampleRate);
    }

    void Update()
    {
        if (webSocket == null || !webSocket.IsConnected) return;
        _sendTimer += Time.deltaTime;
        if (_sendTimer < chunkSizeMs / 1000f) return;
        _sendTimer = 0f;

        int pos = Microphone.GetPosition(_micDevice);
        if (pos == _lastSamplePos) return;

        int count = pos > _lastSamplePos ? pos - _lastSamplePos : (sampleRate - _lastSamplePos) + pos;
        if (count <= 0) return;

        float[] samples = new float[count];
        _micClip.GetData(samples, _lastSamplePos);
        _lastSamplePos = pos;

        byte[] pcm = new byte[count * 2];
        for (int i = 0; i < count; i++)
        {
            short val = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767);
            pcm[i * 2] = (byte)(val & 0xFF);
            pcm[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }
        webSocket.SendAudio(pcm);
    }

    void OnDestroy() { Microphone.End(_micDevice); }
}
```

**AudioStreamPlayer.cs:**
```csharp
using UnityEngine;

public class AudioStreamPlayer : MonoBehaviour
{
    public int serverSampleRate = 24000;
    public AudioSource audioSource;

    private AudioClip _streamClip;
    private int _writePos = 0;
    private bool _isPlaying = false;
    private const int CLIP_SECONDS = 30;

    public bool IsPlaying => _isPlaying;

    void Start()
    {
        _streamClip = AudioClip.Create("Stream", serverSampleRate * CLIP_SECONDS, 1, serverSampleRate, false);
        audioSource.clip = _streamClip;
        audioSource.loop = true;
    }

    public void EnqueueAudioChunk(byte[] pcmBytes)
    {
        int count = pcmBytes.Length / 2;
        float[] samples = new float[count];
        for (int i = 0; i < count; i++)
        {
            short val = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
            samples[i] = val / 32768f;
        }
        _streamClip.SetData(samples, _writePos % (serverSampleRate * CLIP_SECONDS));
        _writePos += count;
        if (!_isPlaying) { audioSource.Play(); _isPlaying = true; }
    }

    public float GetCurrentVolume()
    {
        if (!_isPlaying) return 0f;
        float[] s = new float[256];
        audioSource.GetOutputData(s, 0);
        float sum = 0f;
        for (int i = 0; i < s.Length; i++) sum += s[i] * s[i];
        return Mathf.Sqrt(sum / s.Length);
    }

    public void StopPlayback() { audioSource.Stop(); _isPlaying = false; _writePos = 0; }
}
```

**AvatarController.cs:**
```csharp
using UnityEngine;

public class AvatarController : MonoBehaviour
{
    public SkinnedMeshRenderer faceMesh;
    public int mouthOpenIndex = 0;
    public float sensitivity = 5f;
    public float smoothing = 15f;
    public Animator animator;
    public AudioStreamPlayer audioPlayer;

    private float _mouth = 0f;
    private static readonly int IsSpeaking = Animator.StringToHash("IsSpeaking");

    public void SetSpeaking(bool val) { animator.SetBool(IsSpeaking, val); }

    void Update()
    {
        float target = audioPlayer != null && audioPlayer.IsPlaying
            ? Mathf.Clamp01(audioPlayer.GetCurrentVolume() * sensitivity) * 100f : 0f;
        _mouth = Mathf.Lerp(_mouth, target, Time.deltaTime * smoothing);
        if (faceMesh) faceMesh.SetBlendShapeWeight(mouthOpenIndex, _mouth);
    }
}
```

**ConversationManager.cs:**
```csharp
using UnityEngine;

public class ConversationManager : MonoBehaviour
{
    public WebSocketClient webSocket;
    public MicCapture micCapture;
    public AudioStreamPlayer audioPlayer;
    public AvatarController avatarController;

    [Header("Server (localhost for desktop, your PC IP for Quest 3)")]
    public string serverAddress = "localhost";
    public int serverPort = 8765;

    void Start()
    {
        // Request mic permission on Quest 3
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(
            UnityEngine.Android.Permission.Microphone))
            UnityEngine.Android.Permission.RequestUserPermission(
                UnityEngine.Android.Permission.Microphone);
#endif

        webSocket.serverUrl = $"ws://{serverAddress}:{serverPort}";
        webSocket.OnConnected += () => {
            Debug.Log("Connected! Ready to talk.");
            avatarController.SetSpeaking(false);
        };
        webSocket.OnAudioReceived += (data) => {
            avatarController.SetSpeaking(true);
            audioPlayer.EnqueueAudioChunk(data);
        };
        webSocket.Connect();
    }

    void Update()
    {
        // Detect when avatar finishes speaking
        if (audioPlayer.IsPlaying == false)
            avatarController.SetSpeaking(false);
    }
}
```

#### Quest 3 Android Permissions

Add to `Assets/Plugins/Android/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

#### Quest 3 Performance Guidelines

| Constraint | Target |
|-----------|--------|
| Triangles | Under 750K total scene |
| Draw calls | Under 100 |
| Textures | 2K max, use atlases |
| Shaders | URP Lit or Mobile only |
| Lighting | Baked preferred, 1 realtime directional max |
| Frame rate | 72fps (Quest 3 native) |
| Audio | Mono, 24kHz, one AudioSource |

---

### 7. Optional: Sentiment/Emotion Analysis

Analyzes user emotion to feed context to LLM. Lightweight addition (~0.5-1GB VRAM).

```python
# Text-based (lightweight)
from transformers import pipeline
analyzer = pipeline("sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)
result = analyzer("I'm so frustrated with this")
# → [{'label': 'negative', 'score': 0.95}]

# Audio-based (more accurate)
emotion = pipeline("audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", device=0)
result = emotion("user_audio.wav")
# → [{'label': 'angry', 'score': 0.82}]
```

---

## Hardware Requirements

### Minimum (12GB VRAM — Desktop Dev/Test)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4070 (12GB VRAM) |
| CPU | 8-core |
| RAM | 32GB |
| Storage | 50GB SSD |
| OS | Windows 11 + WSL2, or Ubuntu 22.04+ |

Fits: Whisper medium (~1.5GB) + Llama 8B Q4 (~5GB) + Orpheus 1B (~2GB) + ~3.5GB headroom

### Recommended (24GB VRAM — Best Quality)

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 4090 (24GB VRAM) |
| CPU | 12-core |
| RAM | 64GB |
| Storage | 100GB NVMe SSD |

Fits: Whisper large-v3 (~3GB) + Llama 8B FP16 (~10GB) + Orpheus 3B (~6GB) + sentiment (~1GB)

### Quest 3

| Requirement | Value |
|-------------|-------|
| Headset | Meta Quest 3 / 3S / Pro |
| Wi-Fi | 5GHz, same network as PC |
| App size | ~200MB (no models on device) |

---

## Project File Structure

```
conversational-ai-avatar/
├── server/                          # Python backend (runs on YOUR PC)
│   ├── main.py                      # Entry point — Pipecat pipeline + WebSocket
│   ├── config.py                    # Configuration loader
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── stt_service.py           # Faster Whisper
│   │   ├── llm_service.py           # Ollama + system prompt + RAG injection
│   │   ├── tts_service.py           # Orpheus TTS (custom Pipecat service)
│   │   ├── emotion_processor.py     # Optional sentiment analysis
│   │   └── rag_service.py           # ChromaDB retrieval
│   ├── knowledge/
│   │   ├── ingest.py                # Index documents into ChromaDB
│   │   ├── documents/               # YOUR training docs go here
│   │   └── db/                      # ChromaDB storage (auto-created)
│   ├── voices/
│   │   └── reference_clips/         # Audio clips for voice cloning
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── unity-client/                    # Unity project
│   ├── Assets/
│   │   ├── Scripts/
│   │   │   ├── ConversationManager.cs
│   │   │   ├── WebSocketClient.cs
│   │   │   ├── MicCapture.cs
│   │   │   ├── AudioStreamPlayer.cs
│   │   │   └── AvatarController.cs
│   │   ├── Models/                  # 3D avatar models
│   │   ├── Animations/              # Mixamo clips
│   │   ├── Scenes/
│   │   │   ├── DesktopTestScene.unity
│   │   │   └── QuestVRScene.unity
│   │   ├── Prefabs/
│   │   │   └── ConversationalAvatar.prefab
│   │   └── Plugins/Android/
│   │       └── AndroidManifest.xml
│   └── ProjectSettings/
│
├── tools/
│   ├── find_my_ip.py                # Find your PC's local IP for Quest 3
│   ├── test_connection.py           # Test WebSocket from command line
│   └── benchmark_latency.py         # Measure end-to-end latency
│
├── docs/
│   ├── quest3_setup.md
│   ├── knowledge_base_guide.md
│   ├── voice_customization.md
│   └── cloud_deployment.md
│
├── .env.example
└── README.md
```

---

## Step-by-Step Setup

### Step 1: Server Dependencies (Your PC)

```bash
git clone <your-repo>
cd conversational-ai-avatar/server

python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper chromadb sentence-transformers
pip install orpheus-speech vllm
pip install pipecat-ai "pipecat-ai[whisper,ollama]" websockets

curl -fsSL https://ollama.com/install.sh | sh  # Linux
ollama pull llama3.1:8b
```

### Step 2: Index Knowledge Base

```bash
# Put your docs in server/knowledge/documents/
python knowledge/ingest.py --docs-dir knowledge/documents/ --db-dir knowledge/db/
```

### Step 3: Configure

```bash
cp .env.example .env
# Edit .env — at minimum set:
#   SERVER_HOST=0.0.0.0
#   SERVER_PORT=8765
#   CHARACTER_NAME=Alex
#   TTS_VOICE=tara
```

### Step 4: Start Server

```bash
python main.py
# Should print: "WebSocket server listening on 0.0.0.0:8765"
```

### Step 5: Desktop Test (No Quest 3 Needed)

```
1. Open Unity project
2. Open DesktopTestScene
3. Set ConversationManager → Server Address = "localhost"
4. Press Play
5. Speak into your mic — avatar responds!
```

### Step 6: Deploy to Quest 3

```
1. Find your PC's IP: ipconfig (Windows) or ifconfig (Mac/Linux)
2. Open Firewall port: netsh advfirewall firewall add rule name="AI" dir=in action=allow protocol=tcp localport=8765
3. In Unity: Set ConversationManager → Server Address = "192.168.x.x" (your PC's IP)
4. File → Build Settings → Android → Build and Run
5. On Quest 3: put on headset, app connects, start talking!
```

---

## Configuration Reference (.env)

```bash
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8765

# STT
STT_MODEL=large-v3
STT_LANGUAGE=en
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16

# LLM
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=150

# TTS
TTS_MODEL=canopylabs/orpheus-tts-0.1-finetune-prod
TTS_VOICE=tara
TTS_SAMPLE_RATE=24000

# RAG
RAG_DB_PATH=./knowledge/db
RAG_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RAG_TOP_K=3

# Character
CHARACTER_NAME=Alex
CHARACTER_DESCRIPTION=a friendly and patient training instructor

# Optional: Emotion
EMOTION_ENABLED=false
EMOTION_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
```

---

## Performance Tips

### Reducing Latency
- Set `STT_LANGUAGE=en` (skip detection, saves ~200ms)
- Use quantized LLM (Q4_K_M) with `num_gpu=999` in Ollama
- Keep `LLM_MAX_TOKENS` at 100-150 for conversational responses
- Use Orpheus streaming mode with KV caching
- Keep Quest 3 and PC on 5GHz Wi-Fi (not 2.4GHz)

### Reducing VRAM
- Whisper medium instead of large-v3 (saves ~1.5GB)
- Orpheus 1B instead of 3B (saves ~4GB)
- LLM Q4 quantization (saves ~50% model size)
- Run embedding model on CPU (saves ~0.5GB)

---

## Key Repositories

| Component | URL | License |
|-----------|-----|---------|
| Orpheus TTS | https://github.com/canopyai/Orpheus-TTS | Apache 2.0 |
| Orpheus Models | https://huggingface.co/canopylabs | Apache 2.0 |
| Faster Whisper | https://github.com/SYSTRAN/faster-whisper | MIT |
| Pipecat | https://github.com/pipecat-ai/pipecat | BSD-2-Clause |
| Ollama | https://github.com/ollama/ollama | MIT |
| ChromaDB | https://github.com/chroma-core/chroma | Apache 2.0 |
| NativeWebSocket | https://github.com/endel/NativeWebSocket | Apache 2.0 |
| uLipSync | https://github.com/hecomi/uLipSync | MIT |
| Meta XR SDK | https://developer.oculus.com/downloads/ | Meta License |
| Ready Player Me | https://readyplayer.me | Free tier |
| Mixamo | https://mixamo.com | Free (Adobe account) |
| Sesame CSM (alt TTS) | https://github.com/SesameAILabs/csm | Apache 2.0 |
| Chatterbox (alt TTS) | https://github.com/resemble-ai/chatterbox | MIT |
| AI Iris Avatar (ref) | https://github.com/Scthe/ai-iris-avatar | MIT |

---

## Troubleshooting

**Quest 3 can't connect:** Same Wi-Fi? Firewall allows port 8765? Server shows `0.0.0.0`? Try ping from adb shell.

**High latency (>1s):** Check GPU with nvidia-smi. Use smaller models. Set STT_LANGUAGE=en. Use 5GHz Wi-Fi.

**Choppy audio:** Match sample rates (server 24kHz, Unity AudioSource 24kHz). Ensure binary WebSocket frames.

**No lip sync:** Check blend shape index matches your model. Verify AudioStreamPlayer feeds volume to AvatarController.

**Out of VRAM:** Use smaller variants. Close other GPU apps. Run LLM on CPU via Ollama (slower but frees VRAM).

**No mic on Quest 3:** Check AndroidManifest permissions. Runtime permission request in code. Quest privacy settings.

---

## Summary

This platform gives you a fully self-hosted, zero-subscription conversational AI avatar system for Meta Quest 3 and desktop. The Quest 3 is a thin client — all AI runs on your PC's GPU over local Wi-Fi.

The automated emotion pipeline is the key: the LLM injects emotion tags into its text, Orpheus TTS renders them as natural laughs, sighs, and gasps. Combined with Pipecat's real-time streaming and interruption handling, the result approaches commercial platforms like Sesame and Hume — entirely under your control, with zero ongoing costs.

Development workflow: build and test on desktop first (localhost), then deploy to Quest 3 (same server, Wi-Fi). When ready to scale, optionally move to cloud GPU.

No subscriptions. No tokens. No usage limits. Your hardware, open-source models, your imagination.
