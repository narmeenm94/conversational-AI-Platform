# Real-Time Avatar Architecture (2026)

## Decision

Use the Quest as the XR renderer, microphone, speaker, lip-sync, and reaction
client. Run the high-quality AI pipeline on the local PC over Wi-Fi. Keep a
second, explicitly lower-quality standalone Quest profile for deployments where
no PC is allowed.

The development machine has an RTX 4070 Laptop GPU with 8 GB VRAM. Orpheus 3B
and an 8B LLM should not share that GPU. The practical local profile is:

| Stage | Default | Quality alternative | Why |
|---|---|---|---|
| VAD | Silero, 200 ms start / 280 ms stop | tune per microphone and room | fast turn completion with reliable barge-in |
| STT | faster-whisper `base.en`, CPU int8 | `small` for multilingual input | 0.42 s measured for a 2.5 s English utterance without competing for TTS GPU time |
| LLM | Qwen 3.5 4B through Ollama's native stream with thinking disabled | LFM2.5 1.2B fastest profile | Qwen best preserved natural character behavior; the native endpoint avoids hidden-reasoning latency |
| RAG | Chroma + MiniLM on CPU, lazy-loaded | domain-specific embeddings | avoids consuming conversation GPU memory |
| Streaming TTS | Pocket TTS 100M in an isolated CPU worker | Kokoro 82M | first PCM measured at 0.39-0.52 s over local HTTP while leaving the GPU to the brain |
| Expressive TTS | Chatterbox Turbo 350M | Orpheus CPP streaming; Qwen3-TTS 0.6B experiment | voice cloning and non-verbal/emotion control |
| Lip sync | uLipSync in Unity | server phoneme timings later | MIT, realtime, Burst/Jobs, no native service |

Orpheus can use a separate vLLM server or the local quantized `orpheus-cpp`
backend. The actual Windows CUDA benchmark on this laptop delivered first audio
in 0.73 seconds after warm-up with a 0.2-second prebuffer. The CPU path failed
to produce first audio after two minutes. Orpheus therefore remains an opt-in
quality profile rather than the default low-latency backend.

## Deployment profiles

The character editor now stores a runtime selection alongside identity. A
performance preset fills in a tested brain/voice pair, while the brain and
voice dropdowns can still be overridden independently. The active character's
selection is resolved at server startup; switching to a character that requests
different engines displays a restart-required warning rather than silently
using the wrong model.

The recommended laptop preset is Qwen 3.5 4B plus Pocket TTS. Kokoro remains the
smallest fallback, Chatterbox remains the expressive non-verbal option, and
Orpheus vLLM remains the high-end workstation profile. Latency values in the
editor are planning estimates; benchmark the target machine and microphone.

### Local PC + Quest (production target)

```text
Quest microphone -> PCM16 WebSocket -> VAD -> STT -> character/RAG -> LLM
Quest avatar <- expressions + semantic animation states + PCM16 <- streaming TTS
```

This gives the best balance of realism, privacy, and latency. The control
channel is versioned JSON while audio stays unwrapped binary PCM. A user speech
start event immediately interrupts generation and flushes queued Unity audio.

### Quest-only (experimental edge profile)

Use sherpa-onnx or whisper.cpp for ARM64 STT, a 0.8-2B 4-bit LLM through a native
runtime, and MOSS-TTS-Nano ONNX for speech. MOSS-TTS-Nano is 100M parameters,
CPU-streaming, voice-cloning capable, and includes an Android ONNX example.

Do not promise the same quality as the PC profile. Quest 3 shares memory and
thermal headroom between XR rendering and inference; a realistic avatar at
72/90 FPS must remain the priority. Validate one stage at a time on-device.

## Latency acceptance criteria

Measure from the last user speech sample, not from when a test client finishes
sending a synthetic tone.

| Metric | Target | Maximum acceptable P95 |
|---|---:|---:|
| End-of-speech -> final transcript | 150 ms | 300 ms |
| Final transcript -> first LLM text | 200 ms | 400 ms |
| First speakable text -> first audio | 300 ms | 600 ms |
| End-of-speech -> audible response | 850 ms | 1,300 ms |
| Confirmed interruption -> silence | 150 ms | 300 ms |
| Unity playback underruns | 0 | 0 |
| Quest frame rate | 72 FPS | no sustained drops |

The VAD silence window is part of the perceived delay. A 500 ms stop threshold
makes a true 500 ms end-to-end response impossible, so optimize for a natural
sub-second handoff first and add semantic end-of-turn prediction only after the
baseline is stable.

The local profile now uses a 280 ms VAD stop window and an 80 ms Unity audio
prebuffer. Generated speech is DC-corrected and faded at utterance boundaries,
and Unity applies an additional 8 ms fade-in to prevent a non-zero first sample
from producing an audible click.

## Natural speech target

Sesame CSM is a useful architectural reference because it conditions speech on
prior text and audio segments, preserving more conversational continuity than
isolated sentence TTS. The public CSM-1B release is a base generation model,
not the fine-tuned model used in Sesame's interactive demo, and its official
example generates complete audio rather than exposing a production streaming
contract. Keep it as a research profile until a local benchmark proves both
voice quality and first-audio latency.

## Character and knowledge contract

- Put stable identity, backstory, traits, voice style, relationships, and hard
  boundaries in a character JSON file.
- Put factual deployment knowledge in the indexed knowledge base.
- Keep only the latest turn pairs in the live LLM context; long-term memory
  should be summarized or retrieved, not appended forever.
- Retrieval runs on CPU and is lazy: an empty knowledge base loads no embedding
  model.
- Only the exact supported emotion tokens may be emitted. Unknown stage
  directions are removed before TTS so the avatar never says “pause” or
  “emotional” aloud.

## Avatar contract

The included `Avatar.glb` is ready for facial animation: 54 skin joints, 72 head
morph targets, ARKit-style expressions, and the full Oculus viseme set. Use the
Unity menu **Conversational AI > Configure Selected Avaturn Avatar**, then add
uLipSync to the avatar's AudioSource.

Recommended uLipSync mapping:

| uLipSync | Avaturn blend shape |
|---|---|
| A | `viseme_aa` |
| I | `viseme_I` |
| U | `viseme_U` |
| E | `viseme_E` |
| O | `viseme_O` |
| N | `viseme_nn` |
| silence/noise | `viseme_sil` |

After mapping, disable `AvatarController.driveMouthFromVolume`; the volume jaw
driver is only a dependency-free fallback.

## Licensing checkpoint

Code/model licenses must be checked separately before a commercial release.
Current preferred components use permissive licenses: Chatterbox and uLipSync
are MIT; Qwen3/Qwen3-TTS, MOSS-TTS-Nano, CosyVoice, and sherpa-onnx are Apache
2.0; faster-whisper and whisper.cpp are MIT. Preserve notices and re-check the
exact downloaded model card at release time.

The balanced profile uses the Meta Llama 3.2 Community License and must retain
its required notices. The fastest LFM2.5 profile uses the LFM Open License:
commercial use is free below the license's $10M annual-revenue threshold, after
which a commercial license is required. Qwen 3.5 and Granite remain Apache-2.0
deployment profiles when those terms do not fit a deployment.

Voice cloning also requires consent from the speaker even when the software
license permits commercial use.
