# Voice Customization Guide

How to select, customize, and clone voices for the Orpheus TTS system.

## Built-in Voices

Orpheus TTS includes 8 built-in voices:

| Voice | Description |
|-------|-------------|
| `tara` | Female, warm and expressive (most realistic, recommended) |
| `leah` | Female, clear and professional |
| `jess` | Female, youthful |
| `mia` | Female, soft-spoken |
| `leo` | Male, confident |
| `dan` | Male, conversational |
| `zac` | Male, energetic |
| `zoe` | Female, friendly |

## Changing the Default Voice

In your `.env` file:

```bash
TTS_VOICE=tara
```

Change to any of the built-in voice names above. The server picks up the change on next restart.

## Voice Cloning

Orpheus supports zero-shot voice cloning using reference audio clips. This lets you create a custom voice without fine-tuning.

### Preparing Reference Clips

1. Record 3-5 audio clips of the target voice
2. Each clip should be 5-15 seconds long
3. Record in a quiet environment
4. Use clear, natural speech (not reading/monotone)
5. Include varied intonation and emotions
6. Save as WAV files (16kHz or higher, mono)

Place clips in:
```
server/voices/reference_clips/
├── clip_01.wav
├── clip_02.wav
├── clip_03.wav
├── clip_04.wav
└── clip_05.wav
```

### Using Cloned Voice

Voice cloning with Orpheus uses the pretrained model with reference audio. Modify the TTS service to pass reference clips:

```python
# In pipeline/tts_service.py, adjust the generate call:
stream = self._model.generate_speech_streaming(
    prompt=prompt,
    voice=self._voice,
    request_id=request_id,
    reference_audio=["voices/reference_clips/clip_01.wav"],
)
```

Consult the [Orpheus TTS documentation](https://github.com/canopyai/Orpheus-TTS) for the latest voice cloning API.

## Fine-tuning a Custom Voice

For higher quality custom voices, fine-tune the Orpheus model on your voice data.

### Data Requirements

| Quality Level | Samples Needed | Total Audio |
|--------------|----------------|-------------|
| Decent | ~50 clips | ~10 minutes |
| Good | ~150 clips | ~30 minutes |
| High quality | 300+ clips | ~60+ minutes |

### Preparing Training Data

1. Record the target voice reading varied content
2. Clean audio: remove background noise, normalize volume
3. Segment into 5-15 second clips
4. Create a transcript for each clip

### Training

Follow the fine-tuning instructions in the [Orpheus TTS repo](https://github.com/canopyai/Orpheus-TTS). The training code is provided and supports LoRA fine-tuning for efficient adaptation.

## Orpheus on Ollama

Orpheus can also run through Ollama for simpler deployment:

```bash
ollama pull legraphista/Orpheus
```

This is useful if you want a unified Ollama-based stack. Note that the Ollama version may have fewer configuration options than the native Python package.

## Emotion Tags Reference

These tags can appear in the LLM's text output and Orpheus renders them as natural vocal sounds:

| Tag | Sound | When to Use |
|-----|-------|-------------|
| `<laugh>` | Full laughter | Something genuinely funny |
| `<chuckle>` | Light laugh | Mild amusement |
| `<sigh>` | Audible sigh | Reflection, empathy, frustration |
| `<gasp>` | Sharp inhale | Surprise, shock, excitement |
| `<groan>` | "Ugh" sound | Displeasure, pain |
| `<yawn>` | Yawning | Tiredness (use sparingly) |
| `<cough>` | Throat clear | Very rare, natural filler |
| `<sniffle>` | Sniffle | Emotional moments (very rare) |

The LLM automatically places these tags in its responses based on conversation context. The system prompt (in `llm_service.py`) controls how and when tags are used.

## Alternative TTS Engines

If Orpheus doesn't meet your needs, consider these alternatives:

| Engine | Repo | Strengths |
|--------|------|-----------|
| Sesame CSM | [github.com/SesameAILabs/csm](https://github.com/SesameAILabs/csm) | High quality conversational speech |
| Chatterbox | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | Easy voice cloning |

To use an alternative, create a new TTS service class in `pipeline/` following the same `TTSService` interface pattern as `tts_service.py`.
