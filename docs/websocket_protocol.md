# Unity WebSocket Protocol v1

- Client -> server binary: signed little-endian PCM16, mono, 16 kHz.
- Server -> client binary: signed little-endian PCM16, mono, 24 kHz.
- Server -> client text: compact UTF-8 JSON with `"v":1`.

| Type | Extra fields | Meaning |
|---|---|---|
| `user_speech_started` | none | VAD confirmed speech; may interrupt the bot |
| `user_speech_stopped` | none | VAD confirmed the end of speech |
| `user_transcript` | `text` | final STT text |
| `assistant_speech_started` | none | a TTS fragment began |
| `assistant_speech_stopped` | none | a TTS fragment ended; audio may still be queued |
| `assistant_interrupted` | none | discard queued assistant audio immediately |
| `assistant_expression` | `emotion`, `intensity` | drive ARKit-style facial reaction |
| `assistant_animation` | `state`, `blend_seconds` | crossfade to `listening`, `thinking`, `remembering`, `searching`, `speaking`, `walking`, or `idle` |
| `assistant_spoken_text` | `text` | text for audio that actually entered the playback stream |
| `assistant_response_started` | none | guard the microphone for a new response |
| `assistant_response_finished` | none | the model response ended; queued speech may continue |

Audio is deliberately not base64 or JSON-wrapped. Control messages are never
placed in the audio queue.

Animation state names are configured per character in the control panel. Unity
checks that a state exists before crossfading and falls back to the existing
speaking/listening Animator booleans when it does not. Walking is exposed as a
gameplay call because locomotion authority remains with the Unity scene.

Pipecat may also emit RTVI JSON messages (`"label":"rtvi-ai"`). The Unity
client accepts both formats and maps `user-started-speaking`,
`user-stopped-speaking`, `user-transcription`, `bot-started-speaking`,
`bot-stopped-speaking`, and `bot-transcription` to the same avatar states.
