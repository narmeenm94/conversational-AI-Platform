# Voice Customization Guide

The local expressive profile uses **Chatterbox**. English characters use the
350M-parameter Turbo model for the lowest latency, zero-shot voice cloning, and
native paralinguistic cues. Characters set to another supported language use
the multilingual model. Kokoro remains the ultra-fast neutral fallback, while
Orpheus is available either through a separate vLLM server or through the
fully local quantized `orpheus-cpp` streaming backend on Windows.

## Select the spoken language

Language belongs to the character profile, so different avatars can speak
different languages. Open `http://127.0.0.1:8766`, select a character, choose
**Spoken language**, save, and activate it.

Supported output languages are Arabic, Danish, German, Greek, English, Spanish,
Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch,
Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, and Chinese.

- English uses Chatterbox Turbo and supports native tags such as `[laugh]` and
  `[sigh]`.
- Other languages use Chatterbox Multilingual. Emotion intensity controls its
  acoustic exaggeration, and unsupported stage tags are removed before speech.
- The first switch to a model variant can take longer while it downloads and
  warms up. Later activations use the local cache.

The spoken output language and microphone recognition language are independent.
For best recognition, set `STT_LANGUAGE` in `server/.env` to the language users
will speak, or leave it empty for automatic detection.

## Select or upload a character voice

1. Start the server and open `http://127.0.0.1:8766`.
2. Select a character.
3. Choose a WAV from **Voice library**, or upload a new WAV.
4. Save and activate the character.

The editor stores voice files locally under `server/voices/`. A reference clip
must be longer than five seconds; a clean, mono 6-10 second recording with one
speaker is ideal. The clip controls voice identity and gender. It is prepared
and cached when the character becomes active.

Alex currently uses `voices/alex_female_reference.wav`, a warm female reference
created locally for the included female avatar. Replace it with a consented
human recording whenever you want a specific production voice.

## Emotion and performance

The character's **Default emotion intensity** controls how strongly the LLM is
directed to perform emotion and use cues. Chatterbox Turbo supports:

- `[laugh]`, `[chuckle]`
- `[sigh]`, `[gasp]`, `[groan]`
- `[cough]`, `[sniff]`, `[shush]`, `[clear throat]`

The response prompt uses cues sparingly. The speech sanitizer converts common
emoji and stage directions into supported cues, removes unsupported symbols,
expands abbreviations such as `e.g.`, and strips Markdown so those tokens are
never pronounced aloud.

The default **Pocket TTS** streaming profile prioritizes response latency and
does not accept native laugh, sigh, gasp, or groan tokens. When Pocket is
selected, the dialogue model is explicitly told not to emit performance markup,
and the final speech guard removes accidental parenthetical actions such as
`(laughs)` or `(sarcastic tone)` instead of reading them. Their recognized
emotion can still drive the avatar's facial-expression event. Select Chatterbox
Turbo or Orpheus for generated non-verbal vocal cues.

**Voice temperature** affects sampling variation. Start at `0.8`; reduce it for
more consistent delivery or increase it slightly for more variation. Turbo does
not expose the continuous acoustic exaggeration control of the larger
Chatterbox model, so the platform's intensity value acts through performance
direction and cue frequency while keeping the lower-latency Turbo backend. The
multilingual model uses the same intensity value as acoustic exaggeration.

## Fast fallback

Set `TTS_BACKEND=kokoro` and `TTS_VOICE=af_heart` in `server/.env` for the
smallest, fastest female voice. Kokoro is natural but does not render the
Chatterbox performance cues; the sanitizer removes them cleanly.

## Local Orpheus streaming option

Orpheus is an optional quality experiment when expressive human delivery and
native emotion cues matter more than the lowest latency. Install `orpheus-cpp`
plus its documented CUDA `llama-cpp-python` wheel, set
`TTS_BACKEND=orpheus-cpp`, and restart the server. Each character can select
Tara, Leah, Jess, Mia, Zoe, Leo, Dan, or Zac in the control interface.
`ORPHEUS_CPP_GPU_LAYERS=-1` offloads all possible layers and
`ORPHEUS_CPP_PREBUFFER_SECONDS=0.2` minimizes the startup buffer.

On the development RTX 4070 Laptop, the quantized 3B model produced first audio
in 0.73 seconds after warm-up with a 0.2-second setting. Its CPU path produced
no first audio after two minutes and is not viable for realtime use. The CUDA
package also needs an isolated Python environment because its NumPy dependency
conflicts with the pinned Chatterbox environment. For those reasons Pocket
remains the lowest-latency laptop default, Chatterbox is the integrated
expressive preset, and Orpheus remains a separately tested quality-focused
deployment profile.

## Commercial and consent notes

Chatterbox code and model are MIT licensed. Preserve its license notice. Voice
cloning still requires the speaker's permission even when the model license
allows commercial use.
