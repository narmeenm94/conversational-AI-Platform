"""Orpheus TTS service for Pipecat — vllm streaming + SNAC decoding.

Architecture:
  1. Formats prompt with Orpheus special tokens
  2. Streams token IDs from a separate vllm server (OpenAI-compatible API)
  3. Collects audio code tokens into groups of 7
  4. Decodes each group with the SNAC model into PCM16 audio
  5. Yields audio frames to Pipecat in real-time

This generates audio faster than real-time on an A40, enabling smooth
streaming playback with no chopping.

Audio output: 24 kHz, 16-bit, mono PCM.
"""

import asyncio
import logging
from typing import AsyncGenerator

import numpy as np
import torch
from pipecat.frames.frames import (
    Frame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

logger = logging.getLogger(__name__)

CODE_TOKEN_OFFSET = 128266
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
SNAC_TOKENS_PER_GROUP = 7
INITIAL_CHUNK_GROUPS = 3
STREAM_CHUNK_GROUPS = 7
FADE_SAMPLES = 120  # 5ms at 24kHz


def _redistribute_codes(code_list: list[int], device: torch.device) -> list[torch.Tensor]:
    """Convert flat audio code list into 3-layer SNAC format."""
    num_groups = len(code_list) // SNAC_TOKENS_PER_GROUP
    code_list = code_list[:num_groups * SNAC_TOKENS_PER_GROUP]

    layer_1, layer_2, layer_3 = [], [], []
    for i in range(num_groups):
        b = SNAC_TOKENS_PER_GROUP * i
        layer_1.append(code_list[b])
        layer_2.append(code_list[b + 1] - 4096)
        layer_3.append(code_list[b + 2] - 2 * 4096)
        layer_3.append(code_list[b + 3] - 3 * 4096)
        layer_2.append(code_list[b + 4] - 4 * 4096)
        layer_3.append(code_list[b + 5] - 5 * 4096)
        layer_3.append(code_list[b + 6] - 6 * 4096)

    return [
        torch.tensor(layer_1, device=device).unsqueeze(0),
        torch.tensor(layer_2, device=device).unsqueeze(0),
        torch.tensor(layer_3, device=device).unsqueeze(0),
    ]


def _apply_fade(audio: np.ndarray, fade_samples: int = FADE_SAMPLES) -> np.ndarray:
    """Apply short fade-in/out to prevent pops between chunks."""
    if len(audio) < 2 * fade_samples:
        return audio
    audio = audio.astype(np.float32)
    audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
    audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return audio


class OrpheusTTSService(TTSService):
    """Streams audio from Orpheus TTS via vllm server + SNAC decoding."""

    def __init__(
        self,
        *,
        vllm_base_url: str = "http://localhost:8000/v1",
        model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod",
        voice: str = "tara",
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        max_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(sample_rate=24000, push_stop_frames=True, **kwargs)
        self._vllm_base_url = vllm_base_url
        self._model_name = model_name
        self._voice = voice
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._max_tokens = max_tokens
        self._snac_model = None
        self._tokenizer = None
        self._client = None
        self._snac_device = None

    async def start(self, frame):
        await super().start(frame)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models)
        logger.info("Orpheus TTS ready (vllm + SNAC).")

    def _load_models(self):
        from openai import AsyncOpenAI
        from snac import SNAC
        from transformers import AutoTokenizer

        logger.info("Loading SNAC decoder...")
        self._snac_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self._snac_model = self._snac_model.to(self._snac_device)
        if self._snac_device.type == "cuda":
            self._snac_model = self._snac_model.half()
        self._snac_model.eval()
        logger.info("SNAC decoder loaded on %s", self._snac_device)

        logger.info("Loading Orpheus tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        logger.info("Tokenizer loaded.")

        self._client = AsyncOpenAI(base_url=self._vllm_base_url, api_key="not-needed")

    def _format_prompt(self, text: str) -> str:
        """Format text into Orpheus prompt with special tokens."""
        full_text = f"{self._voice}: {text}"
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        input_ids = self._tokenizer(full_text, return_tensors="pt").input_ids
        modified_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        return self._tokenizer.decode(modified_ids[0], skip_special_tokens=False)

    def _decode_audio(self, codes: list[int]) -> bytes:
        """Decode a batch of audio codes into PCM16 bytes."""
        snac_codes = _redistribute_codes(codes, self._snac_device)
        with torch.no_grad():
            audio_hat = self._snac_model.decode(snac_codes)
        audio_np = audio_hat.squeeze().cpu().to(torch.float32).numpy()
        audio_np = _apply_fade(audio_np)
        pcm = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        return pcm.tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream audio frames from vllm token generation + SNAC decoding."""
        if not text or not text.strip():
            return

        logger.debug("TTS [%s]: voice=%s text=%s", context_id, self._voice, text[:80])
        yield TTSStartedFrame()

        try:
            prompt = self._format_prompt(text)

            stream = await self._client.completions.create(
                model=self._model_name,
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                repetition_penalty=self._repetition_penalty,
                stream=True,
            )

            audio_codes: list[int] = []
            code_started = False
            first_chunk = True
            chunk_threshold = INITIAL_CHUNK_GROUPS * SNAC_TOKENS_PER_GROUP

            async for chunk in stream:
                if not chunk.choices:
                    continue

                token_text = chunk.choices[0].text or ""
                if not token_text:
                    continue

                token_ids = self._tokenizer.encode(token_text, add_special_tokens=False)

                for token_id in token_ids:
                    if token_id == CODE_START_TOKEN_ID:
                        code_started = True
                        continue
                    if token_id == CODE_END_TOKEN_ID:
                        code_started = False
                        continue

                    if not code_started:
                        continue

                    code_value = token_id - CODE_TOKEN_OFFSET
                    if 0 <= code_value < 7 * 4096:
                        audio_codes.append(code_value)

                    if len(audio_codes) >= chunk_threshold:
                        n_groups = len(audio_codes) // SNAC_TOKENS_PER_GROUP
                        to_decode = audio_codes[:n_groups * SNAC_TOKENS_PER_GROUP]
                        audio_codes = audio_codes[n_groups * SNAC_TOKENS_PER_GROUP:]

                        pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                            None, self._decode_audio, to_decode
                        )

                        if pcm_bytes:
                            yield TTSAudioRawFrame(
                                audio=pcm_bytes,
                                sample_rate=24000,
                                num_channels=1,
                            )

                        if first_chunk:
                            first_chunk = False
                            chunk_threshold = STREAM_CHUNK_GROUPS * SNAC_TOKENS_PER_GROUP

            if len(audio_codes) >= SNAC_TOKENS_PER_GROUP:
                n_groups = len(audio_codes) // SNAC_TOKENS_PER_GROUP
                to_decode = audio_codes[:n_groups * SNAC_TOKENS_PER_GROUP]
                pcm_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, self._decode_audio, to_decode
                )
                if pcm_bytes:
                    yield TTSAudioRawFrame(
                        audio=pcm_bytes,
                        sample_rate=24000,
                        num_channels=1,
                    )

        except Exception as e:
            logger.error("TTS error: %s", e, exc_info=True)
            yield ErrorFrame(error=f"TTS generation failed: {e}")

        yield TTSStoppedFrame()
