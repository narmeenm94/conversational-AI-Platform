"""Orpheus TTS service for Pipecat — vllm streaming + SNAC decoding.

Architecture:
  1. Formats prompt with Orpheus special tokens
  2. Streams token text from a separate vllm server (OpenAI-compatible API)
  3. Accumulates text and re-tokenizes to extract audio code token IDs
  4. Collects audio code tokens into groups of 7
  5. Decodes each group with the SNAC model into PCM16 audio
  6. Yields audio frames to Pipecat in real-time

Based on the proven Orpheus_Distributed_FastAPI reference implementation:
  https://github.com/SebastianBodza/Orpheus_Distributed_FastAPI

Audio output: 24 kHz, 16-bit, mono PCM.
"""

import asyncio
import functools
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
AUDIO_SAMPLE_RATE = 24000
FADE_MS = 5


def _redistribute_codes(code_list: list[int], snac_model, device: torch.device):
    """Convert flat audio code list into 3-layer SNAC format and decode to audio."""
    if not code_list:
        return b""

    num_groups = len(code_list) // SNAC_TOKENS_PER_GROUP
    if num_groups == 0:
        return b""

    code_list = code_list[:num_groups * SNAC_TOKENS_PER_GROUP]

    layer_1, layer_2, layer_3 = [], [], []
    for i in range(num_groups):
        b = SNAC_TOKENS_PER_GROUP * i
        try:
            layer_1.append(code_list[b])
            layer_2.append(code_list[b + 1] - 4096)
            layer_3.append(code_list[b + 2] - 2 * 4096)
            layer_3.append(code_list[b + 3] - 3 * 4096)
            layer_2.append(code_list[b + 4] - 4 * 4096)
            layer_3.append(code_list[b + 5] - 5 * 4096)
            layer_3.append(code_list[b + 6] - 6 * 4096)
        except IndexError:
            logger.warning("IndexError during code redistribution at group %d", i)
            break

    if not layer_1:
        return b""

    codes = [
        torch.tensor(layer_1, device=device).unsqueeze(0),
        torch.tensor(layer_2, device=device).unsqueeze(0),
        torch.tensor(layer_3, device=device).unsqueeze(0),
    ]

    with torch.no_grad():
        audio_hat = snac_model.decode(codes)

    audio_tensor = audio_hat.squeeze().detach()

    fade_samples = int(AUDIO_SAMPLE_RATE * FADE_MS / 1000)
    fade_samples = (fade_samples // 2) * 2
    if fade_samples > 0 and audio_tensor.numel() >= 2 * fade_samples:
        if audio_tensor.ndim == 0:
            pass
        else:
            fade_in = torch.linspace(0.0, 1.0, fade_samples, device=audio_tensor.device)
            fade_out = torch.linspace(1.0, 0.0, fade_samples, device=audio_tensor.device)
            audio_tensor[:fade_samples] *= fade_in
            audio_tensor[-fade_samples:] *= fade_out

    audio_np = audio_tensor.cpu().to(torch.float32).numpy() * 32767
    pcm = np.clip(audio_np, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


class OrpheusTTSService(TTSService):
    """Streams audio from Orpheus TTS via vllm server + SNAC decoding."""

    def __init__(
        self,
        *,
        vllm_base_url: str = "http://localhost:8000/v1",
        model_name: str = "canopylabs/orpheus-3b-0.1-ft",
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
        await self._verify_vllm_connection()
        logger.info("Orpheus TTS ready (vllm + SNAC).")

    async def _verify_vllm_connection(self):
        """Verify the vllm server is reachable and serving the model."""
        for attempt in range(10):
            try:
                models = await self._client.models.list()
                model_ids = [m.id for m in models.data]
                if self._model_name in model_ids:
                    logger.info("vllm server verified: model '%s' is loaded.", self._model_name)
                    return
                logger.warning("vllm server is up but model '%s' not in %s", self._model_name, model_ids)
            except Exception as e:
                if attempt < 9:
                    logger.warning("vllm not ready (attempt %d/10): %s", attempt + 1, e)
                    await asyncio.sleep(5)
                else:
                    logger.error("vllm server not reachable after 10 attempts at %s", self._vllm_base_url)
                    raise RuntimeError(f"Cannot connect to vllm server at {self._vllm_base_url}: {e}")

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

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text back to token IDs."""
        return self._tokenizer.encode(text)

    def _decode_codes_to_audio(self, raw_codes: list[int]) -> bytes:
        """Subtract offset from raw token IDs and decode through SNAC."""
        codes = [t - CODE_TOKEN_OFFSET for t in raw_codes]
        return _redistribute_codes(codes, self._snac_model, self._snac_device)

    async def _create_stream_with_retry(self, prompt: str, max_retries: int = 3):
        """Create a vllm completions stream with retry logic for connection errors."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await self._client.completions.create(
                    model=self._model_name,
                    prompt=prompt,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    stream=True,
                    extra_body={
                        "repetition_penalty": self._repetition_penalty,
                    },
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "vllm connection attempt %d/%d failed: %s — retrying in %ds",
                        attempt + 1, max_retries, e, wait,
                    )
                    await asyncio.sleep(wait)
        raise last_error

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream audio frames from vllm token generation + SNAC decoding.

        Uses the same accumulate-and-retokenize approach as the reference
        Orpheus_Distributed_FastAPI implementation, which handles the case
        where vllm splits special token text across multiple stream chunks.
        """
        if not text or not text.strip():
            return

        logger.info("TTS [%s]: voice=%s text='%s'", context_id, self._voice, text[:80])
        yield TTSStartedFrame()

        loop = asyncio.get_event_loop()

        try:
            prompt = await loop.run_in_executor(
                None, functools.partial(self._format_prompt, text)
            )

            stream = await self._create_stream_with_retry(prompt)

            accumulated_text = ""
            processed_code_count = 0
            start_token_found = False
            start_idx = -1
            first_chunk_yielded = False

            async for chunk in stream:
                if not chunk.choices:
                    continue

                chunk_text = chunk.choices[0].text or ""
                if not chunk_text:
                    continue

                accumulated_text += chunk_text

                all_token_ids = await loop.run_in_executor(
                    None, self._tokenize, accumulated_text
                )

                if not start_token_found:
                    try:
                        start_idx = all_token_ids.index(CODE_START_TOKEN_ID)
                        start_token_found = True
                    except ValueError:
                        continue

                if not start_token_found:
                    continue

                potential_tokens = all_token_ids[start_idx + 1:]
                valid_raw_codes = [
                    t for t in potential_tokens
                    if t != CODE_END_TOKEN_ID and t >= CODE_TOKEN_OFFSET
                ]

                current_total = len(valid_raw_codes)

                if not first_chunk_yielded:
                    chunk_size = INITIAL_CHUNK_GROUPS * SNAC_TOKENS_PER_GROUP
                else:
                    chunk_size = STREAM_CHUNK_GROUPS * SNAC_TOKENS_PER_GROUP

                if current_total >= processed_code_count + chunk_size:
                    n_to_process = ((current_total - processed_code_count) // chunk_size) * chunk_size
                    end_idx = processed_code_count + n_to_process

                    if end_idx > processed_code_count:
                        codes_batch = valid_raw_codes[processed_code_count:end_idx]

                        pcm_bytes = await loop.run_in_executor(
                            None, self._decode_codes_to_audio, codes_batch
                        )

                        if pcm_bytes:
                            yield TTSAudioRawFrame(
                                audio=pcm_bytes,
                                sample_rate=24000,
                                num_channels=1,
                            )
                            first_chunk_yielded = True

                        processed_code_count = end_idx

            # Process remaining codes after stream ends
            if start_token_found:
                all_token_ids = await loop.run_in_executor(
                    None, self._tokenize, accumulated_text
                )
                potential_tokens = all_token_ids[start_idx + 1:]
                valid_raw_codes = [
                    t for t in potential_tokens
                    if t != CODE_END_TOKEN_ID and t >= CODE_TOKEN_OFFSET
                ]
                current_total = len(valid_raw_codes)

                if current_total > processed_code_count:
                    remaining = valid_raw_codes[processed_code_count:]
                    final_len = (len(remaining) // SNAC_TOKENS_PER_GROUP) * SNAC_TOKENS_PER_GROUP

                    if final_len > 0:
                        pcm_bytes = await loop.run_in_executor(
                            None, self._decode_codes_to_audio, remaining[:final_len]
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
