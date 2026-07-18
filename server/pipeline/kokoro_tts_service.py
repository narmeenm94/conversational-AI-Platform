"""Kokoro TTS service for Pipecat — fast, natural local TTS for Windows.

Used as the default local backend (8 GB GPU friendly, ~80 MB model, ~24 kHz).
For high-quality cloud / pod runs, the Orpheus + vLLM service stays the default.

Pipeline:
  text -> KPipeline (chunks by sentence) -> float32 audio @ 24 kHz
       -> int16 PCM -> TTSAudioRawFrame (streamed to client)

Notes:
  * KPipeline is synchronous and runs in a background executor so we never
    block the asyncio loop.
  * VRAM <2 GB on GPU; runs fine on CPU as a fallback.
"""

from __future__ import annotations

import asyncio
import logging
import queue as _stdq
from typing import AsyncGenerator

import numpy as np
from pipecat.frames.frames import (
    Frame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from pipeline.speech_text import expression_message, prepare_for_kokoro

logger = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24000
DEFAULT_LANG_CODE = "a"  # American English (Kokoro uses single-letter codes)
DEFAULT_VOICE = "af_heart"  # warm American female; swap via TTS_VOICE


class KokoroTTSService(TTSService):
    """Local, lightweight TTS using the Kokoro 82M model."""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        lang_code: str = DEFAULT_LANG_CODE,
        speed: float = 1.0,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(sample_rate=KOKORO_SAMPLE_RATE, push_stop_frames=True, **kwargs)
        self._voice = voice
        self._lang_code = lang_code
        self._speed = speed
        self._device = device
        self._pipeline = None
        self._loop_executor_warned = False

    async def start(self, frame):
        await super().start(frame)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_pipeline)
        logger.info(
            "Kokoro TTS ready (voice=%s, lang=%s, device=%s).",
            self._voice, self._lang_code, self._device or "auto",
        )

    def _load_pipeline(self):
        try:
            import torch
            from kokoro import KPipeline
        except ImportError as e:
            raise RuntimeError(
                "Kokoro TTS not installed. Add to your local venv:\n"
                "    pip install kokoro soundfile\n"
                f"(import error: {e})"
            ) from e

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading Kokoro KPipeline on %s ...", self._device)
        self._pipeline = KPipeline(lang_code=self._lang_code, device=self._device)

    @staticmethod
    def _audio_to_int16_pcm(audio) -> bytes:
        """Convert float audio (numpy or torch tensor) in [-1, 1] to int16 PCM bytes."""
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        np.clip(audio, -1.0, 1.0, out=audio)
        return (audio * 32767.0).astype(np.int16).tobytes()

    _SENTINEL = object()

    def _produce(self, text: str, out_queue: "_stdq.Queue") -> None:
        """Run KPipeline in a worker thread, push each segment's PCM bytes
        into ``out_queue`` as soon as it's ready, then push the sentinel.
        """
        try:
            for _graphemes, _phonemes, audio in self._pipeline(
                text, voice=self._voice, speed=self._speed
            ):
                if audio is None:
                    continue
                pcm = self._audio_to_int16_pcm(audio)
                if pcm:
                    out_queue.put(pcm)
        except Exception:
            logger.exception("Kokoro generation failed for text: %r", text[:80])
        finally:
            out_queue.put(self._SENTINEL)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        prepared = prepare_for_kokoro(text or "")
        if not prepared.speakable:
            return

        text = prepared.text

        logger.info("TTS [%s]: voice=%s text=%r", context_id, self._voice, text[:80])
        cue = expression_message(prepared)
        if cue:
            yield OutputTransportMessageUrgentFrame(message=cue)
        yield TTSStartedFrame()

        loop = asyncio.get_event_loop()
        pcm_queue: _stdq.Queue = _stdq.Queue()
        producer_fut = loop.run_in_executor(None, self._produce, text, pcm_queue)

        try:
            while True:
                pcm = await loop.run_in_executor(None, pcm_queue.get)
                if pcm is self._SENTINEL:
                    break
                yield TTSAudioRawFrame(
                    audio=pcm,
                    sample_rate=KOKORO_SAMPLE_RATE,
                    num_channels=1,
                )
        finally:
            try:
                await producer_fut
            except Exception:
                logger.exception("Kokoro producer task failed")
            yield TTSStoppedFrame()
