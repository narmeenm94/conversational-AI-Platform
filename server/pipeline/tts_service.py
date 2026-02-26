"""Orpheus TTS service for Pipecat — custom TTSService subclass.

Uses the orpheus-speech community package (OrpheusModel) which wraps
the Canopy Labs Orpheus model with vllm or HuggingFace Transformers
backend.

Audio output: 24 kHz, 16-bit, mono PCM.
"""

import asyncio
import logging
import platform
from typing import AsyncGenerator

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

_SENTINEL = object()


class OrpheusTTSService(TTSService):
    """Streams audio from Orpheus TTS through the Pipecat pipeline.

    Audio chunks are yielded as soon as they're generated (not batched),
    so the user hears audio while the rest is still being synthesized.
    """

    _MODEL_NAME_MAP = {
        "medium-3b": "canopylabs/orpheus-tts-0.1-finetune-prod",
    }

    def __init__(
        self,
        *,
        model_name: str = "medium-3b",
        voice: str = "tara",
        **kwargs,
    ):
        super().__init__(sample_rate=24000, push_stop_frames=True, **kwargs)
        self._voice = voice
        self._model_name = model_name
        self._model = None

    async def start(self, frame):
        await super().start(frame)
        logger.info("Loading Orpheus TTS model: %s (voice=%s)", self._model_name, self._voice)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("Orpheus TTS model loaded.")

    def _load_model(self):
        """Load OrpheusModel with transformers backend.

        We intentionally skip vllm: it pre-allocates all remaining GPU memory,
        starving Ollama (LLM) and causing EngineDeadError on generation.
        Transformers + streaming gives good perceived latency without the
        memory conflict.
        """
        from orpheus_tts import OrpheusModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = self._MODEL_NAME_MAP.get(self._model_name, self._model_name)

        orpheus = OrpheusModel.__new__(OrpheusModel)
        orpheus.model_name = resolved
        orpheus.dtype = torch.bfloat16
        orpheus.platform = platform.system()
        orpheus.available_voices = ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]

        if torch.cuda.is_available():
            logger.info("Loading TTS model on CUDA (device_map={'': 0})...")
            orpheus.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
            )
        else:
            logger.info("Loading TTS model on CPU...")
            orpheus.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                torch_dtype=torch.float32,
            )

        orpheus.tokeniser = AutoTokenizer.from_pretrained(resolved)
        orpheus.engine = None

        return orpheus

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream audio chunks to the pipeline as soon as each is generated."""
        if not text or not text.strip():
            return

        logger.debug("TTS [%s]: voice=%s text=%s", context_id, self._voice, text[:80])

        yield TTSStartedFrame()

        try:
            prompt = f"{self._voice}: {text}"
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _generate():
                try:
                    for chunk in self._model.generate_speech(prompt=prompt, voice=self._voice):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

            gen_future = loop.run_in_executor(None, _generate)

            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                if item is not None and len(item) > 0:
                    yield TTSAudioRawFrame(
                        audio=item,
                        sample_rate=24000,
                        num_channels=1,
                    )

            await gen_future

        except Exception as e:
            logger.error("TTS error: %s", e, exc_info=True)
            yield ErrorFrame(error=f"TTS generation failed: {e}")

        yield TTSStoppedFrame()
