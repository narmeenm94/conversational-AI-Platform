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


class OrpheusTTSService(TTSService):
    """Streams audio from Orpheus TTS through the Pipecat pipeline.

    The LLM output (including emotion tags like <laugh>, <sigh>, <gasp>)
    goes directly to Orpheus, which renders them as natural vocal sounds.
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
        """Load OrpheusModel with explicit device placement.

        We bypass OrpheusModel.__init__ to work around two library bugs:
        1. The tokenizer uses the unmapped model name ("medium-3b")
        2. device_map="auto" causes meta-tensor errors on smaller GPUs
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
        """Convert text (with emotion tags) to streaming audio frames.

        Pipecat calls this method with each sentence/chunk of LLM output.
        We must yield TTSStartedFrame, then audio frames, then TTSStoppedFrame.
        """
        if not text or not text.strip():
            return

        logger.debug("TTS [%s]: voice=%s text=%s", context_id, self._voice, text[:80])

        yield TTSStartedFrame()

        try:
            prompt = f"{self._voice}: {text}"

            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                lambda: list(
                    self._model.generate_speech(prompt=prompt, voice=self._voice)
                ),
            )

            for audio_chunk in chunks:
                if audio_chunk and len(audio_chunk) > 0:
                    yield TTSAudioRawFrame(
                        audio=audio_chunk,
                        sample_rate=24000,
                        num_channels=1,
                    )

        except Exception as e:
            logger.error("TTS error: %s", e, exc_info=True)
            yield ErrorFrame(error=f"TTS generation failed: {e}")

        yield TTSStoppedFrame()
