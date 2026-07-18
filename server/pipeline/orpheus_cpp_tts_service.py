"""Native local streaming adapter for the quantized Orpheus speech model."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from pipeline.speech_text import expression_message, prepare_for_orpheus

logger = logging.getLogger(__name__)


class OrpheusCppTTSService(TTSService):
    """Stream Orpheus GGUF audio locally through ``orpheus-cpp``."""

    def __init__(
        self,
        *,
        voice: str = "tara",
        language: str = "en",
        prebuffer_seconds: float = 0.5,
        gpu_layers: int = -1,
        verbose: bool = False,
        profile_provider: Callable[[], dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(sample_rate=24000, push_stop_frames=True, **kwargs)
        self._default_voice = voice
        self._language = language
        self._prebuffer_seconds = max(0.2, float(prebuffer_seconds))
        self._gpu_layers = int(gpu_layers)
        self._verbose = verbose
        self._profile_provider = profile_provider
        self._model = None
        self._epoch = 0

    def _settings(self) -> dict[str, Any]:
        return dict(self._profile_provider() if self._profile_provider else {})

    def _active_language(self) -> str:
        return str(self._settings().get("language") or self._language or "en").lower()

    def _active_voice(self) -> str:
        return str(
            self._settings().get("orpheus_voice") or self._default_voice or "tara"
        ).lower()

    def _load_model(self) -> None:
        try:
            from orpheus_cpp import OrpheusCpp
        except ImportError as exc:
            raise RuntimeError(
                "Orpheus CPP is not installed. Install `orpheus-cpp` and the "
                "Windows llama-cpp-python wheel, then restart the server."
            ) from exc
        language = self._active_language()
        logger.info("Loading local Orpheus CPP model for language=%s ...", language)
        self._model = OrpheusCpp(
            n_gpu_layers=self._gpu_layers,
            verbose=self._verbose,
            lang=language,
        )
        self._language = language

    async def start(self, frame):
        await super().start(frame)
        if self._model is None:
            await asyncio.to_thread(self._load_model)
        logger.info(
            "Orpheus CPP ready (voice=%s, prebuffer=%.2fs).",
            self._active_voice(),
            self._prebuffer_seconds,
        )

    async def warm_for_startup(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self._load_model)

    async def prepare_active_profile(self) -> None:
        language = self._active_language()
        if self._model is None or language != self._language:
            await asyncio.to_thread(self._load_model)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            self._epoch += 1
        await super().process_frame(frame, direction)

    @staticmethod
    def _pcm16(samples: Any) -> bytes:
        audio = np.asarray(samples).reshape(-1)
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        else:
            audio = np.clip(audio, -32768, 32767).astype(np.int16, copy=False)
        return audio.tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        prepared = prepare_for_orpheus(text or "")
        if not prepared.speakable:
            return

        cue = expression_message(prepared)
        if cue:
            yield OutputTransportMessageUrgentFrame(message=cue)
        yield TTSStartedFrame()

        request_epoch = self._epoch
        started = time.perf_counter()
        first_audio = None
        options = {
            "voice_id": self._active_voice(),
            "pre_buffer_size": self._prebuffer_seconds,
        }
        logger.info(
            "TTS [%s]: orpheus-cpp voice=%s text=%r",
            context_id,
            options["voice_id"],
            prepared.text[:80],
        )

        async for sample_rate, samples in self._model.stream_tts(
            prepared.text, options=options
        ):
            if request_epoch != self._epoch:
                break
            pcm = self._pcm16(samples)
            if not pcm:
                continue
            if first_audio is None:
                first_audio = time.perf_counter()
                logger.info(
                    "Orpheus CPP first audio in %.3fs.", first_audio - started
                )
            yield TTSAudioRawFrame(
                audio=pcm,
                sample_rate=int(sample_rate),
                num_channels=1,
            )
        yield TTSStoppedFrame()
