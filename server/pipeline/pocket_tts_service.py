"""Pipecat adapter for the isolated, genuinely streaming Pocket TTS worker."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable

import httpx
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType

from pipeline.speech_text import expression_message, prepare_for_pocket


logger = logging.getLogger(__name__)
POCKET_SAMPLE_RATE = 24000


class CompleteSentenceAggregator(SimpleTextAggregator):
    """Never ask the voice to speak a token-limit fragment."""

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Release terminally punctuated frames without redundant look-ahead.

        VoiceResponseLimiter already emits only complete sentences.  The stock
        Pipecat aggregator waits for the first non-space character of the next
        sentence, which unnecessarily delayed first audio until most or all of
        the LLM response had finished.
        """
        self._text += text
        pending = self._text.strip()
        if pending and pending[-1:] in ".!?":
            await self.reset()
            yield Aggregation(text=pending, type=AggregationType.SENTENCE)

    async def flush(self):
        pending = self._text.strip()
        if pending and pending[-1:] not in ".!?":
            logger.info("Dropping incomplete Pocket tail: %r", pending)
            await self.reset()
            return None
        return await super().flush()


class PocketTTSService(TTSService):
    """Streams 24 kHz PCM from Pocket without loading it into the main venv."""

    def __init__(
        self,
        *,
        base_url: str,
        worker_python: str,
        worker_script: str,
        default_voice: str = "azelma",
        language: str = "english",
        autostart: bool = True,
        startup_timeout: float = 45.0,
        profile_provider=None,
        audit_callback: Callable[[dict[str, Any]], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=POCKET_SAMPLE_RATE,
            push_stop_frames=True,
            text_aggregator=CompleteSentenceAggregator(),
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._worker_python = Path(worker_python).resolve()
        self._worker_script = Path(worker_script).resolve()
        self._default_voice = default_voice
        self._language = language
        self._autostart = autostart
        self._startup_timeout = startup_timeout
        self._profile_provider = profile_provider
        self._audit_callback = audit_callback
        self._active_voice = ""
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=None, write=10.0, pool=3.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )
        self._ensure_lock = asyncio.Lock()
        self._worker_ready = False
        self._active_request_ids: set[str] = set()
        # Cancellation must never wait for the streaming request's connection.
        self._cancel_client = httpx.AsyncClient(
            timeout=httpx.Timeout(1.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    def _profile_settings(self) -> dict[str, Any]:
        return dict(self._profile_provider() if self._profile_provider else {})

    def _requested_voice(self) -> str:
        settings = self._profile_settings()
        reference = str(settings.get("reference_audio") or "").strip()
        if reference and Path(reference).is_file():
            return str(Path(reference).resolve())
        return str(settings.get("pocket_voice") or self._default_voice).strip()

    def _voice(self) -> str:
        return self._active_voice or self._requested_voice()

    async def _healthy(self) -> bool:
        try:
            response = await self._client.get(self._base_url + "/health")
            data = response.json()
            return (
                response.is_success
                and data.get("status") == "healthy"
                and data.get("engine") == "pocket-tts"
            )
        except (httpx.HTTPError, ValueError):
            return False

    def _spawn_worker(self) -> None:
        if not self._worker_python.is_file():
            raise RuntimeError(
                "Pocket TTS isolated runtime is missing. Run setup_local.ps1 "
                "or create server/.venv-pocket first."
            )
        if not self._worker_script.is_file():
            raise RuntimeError(f"Pocket TTS worker not found: {self._worker_script}")
        runtime_dir = self._worker_script.parent / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (runtime_dir / "pocket_worker.log").open(
            "a", encoding="utf-8", buffering=1
        )
        environment = os.environ.copy()
        environment.setdefault("HF_HOME", str(runtime_dir / "pocket_hf_cache"))
        command = [
            str(self._worker_python),
            str(self._worker_script),
            "--host", "127.0.0.1",
            "--port", str(httpx.URL(self._base_url).port or 8770),
            "--language", self._language,
            "--voice", self._default_voice,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            command,
            cwd=str(self._worker_script.parent),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        logger.info("Started isolated Pocket TTS worker (pid=%s).", self._process.pid)

    async def _ensure_worker(self) -> None:
        if self._worker_ready:
            return
        async with self._ensure_lock:
            if self._worker_ready:
                return
            if await self._healthy():
                self._worker_ready = True
                return
            if not self._autostart:
                raise RuntimeError(f"Pocket TTS worker is unavailable at {self._base_url}")
            if self._process is None or self._process.poll() is not None:
                self._spawn_worker()
            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        "Pocket TTS worker exited during startup. See runtime/pocket_worker.log."
                    )
                if await self._healthy():
                    self._worker_ready = True
                    return
                await asyncio.sleep(0.2)
            raise RuntimeError("Pocket TTS worker did not become ready in time.")

    async def _prepare_voice(self) -> None:
        await self._ensure_worker()
        requested = self._requested_voice()
        response = await self._client.post(
            self._base_url + "/prepare", json={"voice": requested}
        )
        if response.is_success:
            self._active_voice = requested
            return
        # Pocket's cloning checkpoint is gated separately from its built-in
        # voices. A missing Hugging Face acceptance must not take the entire
        # conversation server down; stay usable with the selected female voice.
        if Path(requested).is_file() and response.status_code == 400:
            logger.warning(
                "Pocket voice cloning is unavailable; falling back to built-in %s. "
                "Accept the Pocket model terms and configure HF_TOKEN to clone WAV voices.",
                self._default_voice,
            )
            fallback = await self._client.post(
                self._base_url + "/prepare", json={"voice": self._default_voice}
            )
            fallback.raise_for_status()
            self._active_voice = self._default_voice
            return
        response.raise_for_status()

    async def warm_for_startup(self) -> None:
        await self._prepare_voice()
        logger.info("Pocket TTS worker and active voice are warm.")

    async def prepare_active_profile(self) -> None:
        try:
            self._active_voice = ""
            await self._prepare_voice()
        except Exception:
            logger.exception("Could not prepare the active Pocket TTS voice.")

    async def start(self, frame):
        await super().start(frame)
        await self._ensure_worker()

    async def _cancel_requests(self, request_ids: list[str]) -> None:
        if not request_ids:
            return
        try:
            await self._cancel_client.post(
                self._base_url + "/cancel",
                json={"request_ids": request_ids},
            )
        except httpx.HTTPError:
            # The local stream cancellation still propagates; this endpoint is
            # the extra guard that frees Pocket's model lock promptly.
            logger.warning("Could not notify Pocket worker of interruption.")

    async def _cancel_active_requests(self) -> None:
        await self._cancel_requests(list(self._active_request_ids))

    async def interrupt_active_speech(self) -> None:
        """Called at VAD-start, outside TTS's possibly busy frame queue."""
        await self._cancel_active_requests()

    async def _handle_interruption(
        self, frame, direction: FrameDirection
    ) -> None:
        await self._cancel_active_requests()
        await super()._handle_interruption(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        prepared = prepare_for_pocket(text or "")
        if not prepared.speakable:
            return
        cue = expression_message(prepared)
        if cue:
            yield OutputTransportMessageUrgentFrame(message=cue)
        yield TTSStartedFrame()

        started = time.perf_counter()
        first_audio_at: float | None = None
        audio_bytes = 0
        spoken_event_sent = False
        odd_byte = b""
        self._active_request_ids.add(context_id)
        try:
            await self._ensure_worker()
            async with self._client.stream(
                "POST",
                self._base_url + "/tts",
                json={
                    "text": prepared.text,
                    "voice": self._voice(),
                    "request_id": context_id,
                },
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    chunk = odd_byte + chunk
                    if len(chunk) % 2:
                        odd_byte, chunk = chunk[-1:], chunk[:-1]
                    else:
                        odd_byte = b""
                    if not chunk:
                        continue
                    if first_audio_at is None:
                        first_audio_at = time.perf_counter()
                    if not spoken_event_sent:
                        spoken_event_sent = True
                        yield OutputTransportMessageUrgentFrame(message={
                            "v": 1,
                            "type": "assistant_spoken_text",
                            "text": prepared.text,
                        })
                    audio_bytes += len(chunk)
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=POCKET_SAMPLE_RATE,
                        num_channels=1,
                    )
        except asyncio.CancelledError:
            logger.info("Cancelled interrupted Pocket speech: %r", prepared.text[:80])
            # Pipecat can cancel run_tts before the InterruptionFrame itself is
            # processed by this service. Notify the isolated worker from the
            # cancellation path as well, otherwise its model lock keeps running
            # speech nobody will ever hear.
            await asyncio.shield(self._cancel_requests([context_id]))
            raise
        except Exception as exc:
            self._worker_ready = False
            logger.error("Pocket TTS failed: %s", exc, exc_info=True)
            yield ErrorFrame(error=f"Pocket TTS failed: {exc}")
        finally:
            self._active_request_ids.discard(context_id)
            elapsed = time.perf_counter() - started
            first_seconds = (
                round(first_audio_at - started, 4) if first_audio_at is not None else None
            )
            audio_seconds = audio_bytes / (POCKET_SAMPLE_RATE * 2)
            logger.info(
                "Pocket streamed %.2fs audio; first chunk=%s total=%.2fs.",
                audio_seconds,
                f"{first_seconds:.3f}s" if first_seconds is not None else "none",
                elapsed,
            )
            if self._audit_callback:
                self._audit_callback({
                    "text": prepared.text,
                    "variant": "pocket",
                    "first_audio_seconds": first_seconds,
                    "generation_seconds": round(elapsed, 4),
                    "audio_seconds": round(audio_seconds, 4),
                })
            yield TTSStoppedFrame()

    async def cleanup(self):
        await self._cancel_active_requests()
        await self._cancel_client.aclose()
        await self._client.aclose()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._log_handle is not None:
            self._log_handle.close()
        await super().cleanup()
