"""Pipecat adapter for an isolated streaming Moonshine recognizer."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import AsyncGenerator

import httpx
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601


logger = logging.getLogger(__name__)


class MoonshineSTTService(STTService):
    """Streams live PCM into Moonshine and finalizes immediately after VAD."""

    def __init__(
        self,
        *,
        base_url: str,
        worker_python: str,
        worker_script: str,
        language: str = "en",
        architecture: str = "tiny-streaming",
        cache_dir: str,
        autostart: bool = True,
        startup_timeout: float = 45.0,
        pre_roll_seconds: float = 0.45,
        **kwargs,
    ):
        super().__init__(sample_rate=16000, ttfs_p99_latency=0.35, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._worker_python = Path(worker_python).resolve()
        self._worker_script = Path(worker_script).resolve()
        self._language = language
        self._architecture = architecture
        self._cache_dir = Path(cache_dir).resolve()
        self._autostart = autostart
        self._startup_timeout = startup_timeout
        self._pre_roll_limit = max(3200, int(16000 * 2 * pre_roll_seconds))
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._turn_audio = bytearray()
        self._session_active = False
        self._stream_ok = True
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._ensure_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=15.0, write=10.0, pool=3.0),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        # Recognition happens continuously in process_audio_frame.
        if False:
            yield Frame()

    async def _healthy(self) -> bool:
        try:
            response = await self._client.get(self._base_url + "/health")
            payload = response.json()
            return response.is_success and payload.get("engine") == "moonshine"
        except (httpx.HTTPError, ValueError):
            return False

    def _spawn_worker(self) -> None:
        if not self._worker_python.is_file():
            raise RuntimeError(
                "Moonshine runtime missing. Create server/.venv-moonshine and "
                "install moonshine-voice."
            )
        runtime_dir = self._worker_script.parent / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (runtime_dir / "moonshine_worker.log").open(
            "a", encoding="utf-8", buffering=1
        )
        command = [
            str(self._worker_python),
            str(self._worker_script),
            "--host", "127.0.0.1",
            "--port", str(httpx.URL(self._base_url).port or 8771),
            "--language", self._language,
            "--architecture", self._architecture,
            "--cache-dir", str(self._cache_dir),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(self._worker_script.parent),
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        logger.info("Started isolated Moonshine worker (pid=%s).", self._process.pid)

    async def _ensure_worker(self) -> None:
        async with self._ensure_lock:
            if await self._healthy():
                return
            if not self._autostart:
                raise RuntimeError(f"Moonshine worker unavailable at {self._base_url}")
            if self._process is None or self._process.poll() is not None:
                self._spawn_worker()
            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        "Moonshine worker exited during startup. See "
                        "runtime/moonshine_worker.log."
                    )
                if await self._healthy():
                    return
                await asyncio.sleep(0.15)
            raise RuntimeError("Moonshine worker did not become ready in time.")

    async def warm_for_startup(self) -> None:
        await self._ensure_worker()
        logger.info(
            "Moonshine streaming STT ready (architecture=%s).", self._architecture
        )

    async def start(self, frame):
        await self._ensure_worker()
        await super().start(frame)

    def _remember_pre_roll(self, audio: bytes) -> None:
        self._pre_roll.append(audio)
        self._pre_roll_bytes += len(audio)
        while self._pre_roll and self._pre_roll_bytes > self._pre_roll_limit:
            self._pre_roll_bytes -= len(self._pre_roll.popleft())

    async def _post_audio(self, audio: bytes) -> None:
        if not audio:
            return
        response = await self._client.post(
            self._base_url + "/audio",
            content=audio,
            headers={"content-type": "application/octet-stream"},
        )
        response.raise_for_status()

    async def process_audio_frame(self, frame, direction):
        audio = bytes(frame.audio or b"")
        if not audio:
            return
        if self._session_active:
            self._turn_audio.extend(audio)
            if self._stream_ok:
                try:
                    await self._post_audio(audio)
                except Exception:
                    self._stream_ok = False
                    logger.exception(
                        "Moonshine streaming audio failed; the turn will be replayed."
                    )
        else:
            self._remember_pre_roll(audio)

    async def _handle_vad_user_started_speaking(self, frame):
        await super()._handle_vad_user_started_speaking(frame)
        await self._ensure_worker()
        response = await self._client.post(self._base_url + "/start")
        response.raise_for_status()
        pre_roll = b"".join(self._pre_roll)
        self._pre_roll.clear()
        self._pre_roll_bytes = 0
        self._turn_audio = bytearray(pre_roll)
        self._session_active = True
        self._stream_ok = True
        try:
            await self._post_audio(pre_roll)
        except Exception:
            self._stream_ok = False
            logger.exception("Could not send Moonshine speech pre-roll.")

    async def _finalize(self) -> dict:
        if not self._stream_ok:
            # Replay the buffered utterance if the streaming connection had a
            # transient failure, so a user turn is not silently lost.
            await self._ensure_worker()
            response = await self._client.post(self._base_url + "/start")
            response.raise_for_status()
            await self._post_audio(bytes(self._turn_audio))
        response = await self._client.post(self._base_url + "/stop")
        response.raise_for_status()
        return response.json()

    async def _handle_vad_user_stopped_speaking(self, frame):
        await super()._handle_vad_user_stopped_speaking(frame)
        if not self._session_active:
            return
        self._session_active = False
        started = time.perf_counter()
        try:
            result = await self._finalize()
            text = " ".join(str(result.get("text") or "").split())
            logger.info(
                "[STT] Moonshine finalized %r in %.3fs (worker %.3fs).",
                text or "<empty>",
                time.perf_counter() - started,
                float(result.get("finalize_seconds") or 0.0),
            )
            if text:
                transcript = TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                )
                transcript.finalized = True
                await self.push_frame(transcript)
        except Exception:
            logger.exception("Moonshine could not finalize the speech turn.")
        finally:
            self._turn_audio.clear()

    async def cleanup(self):
        await self._client.aclose()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._log_handle is not None:
            self._log_handle.close()
        await super().cleanup()
