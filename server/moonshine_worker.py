"""Isolated streaming Moonshine STT worker."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from moonshine_voice import ModelArch, Transcriber, get_model_for_language


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("moonshine-worker")

ARCHITECTURES = {
    "tiny": ModelArch.TINY,
    "base": ModelArch.BASE,
    "tiny-streaming": ModelArch.TINY_STREAMING,
    "base-streaming": ModelArch.BASE_STREAMING,
    "small-streaming": ModelArch.SMALL_STREAMING,
    "medium-streaming": ModelArch.MEDIUM_STREAMING,
}


def create_app(*, language: str, architecture: str, cache_dir: Path) -> FastAPI:
    app = FastAPI(title="Moonshine streaming STT worker")
    lock = threading.RLock()
    active = False
    arch = ARCHITECTURES.get(architecture)
    if arch is None:
        raise ValueError(f"Unknown Moonshine architecture: {architecture}")

    logger.info("Preparing Moonshine %s model for %s", architecture, language)
    model_path, resolved_arch = get_model_for_language(
        language,
        arch,
        cache_root=cache_dir,
    )
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=resolved_arch,
        update_interval=0.2,
    )
    logger.info("Moonshine model ready: %s", model_path)

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "engine": "moonshine",
            "architecture": architecture,
            "language": language,
            "active": active,
        }

    @app.post("/start")
    def start_turn():
        nonlocal active
        with lock:
            if active:
                try:
                    transcriber.stop()
                except Exception:
                    logger.exception("Could not close the stale Moonshine turn")
            transcriber.start()
            active = True
        return {"ok": True}

    @app.post("/audio")
    async def add_audio(request: Request):
        payload = await request.body()
        if not payload:
            return {"ok": True, "samples": 0}
        if len(payload) % 2:
            payload = payload[:-1]
        samples = (
            np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
        )
        with lock:
            if not active:
                raise HTTPException(status_code=409, detail="No active speech turn")
            transcriber.add_audio(samples.tolist(), 16000)
        return {"ok": True, "samples": int(samples.size)}

    @app.post("/stop")
    def stop_turn():
        nonlocal active
        started = time.perf_counter()
        with lock:
            if not active:
                raise HTTPException(status_code=409, detail="No active speech turn")
            transcriber.stop()
            transcript = transcriber.update_transcription()
            active = False
        text = " ".join(
            line.text.strip() for line in transcript.lines if line.text.strip()
        ).strip()
        return {
            "text": text,
            "finalize_seconds": round(time.perf_counter() - started, 4),
            "lines": len(transcript.lines),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--language", default="en")
    parser.add_argument("--architecture", default="tiny-streaming")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runtime" / "moonshine_cache",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_app(
            language=args.language,
            architecture=args.architecture,
            cache_dir=args.cache_dir.resolve(),
        ),
        host=args.host,
        port=args.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
