"""Isolated Kyutai Pocket TTS worker.

The main conversational server intentionally does not import Pocket or its
PyTorch stack.  This process owns the 100M CPU model and exposes raw PCM so the
first generated chunk can go straight to Unity without a WAV header or the
reference server's trailing silence.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pocket_tts.models.tts_model import TTSModel


logger = logging.getLogger("pocket-worker")
app = FastAPI(title="Conversational AI Pocket TTS worker")
model: TTSModel | None = None
default_voice = "azelma"
voice_states: dict[str, dict] = {}
generation_lock = threading.Lock()
cancelled_requests: set[str] = set()
cancel_lock = threading.Lock()
request_cancel_events: dict[str, threading.Event] = {}
active_generation_cancel_event: threading.Event | None = None


class PocketGenerationCancelled(RuntimeError):
    """Raised inside Pocket's latent loop so abandoned work really stops."""


def _install_generation_cancellation() -> None:
    """Make Pocket's background latent thread cooperatively cancellable.

    Closing ``generate_audio_stream`` alone does not stop Pocket's internal
    autoregressive thread. It then continues to use the model after the HTTP
    client has gone away, which blocks or slows every subsequent utterance.
    This lightweight hook is checked once per latent step (roughly 20-50 ms).
    """
    assert model is not None
    def cancellable_autoregressive_generation(
        model_state, max_gen_len, frames_after_eos, latents_queue
    ):
        backbone_input = torch.full(
            (1, 1, model.flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(model.flow_lm.parameters())).device,
            dtype=model.flow_lm.dtype,
        )
        eos_step = None
        generated_steps = 0
        for generation_step in range(max_gen_len):
            event = active_generation_cancel_event
            if event is not None and event.is_set():
                logger.info("Pocket latent generation stopped after %d steps", generated_steps)
                break
            next_latent, is_eos = model._run_flow_lm_and_increment_step(
                model_state=model_state,
                backbone_input_latents=backbone_input,
            )
            if is_eos.item() and eos_step is None:
                eos_step = generation_step
            if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                break
            latents_queue.put(next_latent)
            backbone_input = next_latent
            generated_steps += 1
        latents_queue.put(None)

    model._autoregressive_generation = cancellable_autoregressive_generation


def _voice_key(value: str | None) -> str:
    return str(value or default_voice).strip() or default_voice


def _state_for_voice(value: str | None) -> dict:
    if model is None:
        raise RuntimeError("Pocket TTS model is not loaded")
    key = _voice_key(value)
    if key in voice_states:
        return voice_states[key]
    path = Path(key)
    if path.is_file():
        state = model.get_state_for_audio_prompt(path, truncate=True)
    else:
        state = model._cached_get_state_for_audio_prompt(key)
    voice_states[key] = state
    logger.info("Prepared Pocket voice: %s", key)
    return state


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "engine": "pocket-tts",
        "sample_rate": 24000,
    }


@app.post("/prepare")
def prepare(payload: dict = Body(default_factory=dict)):
    try:
        _state_for_voice(payload.get("voice"))
    except Exception as exc:
        logger.exception("Could not prepare Pocket voice")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ready", "voice": _voice_key(payload.get("voice"))}


@app.post("/tts")
def synthesize(payload: dict = Body(...)):
    text = " ".join(str(payload.get("text") or "").split())
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    request_id = str(payload.get("request_id") or "").strip()
    try:
        state = _state_for_voice(payload.get("voice"))
    except Exception as exc:
        logger.exception("Could not load Pocket voice")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def raw_pcm_stream():
        # A synchronous StreamingResponse iterator can be abandoned without
        # being closed when its HTTP client disappears. Keep Pocket and its
        # stateful lock in an independent producer that always runs a finally
        # block, then bridge audio into an async response queue.
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue = asyncio.Queue()
        end_of_stream = object()
        cancel_event = threading.Event()

        def produce_audio():
            global active_generation_cancel_event
            chunks = None
            try:
                with generation_lock:
                    if request_id:
                        with cancel_lock:
                            request_cancel_events[request_id] = cancel_event
                            if request_id in cancelled_requests:
                                cancelled_requests.discard(request_id)
                                cancel_event.set()
                    active_generation_cancel_event = cancel_event
                    assert model is not None
                    chunks = model.generate_audio_stream(
                        model_state=state,
                        text_to_generate=text,
                    )
                    for chunk in chunks:
                        if cancel_event.is_set():
                            continue
                        if hasattr(chunk, "detach"):
                            chunk = chunk.detach().cpu().numpy()
                        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if audio.size == 0:
                            continue
                        np.clip(audio, -1.0, 1.0, out=audio)
                        pcm = (audio * 32767.0).astype(np.int16).tobytes()
                        loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
            except PocketGenerationCancelled:
                logger.info("Cancelled Pocket request %s", request_id or "<anonymous>")
            except Exception:
                logger.exception("Pocket generation failed for %s", request_id or "<anonymous>")
            finally:
                cancel_event.set()
                active_generation_cancel_event = None
                if request_id:
                    with cancel_lock:
                        cancelled_requests.discard(request_id)
                        request_cancel_events.pop(request_id, None)
                close = getattr(chunks, "close", None)
                if close:
                    close()
                loop.call_soon_threadsafe(audio_queue.put_nowait, end_of_stream)

        producer = threading.Thread(target=produce_audio, daemon=True)
        producer.start()
        try:
            while True:
                item = await audio_queue.get()
                if item is end_of_stream:
                    break
                yield item
        finally:
            cancel_event.set()
            await asyncio.to_thread(producer.join, 2.0)
            if producer.is_alive():
                logger.error("Pocket producer did not stop within two seconds.")

    return StreamingResponse(
        raw_pcm_stream(),
        media_type="application/octet-stream",
        headers={
            "X-Audio-Sample-Rate": "24000",
            "X-Audio-Channels": "1",
            "Cache-Control": "no-store",
        },
    )


@app.post("/cancel")
def cancel(payload: dict = Body(default_factory=dict)):
    """Invalidate speech without waiting for the stateful model lock."""
    request_ids = {
        str(value).strip()
        for value in payload.get("request_ids", [])
        if str(value).strip()
    }
    if request_ids:
        with cancel_lock:
            cancelled_requests.update(request_ids)
            for request_id in request_ids:
                event = request_cancel_events.get(request_id)
                if event is not None:
                    event.set()
    return {"status": "cancelled", "request_ids": sorted(request_ids)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--language", default="english")
    parser.add_argument("--voice", default="azelma")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    global model, default_voice
    default_voice = args.voice
    logger.info("Loading Pocket TTS on CPU (language=%s)...", args.language)
    model = TTSModel.load_model(language=args.language, quantize=args.quantize)
    _install_generation_cancellation()
    _state_for_voice(default_voice)
    logger.info("Pocket TTS worker ready on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
