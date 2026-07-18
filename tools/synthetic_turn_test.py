"""Measure one deterministic audio turn through the live WebSocket pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets
import soundfile as sf
import httpx
from scipy.signal import resample_poly


def load_pcm16(path: Path, target_rate: int = 16000) -> bytes:
    try:
        with wave.open(str(path), "rb") as source:
            rate = source.getframerate()
            channels = source.getnchannels()
            width = source.getsampwidth()
            if width != 2:
                raise wave.Error("not PCM16")
            audio = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    except wave.Error:
        audio_float, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=1)
        audio = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)
    if rate != target_rate:
        audio = resample_poly(audio.astype(np.float32), target_rate, rate)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
    return audio.tobytes()


def generate_test_speech(text: str, worker_url: str) -> bytes:
    chunks = []
    with httpx.stream(
        "POST",
        worker_url.rstrip("/") + "/tts",
        json={"text": text, "voice": "azelma"},
        timeout=None,
    ) as response:
        response.raise_for_status()
        chunks.extend(response.iter_bytes())
    audio_24k = np.frombuffer(b"".join(chunks), dtype=np.int16)
    audio_16k = resample_poly(audio_24k.astype(np.float32), 2, 3)
    return np.clip(audio_16k, -32768, 32767).astype(np.int16).tobytes()


async def run(url: str, speech: bytes) -> None:
    started = None
    last_audio = 0.0
    first_audio = None
    transcript_at = None
    spoken_at = None
    transcript = ""
    spoken = ""

    async with websockets.connect(url, ping_interval=30, ping_timeout=120) as socket:
        async def receive():
            nonlocal last_audio, first_audio, transcript_at, spoken_at, transcript, spoken
            async for message in socket:
                now = time.perf_counter()
                if isinstance(message, bytes):
                    last_audio = now
                    if started is not None and first_audio is None:
                        first_audio = now
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "user_transcript" and started is not None:
                    transcript_at = now
                    transcript = event.get("text", "")
                if event.get("type") == "assistant_spoken_text" and started is not None:
                    spoken_at = now
                    spoken = event.get("text", "")

        receiver = asyncio.create_task(receive())
        # Drain the proactive greeting before timing the synthetic user turn.
        deadline = time.perf_counter() + 12.0
        while time.perf_counter() < deadline:
            await asyncio.sleep(0.1)
            if last_audio and time.perf_counter() - last_audio > 0.8:
                break

        chunk_bytes = 1600 * 2  # 100 ms at 16 kHz PCM16
        silence = bytes(chunk_bytes)
        for _ in range(2):
            await socket.send(silence)
            await asyncio.sleep(0.1)
        started = time.perf_counter()
        for offset in range(0, len(speech), chunk_bytes):
            await socket.send(speech[offset : offset + chunk_bytes])
            await asyncio.sleep(0.1)
        speech_ended = time.perf_counter()
        for _ in range(6):
            await socket.send(silence)
            await asyncio.sleep(0.1)

        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline and (
            first_audio is None or transcript_at is None or spoken_at is None
        ):
            await asyncio.sleep(0.05)
        receiver.cancel()

    def since_end(value):
        return round(value - speech_ended, 3) if value is not None else None

    print(json.dumps({
        "input_seconds": round(len(speech) / 32000, 3),
        "transcript_after_end_seconds": since_end(transcript_at),
        "first_audio_after_end_seconds": since_end(first_audio),
        "spoken_event_after_end_seconds": since_end(spoken_at),
        "transcript": transcript,
        "first_spoken_phrase": spoken,
    }, indent=2, ensure_ascii=False))
    if first_audio is None or transcript_at is None:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, nargs="?")
    parser.add_argument("--text", default="How has your day been so far?")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8770")
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    args = parser.parse_args()
    speech = (
        load_pcm16(args.wav.resolve())
        if args.wav
        else generate_test_speech(args.text, args.worker_url)
    )
    asyncio.run(run(args.url, speech))


if __name__ == "__main__":
    main()
