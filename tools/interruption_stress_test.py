"""Exercise a rapid spoken correction while the avatar is answering."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import websockets

from synthetic_turn_test import generate_test_speech


async def run(url: str, first: bytes, correction: bytes) -> dict:
    events: list[tuple[float, str, str]] = []
    audio_at: list[float] = []

    async with websockets.connect(url, ping_interval=30, ping_timeout=120) as socket:
        async def receive():
            async for message in socket:
                now = time.perf_counter()
                if isinstance(message, bytes):
                    audio_at.append(now)
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                events.append((now, str(event.get("type", "")), str(event.get("text", ""))))

        receiver = asyncio.create_task(receive())

        # Drain the proactive greeting.
        await asyncio.sleep(4.0)
        audio_at.clear()
        events.clear()

        async def speak(pcm: bytes) -> float:
            chunk_bytes = 3200
            for _ in range(2):
                await socket.send(bytes(chunk_bytes))
                await asyncio.sleep(0.1)
            for offset in range(0, len(pcm), chunk_bytes):
                await socket.send(pcm[offset : offset + chunk_bytes])
                await asyncio.sleep(0.1)
            ended = time.perf_counter()
            for _ in range(5):
                await socket.send(bytes(chunk_bytes))
                await asyncio.sleep(0.1)
            return ended

        await speak(first)
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline and not any(
            kind == "assistant_spoken_text" for _, kind, _ in events
        ):
            await asyncio.sleep(0.02)

        correction_started = time.perf_counter()
        correction_ended = await speak(correction)

        deadline = time.perf_counter() + 10.0
        correction_transcript_at = None
        correction_text = ""
        correction_spoken_at = None
        correction_spoken = ""
        first_correction_audio = None
        while time.perf_counter() < deadline:
            for at, kind, text in events:
                if at < correction_started:
                    continue
                if kind == "user_transcript" and correction_transcript_at is None:
                    correction_transcript_at, correction_text = at, text
                if (
                    kind == "assistant_spoken_text"
                    and correction_transcript_at is not None
                    and at >= correction_transcript_at
                ):
                    correction_spoken_at, correction_spoken = at, text
            if correction_transcript_at is not None:
                first_correction_audio = next(
                    (at for at in audio_at if at >= correction_transcript_at),
                    None,
                )
            if correction_spoken_at is not None and first_correction_audio is not None:
                break
            await asyncio.sleep(0.02)
        receiver.cancel()

    def elapsed(value):
        return round(value - correction_ended, 3) if value is not None else None

    return {
        "correction_transcript_after_end_seconds": elapsed(correction_transcript_at),
        "correction_audio_after_end_seconds": elapsed(first_correction_audio),
        "correction_transcript": correction_text,
        "correction_first_spoken_phrase": correction_spoken,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8765")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8770")
    parser.add_argument("--first", default="So how was")
    parser.add_argument("--correction", default="Sorry, I meant, how is work today?")
    args = parser.parse_args()
    first = generate_test_speech(args.first, args.worker_url)
    correction = generate_test_speech(args.correction, args.worker_url)
    result = asyncio.run(run(args.url, first, correction))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["correction_audio_after_end_seconds"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
