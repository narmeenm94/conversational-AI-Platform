"""Continuous-streaming voice client for the Conversational AI server.

Talks to the server like a real conversation:
  - Microphone streams audio to the server continuously.
  - The server's VAD finds turn boundaries (no press-to-talk).
  - The bot's voice plays through your speakers in real-time.
  - A live mic-level meter prints below so you can see what's happening.

Echo handling (important on laptops without headphones!):
  - Default mode is HALF-DUPLEX: while the bot is speaking, the mic stops
    sending audio so the bot doesn't hear itself through your speakers.
    This prevents the bot from feeding back into its own conversation.
  - Use --barge-in to enable full-duplex with interruption (recommended
    only when wearing headphones).

Usage:
    python tools/mic_test_client.py --url ws://IP:PORT
    python tools/mic_test_client.py --device 7         # pick input device
    python tools/mic_test_client.py --barge-in         # headphones mode
    python tools/mic_test_client.py --list             # list audio devices

Press Ctrl+C to quit.
"""

import argparse
import asyncio
import collections
import math
import sys
import threading
import time
from typing import Optional

try:
    import numpy as np
    import sounddevice as sd
    import websockets
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install sounddevice numpy websockets")
    sys.exit(1)

SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = "int16"
BYTES_PER_SAMPLE = 2

# 100 ms mic blocks - matches what the server VAD likes.
MIC_BLOCK_SAMPLES = SEND_SAMPLE_RATE // 10
MIC_BLOCK_BYTES = MIC_BLOCK_SAMPLES * BYTES_PER_SAMPLE

# Playback uses one persistent output stream pulling from a queue.
PLAYBACK_BLOCK_SAMPLES = 480  # 20 ms at 24 kHz - tight, low-latency
PLAYBACK_BLOCK_BYTES = PLAYBACK_BLOCK_SAMPLES * BYTES_PER_SAMPLE
PRE_BUFFER_BYTES = int(0.15 * RECV_SAMPLE_RATE * BYTES_PER_SAMPLE)  # 150 ms

# Local barge-in: if the user starts talking, flush bot audio immediately.
# Threshold is on RMS of the int16 samples normalised to [-1, 1].
# This is high on purpose: it has to be loud, deliberate speech to
# beat ambient noise and any speaker bleed-through.
BARGE_IN_RMS = 0.12         # ~-18 dBFS - close-mic talking volume
BARGE_IN_HOLD_BLOCKS = 3    # 3 x 100 ms = 300 ms of speech to confirm

# Half-duplex tail: keep mic muted this long *after* bot stops speaking,
# so any room reverb / late audio buffer doesn't get picked up.
HALF_DUPLEX_TAIL_S = 0.4

# How much buffered bot audio counts as "bot is speaking".
BOT_SPEAKING_BUF_BYTES = int(0.05 * RECV_SAMPLE_RATE * BYTES_PER_SAMPLE)  # 50 ms

# Show a small live mic-level meter on stderr.
METER_INTERVAL_S = 0.1


# ──────────────────────── Playback ────────────────────────


class ContinuousPlayer:
    """One persistent OutputStream + a queue of int16 chunks."""

    def __init__(self, sample_rate: int = RECV_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._priming = True
        self._silence = np.zeros(PLAYBACK_BLOCK_SAMPLES, dtype=np.int16)
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=PLAYBACK_BLOCK_SAMPLES,
            latency="low",
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def add_chunk(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)
            if self._priming and len(self._buffer) >= PRE_BUFFER_BYTES:
                self._priming = False

    def flush(self) -> None:
        """Drop any pending bot audio (used for barge-in)."""
        with self._lock:
            self._buffer.clear()
            self._priming = True

    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._buffer)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ready = (not self._priming) and len(self._buffer) >= PLAYBACK_BLOCK_BYTES
                if ready:
                    chunk = bytes(self._buffer[:PLAYBACK_BLOCK_BYTES])
                    del self._buffer[:PLAYBACK_BLOCK_BYTES]
                else:
                    chunk = None

            if chunk is not None:
                self._stream.write(np.frombuffer(chunk, dtype=np.int16))
            else:
                self._stream.write(self._silence)


# ──────────────────────── Mic + meter ────────────────────────


class MicLevelTracker:
    """Tracks RMS of recent mic blocks for local barge-in detection."""

    def __init__(self):
        self._recent = collections.deque(maxlen=BARGE_IN_HOLD_BLOCKS)
        self._last_rms = 0.0

    def update(self, samples_f32: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(samples_f32 ** 2))) if samples_f32.size else 0.0
        self._last_rms = rms
        self._recent.append(rms)
        return rms

    @property
    def last_rms(self) -> float:
        return self._last_rms

    def speaking(self) -> bool:
        if len(self._recent) < BARGE_IN_HOLD_BLOCKS:
            return False
        return all(r >= BARGE_IN_RMS for r in self._recent)


def _meter_bar(rms: float, width: int = 24) -> str:
    if rms <= 0:
        db = -80.0
    else:
        db = 20.0 * math.log10(rms)
    db = max(-60.0, min(0.0, db))
    fill = int(round((db + 60.0) / 60.0 * width))
    return "[" + "#" * fill + "-" * (width - fill) + f"] {db:+.0f} dB"


# ──────────────────────── Session ────────────────────────


async def session(url: str, device: Optional[int], barge_in: bool) -> None:
    print(f"Connecting to {url} ...")
    async with websockets.connect(url, ping_interval=30, ping_timeout=120) as ws:
        mode = "FULL-DUPLEX (barge-in)" if barge_in else "HALF-DUPLEX (no echo)"
        print(f"Connected. Mode: {mode}")
        if not barge_in:
            print("Tip: use --barge-in if you wear headphones for instant interruption.")
        print("Just talk. Press Ctrl+C to quit.\n")

        send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()
        level = MicLevelTracker()
        player = ContinuousPlayer()
        silence_block = bytes(MIC_BLOCK_BYTES)
        stats = {"sent": 0, "muted": 0, "interrupts": 0}
        bot_audio_until = [0.0]  # mutable so closures can update it

        def mic_callback(indata, frames, time_info, status):  # noqa: ARG001
            samples_i16 = indata.copy().reshape(-1)
            samples_f32 = samples_i16.astype(np.float32) / 32768.0
            level.update(samples_f32)

            now = time.time()
            buffered = player.buffered_bytes()
            bot_speaking = buffered >= BOT_SPEAKING_BUF_BYTES or now < bot_audio_until[0]

            if bot_speaking and not barge_in:
                # Half-duplex: send a silent block so the server's VAD
                # never sees our captured speaker bleed-through.
                payload = silence_block
                stats["muted"] += 1
            else:
                payload = samples_i16.tobytes()
                stats["sent"] += 1

                # Local barge-in: only when the user has explicitly enabled
                # full-duplex AND is loud + sustained enough.
                if barge_in and bot_speaking and level.speaking():
                    player.flush()
                    stats["interrupts"] += 1

            try:
                loop.call_soon_threadsafe(send_queue.put_nowait, payload)
            except RuntimeError:
                pass

        in_stream = sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=MIC_BLOCK_SAMPLES,
            device=device,
            callback=mic_callback,
        )

        async def sender():
            while True:
                chunk = await send_queue.get()
                await ws.send(chunk)

        async def receiver():
            async for msg in ws:
                if isinstance(msg, bytes) and len(msg) > 0:
                    bot_audio_until[0] = time.time() + HALF_DUPLEX_TAIL_S
                    player.add_chunk(msg)
                elif isinstance(msg, str):
                    sys.stderr.write("\n")
                    print(f"[server] {msg}")

        async def meter():
            sys.stderr.write("\n")
            while True:
                rms = level.last_rms
                buf_ms = (player.buffered_bytes() / (RECV_SAMPLE_RATE * BYTES_PER_SAMPLE)) * 1000
                bot = "SPEAKING" if buf_ms > 50 or time.time() < bot_audio_until[0] else "       "
                muted_marker = " [MUTED]" if (bot != "       " and not barge_in) else "        "
                bar = _meter_bar(rms)
                sys.stderr.write(
                    f"\rmic {bar}{muted_marker}  bot {bot}  buf {buf_ms:5.0f}ms  ints {stats['interrupts']:>2d}"
                )
                sys.stderr.flush()
                await asyncio.sleep(METER_INTERVAL_S)

        in_stream.start()
        try:
            await asyncio.gather(sender(), receiver(), meter())
        finally:
            in_stream.stop()
            in_stream.close()
            player.close()


# ──────────────────────── Entry point ────────────────────────


def list_devices() -> None:
    print("=== Audio input devices ===")
    default_in, _ = sd.default.device
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = "  <-- default" if idx == default_in else ""
            print(f"  [{idx}] {dev['name']} (in={dev['max_input_channels']}ch){mark}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous-streaming voice client for the Conversational AI server."
    )
    parser.add_argument("--url", default="ws://localhost:8765", help="Server WebSocket URL")
    parser.add_argument("--device", type=int, default=None, help="Input device index (see --list)")
    parser.add_argument("--list", action="store_true", help="List audio input devices and exit")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Full-duplex mode with interruption (use with headphones to avoid echo).",
    )
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    try:
        asyncio.run(session(args.url, args.device, args.barge_in))
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        print("Goodbye!")
    except websockets.exceptions.ConnectionClosed as e:
        sys.stderr.write("\n")
        print(f"Connection closed: {e}")


if __name__ == "__main__":
    main()
