"""Microphone test client — talk to the AI server from your terminal.

Records from your mic, streams audio to the server over WebSocket,
and plays back the AI's spoken response through your speakers in real-time.

Usage:
    python tools/mic_test_client.py --url ws://IP:PORT

Press ENTER to start recording, ENTER again to stop and send.
Press Ctrl+C to quit.
"""

import argparse
import asyncio
import sys
import threading
import time

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


def record_until_enter(sample_rate: int = SEND_SAMPLE_RATE) -> bytes:
    """Record from the microphone until the user presses ENTER."""
    frames = []
    recording = True

    def callback(indata, frame_count, time_info, status):
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
        blocksize=int(sample_rate * 0.1),
    )

    stream.start()
    input()
    recording = False
    stream.stop()
    stream.close()

    if not frames:
        return b""

    audio = np.concatenate(frames, axis=0)
    return audio.tobytes()


class StreamingPlayer:
    """Plays audio chunks in real-time as they arrive."""

    def __init__(self, sample_rate: int = RECV_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream = None
        self._playing = False

    def start(self):
        self._playing = True
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=2400,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frame_count, time_info, status):
        bytes_needed = frame_count * 2  # int16 = 2 bytes per sample
        with self._lock:
            available = len(self._buffer)
            if available >= bytes_needed:
                chunk = bytes(self._buffer[:bytes_needed])
                del self._buffer[:bytes_needed]
            elif available > 0:
                chunk = bytes(self._buffer) + b'\x00' * (bytes_needed - available)
                self._buffer.clear()
            else:
                chunk = b'\x00' * bytes_needed

        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)

    def feed(self, data: bytes):
        with self._lock:
            self._buffer.extend(data)

    def stop(self):
        if self._stream:
            remaining = True
            while remaining:
                time.sleep(0.05)
                with self._lock:
                    remaining = len(self._buffer) > 0
            time.sleep(0.2)
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._playing = False

    @property
    def buffered_seconds(self):
        with self._lock:
            return len(self._buffer) / (self._sample_rate * 2)


async def session(url: str):
    """Run one interactive session."""
    print(f"Connecting to {url}...")

    async with websockets.connect(url, ping_interval=30, ping_timeout=120) as ws:
        print("Connected to server!\n")

        while True:
            print("--- Press ENTER to start recording (Ctrl+C to quit) ---")
            input()
            print("Recording... speak now. Press ENTER when done.")

            pcm = record_until_enter()

            if len(pcm) < 3200:
                print("Too short, try again.\n")
                continue

            duration = len(pcm) / (SEND_SAMPLE_RATE * 2)
            print(f"Sending {duration:.1f}s of audio...")

            chunk_size = 3200
            for i in range(0, len(pcm), chunk_size):
                await ws.send(pcm[i : i + chunk_size])
                await asyncio.sleep(0.05)

            print("Waiting for AI response...")
            player = StreamingPlayer()
            first_byte_time = None
            start = time.time()
            total_bytes = 0

            try:
                while time.time() - start < 180:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        if isinstance(msg, bytes) and len(msg) > 0:
                            if first_byte_time is None:
                                first_byte_time = time.time() - start
                                print(f"  First audio in {first_byte_time:.2f}s — playing...")
                                player.start()
                            player.feed(msg)
                            total_bytes += len(msg)
                        elif isinstance(msg, str):
                            print(f"  Server: {msg}")
                    except asyncio.TimeoutError:
                        if total_bytes > 0:
                            break
                        continue
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                return

            if total_bytes > 0:
                resp_duration = total_bytes / (RECV_SAMPLE_RATE * 2)
                print(f"  Received {total_bytes} bytes ({resp_duration:.1f}s of audio)")
                player.stop()
            else:
                print("  No audio response received.")

            print()


def main():
    parser = argparse.ArgumentParser(description="Mic test client for Conversational AI server.")
    parser.add_argument("--url", default="ws://localhost:8765", help="Server WebSocket URL")
    args = parser.parse_args()

    try:
        asyncio.run(session(args.url))
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
