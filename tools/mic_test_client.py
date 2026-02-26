"""Microphone test client — talk to the AI server from your terminal.

Records from your mic, streams audio to the server over WebSocket,
and plays back the AI's spoken response through your speakers in real-time
using a jitter buffer for smooth, continuous playback.

Usage:
    python tools/mic_test_client.py --url ws://IP:PORT

Press ENTER to start recording, ENTER again to stop and send.
Press Ctrl+C to quit.
"""

import argparse
import asyncio
import collections
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
BYTES_PER_SAMPLE = 2

PRE_BUFFER_SECONDS = 1.5
PRE_BUFFER_BYTES = int(PRE_BUFFER_SECONDS * RECV_SAMPLE_RATE * BYTES_PER_SAMPLE)

PLAYBACK_BLOCK_SAMPLES = 2400  # 100ms blocks at 24kHz
PLAYBACK_BLOCK_BYTES = PLAYBACK_BLOCK_SAMPLES * BYTES_PER_SAMPLE


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
    """Plays audio through speakers as chunks arrive, with a jitter buffer.

    Pre-buffers PRE_BUFFER_SECONDS of audio before starting playback.
    After that, plays continuously. If the buffer runs dry, inserts silence
    rather than stopping, so there are no pops or clicks.
    """

    def __init__(self, sample_rate: int = RECV_SAMPLE_RATE):
        self._sample_rate = sample_rate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._started = False
        self._finished_receiving = False
        self._stream = None
        self._play_thread = None
        self._total_received = 0
        self._first_chunk_time = None
        self._start_time = None
        self._underrun_count = 0

    def add_chunk(self, data: bytes):
        """Add an audio chunk from the server."""
        with self._lock:
            self._buffer.extend(data)
            self._total_received += len(data)

            if self._first_chunk_time is None:
                self._first_chunk_time = time.time()

            if not self._started and len(self._buffer) >= PRE_BUFFER_BYTES:
                self._started = True
                self._start_time = time.time()
                self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
                self._play_thread.start()

    def finish(self):
        """Signal that all audio has been received."""
        self._finished_receiving = True
        if not self._started and self._total_received > 0:
            self._started = True
            self._start_time = time.time()
            self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self._play_thread.start()

    def wait_done(self):
        """Block until all audio has finished playing."""
        if self._play_thread:
            self._play_thread.join(timeout=120)

    def _play_loop(self):
        """Continuous playback loop running in a background thread."""
        silence = np.zeros(PLAYBACK_BLOCK_SAMPLES, dtype=np.int16)

        try:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=PLAYBACK_BLOCK_SAMPLES,
            )
            stream.start()
            self._stream = stream

            while True:
                with self._lock:
                    available = len(self._buffer)
                    done = self._finished_receiving

                if available >= PLAYBACK_BLOCK_BYTES:
                    with self._lock:
                        chunk = bytes(self._buffer[:PLAYBACK_BLOCK_BYTES])
                        del self._buffer[:PLAYBACK_BLOCK_BYTES]

                    audio = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(audio)

                elif done and available > 0:
                    with self._lock:
                        chunk = bytes(self._buffer)
                        self._buffer.clear()
                    audio = np.frombuffer(chunk, dtype=np.int16)
                    padded = np.zeros(PLAYBACK_BLOCK_SAMPLES, dtype=np.int16)
                    padded[:len(audio)] = audio
                    stream.write(padded)

                elif done:
                    break

                else:
                    self._underrun_count += 1
                    stream.write(silence)

            stream.stop()
            stream.close()

        except Exception as e:
            print(f"  Playback error: {e}")

    @property
    def stats(self) -> str:
        duration = self._total_received / (self._sample_rate * BYTES_PER_SAMPLE)
        first_audio = ""
        if self._first_chunk_time and self._start_time:
            first_audio = f", first audio in {self._first_chunk_time - self._start_time + PRE_BUFFER_SECONDS:.2f}s"
        return f"{duration:.1f}s of audio, {self._underrun_count} underruns{first_audio}"


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

            duration = len(pcm) / (SEND_SAMPLE_RATE * BYTES_PER_SAMPLE)
            print(f"Sending {duration:.1f}s of audio...")

            chunk_size = 3200
            for i in range(0, len(pcm), chunk_size):
                await ws.send(pcm[i : i + chunk_size])
                await asyncio.sleep(0.05)

            print("Waiting for AI response...")
            player = StreamingPlayer()
            send_time = time.time()
            first_byte_time = None

            try:
                while time.time() - send_time < 180:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                        if isinstance(msg, bytes) and len(msg) > 0:
                            if first_byte_time is None:
                                first_byte_time = time.time() - send_time
                                print(f"  First audio byte at {first_byte_time:.2f}s")
                            player.add_chunk(msg)
                        elif isinstance(msg, str):
                            print(f"  Server: {msg}")
                    except asyncio.TimeoutError:
                        if player._total_received > 0:
                            break
                        continue
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                return

            player.finish()

            if player._total_received > 0:
                total = time.time() - send_time
                print(f"  All audio received in {total:.1f}s — streaming playback...")
                player.wait_done()
                print(f"  Done. {player.stats}")
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
