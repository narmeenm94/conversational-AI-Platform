"""Microphone test client — talk to the AI server from your terminal.

Records from your mic, streams audio to the server over WebSocket,
and plays back the AI's spoken response through your speakers.

Usage:
    python tools/mic_test_client.py

Press ENTER to start recording, ENTER again to stop and send.
Press Ctrl+C to quit.
"""

import argparse
import asyncio
import struct
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


def play_audio(pcm_bytes: bytes, sample_rate: int = RECV_SAMPLE_RATE):
    """Play raw PCM audio through the speakers."""
    if not pcm_bytes:
        return
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate, blocking=True)


async def session(url: str):
    """Run one interactive session."""
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
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

            chunk_size = 3200  # 100ms chunks
            for i in range(0, len(pcm), chunk_size):
                await ws.send(pcm[i : i + chunk_size])
                await asyncio.sleep(0.05)

            print("Waiting for AI response (this may take a minute on first run)...")
            response_audio = bytearray()
            first_byte_time = None
            start = time.time()

            try:
                while time.time() - start < 120:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        if isinstance(msg, bytes) and len(msg) > 0:
                            if first_byte_time is None:
                                first_byte_time = time.time() - start
                                print(f"  First response in {first_byte_time:.2f}s")
                            response_audio.extend(msg)
                        elif isinstance(msg, str):
                            print(f"  Server: {msg}")
                    except asyncio.TimeoutError:
                        if len(response_audio) > 0:
                            break
                        continue
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                return

            if response_audio:
                resp_duration = len(response_audio) / (RECV_SAMPLE_RATE * 2)
                print(f"  Received {len(response_audio)} bytes ({resp_duration:.1f}s of audio)")
                print("  Playing response...")
                play_audio(bytes(response_audio))
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
