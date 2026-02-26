"""Test WebSocket connection to the AI server from the command line."""

import argparse
import asyncio
import struct
import sys
import time
import math

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


def generate_sine_tone(
    frequency: float = 440.0,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a sine wave as 16-bit PCM bytes (simulates mic input)."""
    num_samples = int(sample_rate * duration_s)
    pcm = bytearray()
    for i in range(num_samples):
        value = math.sin(2 * math.pi * frequency * i / sample_rate)
        sample = int(max(-32768, min(32767, value * 16000)))
        pcm.extend(struct.pack("<h", sample))
    return bytes(pcm)


async def test_connection(url: str, timeout: float = 10.0):
    """Connect, send test audio, and report results."""
    print(f"Connecting to {url}...")

    try:
        async with websockets.connect(url) as ws:
            print("Connected!")

            # Send a short test tone
            print("Sending 2-second test tone (440Hz sine wave)...")
            audio = generate_sine_tone(440.0, 2.0, 16000)

            chunk_size = 3200  # 100ms at 16kHz, 16-bit
            chunks_sent = 0
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                await ws.send(chunk)
                chunks_sent += 1
                await asyncio.sleep(0.1)

            print(f"Sent {chunks_sent} chunks ({len(audio)} bytes total).")

            # Wait for response
            print(f"Waiting for response (timeout: {timeout}s)...")
            received_bytes = 0
            start = time.time()

            try:
                while time.time() - start < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(msg, bytes):
                            received_bytes += len(msg)
                            if received_bytes <= len(msg):
                                elapsed = time.time() - start
                                print(f"  First audio response after {elapsed:.2f}s")
                            print(f"  Received audio chunk: {len(msg)} bytes "
                                  f"(total: {received_bytes} bytes)")
                        else:
                            print(f"  Received text: {msg}")
                    except asyncio.TimeoutError:
                        if received_bytes > 0:
                            break
                        continue
            except websockets.exceptions.ConnectionClosed:
                pass

            if received_bytes > 0:
                duration_s = received_bytes / (24000 * 2)  # 24kHz, 16-bit
                print(f"\nSuccess! Received {received_bytes} bytes "
                      f"(~{duration_s:.1f}s of audio at 24kHz)")
            else:
                print("\nNo audio response received. Check that the server "
                      "pipeline is running and models are loaded.")

    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running on {url}?")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test AI server WebSocket connection.")
    parser.add_argument(
        "--url", type=str, default="ws://localhost:8765",
        help="WebSocket server URL (default: ws://localhost:8765)",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0,
        help="Response timeout in seconds (default: 15)",
    )
    args = parser.parse_args()

    asyncio.run(test_connection(args.url, args.timeout))


if __name__ == "__main__":
    main()
