"""Benchmark end-to-end latency: audio in → first audio chunk out."""

import argparse
import asyncio
import math
import struct
import sys
import time
import statistics

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


def generate_speech_like_audio(
    duration_s: float = 1.5,
    sample_rate: int = 16000,
) -> bytes:
    """Generate audio that might trigger STT (a simple tone burst)."""
    num_samples = int(sample_rate * duration_s)
    pcm = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        # Mix of frequencies to be more voice-like
        value = (
            0.5 * math.sin(2 * math.pi * 200 * t)
            + 0.3 * math.sin(2 * math.pi * 400 * t)
            + 0.2 * math.sin(2 * math.pi * 800 * t)
        )
        # Apply envelope
        envelope = min(1.0, t * 10) * max(0.0, 1.0 - (t - duration_s + 0.1) * 10)
        sample = int(max(-32768, min(32767, value * envelope * 20000)))
        pcm.extend(struct.pack("<h", sample))
    return bytes(pcm)


async def measure_latency(url: str, response_timeout: float = 20.0) -> float | None:
    """Send audio and measure time to first response byte. Returns seconds or None."""
    try:
        async with websockets.connect(url) as ws:
            audio = generate_speech_like_audio(1.5, 16000)

            chunk_size = 3200
            for i in range(0, len(audio), chunk_size):
                await ws.send(audio[i : i + chunk_size])
                await asyncio.sleep(0.1)

            # Mark the end of sending
            send_done = time.perf_counter()

            # Wait for first response
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=response_timeout)
                recv_time = time.perf_counter()
                if isinstance(msg, bytes) and len(msg) > 0:
                    return recv_time - send_done
            except asyncio.TimeoutError:
                return None

            # Drain remaining messages
            try:
                while True:
                    await asyncio.wait_for(ws.recv(), timeout=2.0)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

    except Exception as e:
        print(f"  Error: {e}")
        return None

    return None


async def run_benchmark(url: str, iterations: int, timeout: float):
    print(f"Benchmarking latency: {url}")
    print(f"Iterations: {iterations}")
    print(f"Response timeout: {timeout}s")
    print("-" * 50)

    latencies = []

    for i in range(iterations):
        print(f"  Run {i + 1}/{iterations}...", end=" ", flush=True)
        latency = await measure_latency(url, timeout)

        if latency is not None:
            latencies.append(latency)
            print(f"{latency * 1000:.0f}ms")
        else:
            print("no response (timeout)")

        if i < iterations - 1:
            await asyncio.sleep(1.0)

    print("-" * 50)

    if not latencies:
        print("No successful measurements. Is the server running and models loaded?")
        return

    print(f"Results ({len(latencies)}/{iterations} successful):")
    print(f"  Min:    {min(latencies) * 1000:.0f}ms")
    print(f"  Max:    {max(latencies) * 1000:.0f}ms")
    print(f"  Mean:   {statistics.mean(latencies) * 1000:.0f}ms")
    if len(latencies) > 1:
        print(f"  Median: {statistics.median(latencies) * 1000:.0f}ms")
        print(f"  Stdev:  {statistics.stdev(latencies) * 1000:.0f}ms")

    target = 0.5
    under_target = sum(1 for l in latencies if l < target)
    print(f"\n  Under 500ms target: {under_target}/{len(latencies)}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AI server end-to-end latency.")
    parser.add_argument(
        "--url", type=str, default="ws://localhost:8765",
        help="WebSocket server URL",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--timeout", type=float, default=20.0,
        help="Response timeout per iteration in seconds (default: 20)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.iterations, args.timeout))


if __name__ == "__main__":
    main()
