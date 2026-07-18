"""Quick microphone diagnostic.

Lists your audio input devices, records 3 seconds from the default device,
and prints the peak/RMS level so we know whether your mic is actually
capturing your voice.

Usage:
    python tools/mic_check.py
"""

import sys
import time

try:
    import numpy as np
    import sounddevice as sd
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install sounddevice numpy")
    sys.exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 3.0


def list_devices() -> None:
    print("=== Audio input devices ===")
    default_in, _ = sd.default.device
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = "  <-- default" if idx == default_in else ""
            print(f"  [{idx}] {dev['name']} (in={dev['max_input_channels']}ch){marker}")
    print()


def record_and_measure() -> None:
    print(f"Recording {DURATION:.0f}s from default input device...")
    print("Speak NOW, normally and clearly: 'Hello, can you hear me?'")
    for i in range(3, 0, -1):
        print(f"  starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  >>> SPEAK NOW <<<                ")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("  ...done.\n")

    samples = audio.flatten().astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples**2)))
    peak_db = 20 * np.log10(peak + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)

    print("=== Mic level ===")
    print(f"  peak amplitude: {peak:.4f}  ({peak_db:+.1f} dBFS)")
    print(f"  rms  amplitude: {rms:.4f}  ({rms_db:+.1f} dBFS)")
    print()

    if peak < 0.01:
        verdict = "SILENT — mic is not picking up your voice."
    elif peak < 0.05:
        verdict = "VERY QUIET — Whisper will likely treat this as silence."
    elif peak < 0.2:
        verdict = "Low but probably usable. Try speaking closer / louder."
    else:
        verdict = "Good level. Mic is working well."
    print(f"  verdict: {verdict}")


def main() -> None:
    list_devices()
    record_and_measure()


if __name__ == "__main__":
    main()
