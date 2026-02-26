"""STT service — thin re-export of Pipecat's built-in Faster Whisper integration.

Pipecat ships WhisperSTTService which handles:
  - Model loading (Faster Whisper backend, CUDA or CPU)
  - VAD-based endpointing (waits for user to stop speaking)
  - Yielding TranscriptionFrame objects downstream

No custom wrapper needed. This module exists so main.py can import
from a consistent `pipeline.*` namespace and so we have one place
to adjust if the upstream import path changes.
"""

from pipecat.services.whisper import WhisperSTTService, Model  # noqa: F401
from pipecat.transcriptions.language import Language  # noqa: F401
