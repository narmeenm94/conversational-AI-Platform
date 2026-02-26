"""Raw PCM audio serializer for Pipecat WebSocket transport.

Converts between raw PCM bytes on the wire and Pipecat audio frames.
This is the simplest possible serializer: binary messages are audio,
text messages are ignored.
"""

import logging

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

logger = logging.getLogger(__name__)


class RawAudioSerializer(FrameSerializer):
    """Passes raw PCM audio bytes directly between WebSocket and pipeline.

    - Incoming binary messages → InputAudioRawFrame
    - OutputAudioRawFrame → outgoing binary messages
    - Everything else is dropped (no text protocol needed).
    """

    def __init__(self, sample_rate: int = 16000, num_channels: int = 1):
        super().__init__()
        self._sample_rate = sample_rate
        self._num_channels = num_channels

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes) and len(data) > 0:
            return InputAudioRawFrame(
                audio=data,
                sample_rate=self._sample_rate,
                num_channels=self._num_channels,
            )
        return None
