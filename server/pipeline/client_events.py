"""Small versioned control protocol for Unity alongside raw PCM audio."""

import re

from pipecat.frames.frames import (
    Frame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ClientEventProcessor(FrameProcessor):
    """Mirrors useful pipeline state to Unity as compact JSON messages."""

    @staticmethod
    def _cognitive_state(text: str) -> str:
        value = str(text or "")
        if re.search(
            r"(?i)\b(?:remember|recall|earlier|before|last time|what did|you said|"
            r"continue|go on|and then)\b",
            value,
        ):
            return "remembering"
        if re.search(
            r"(?i)\b(?:search|look up|find|knowledge|source|website|team|staff|"
            r"colleagues?|who works|role|roles|hxrc)\b",
            value,
        ):
            return "searching"
        return "thinking"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        message = None
        animation = None
        if isinstance(frame, VADUserStartedSpeakingFrame):
            message = {"v": 1, "type": "user_speech_started"}
            animation = "listening"
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            message = {"v": 1, "type": "user_speech_stopped"}
            animation = "thinking"
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            message = {"v": 1, "type": "user_transcript", "text": frame.text.strip()}
            animation = self._cognitive_state(frame.text)

        if message:
            await self.push_frame(OutputTransportMessageUrgentFrame(message=message), direction)
        if animation:
            await self.push_frame(
                OutputTransportMessageUrgentFrame(message={
                    "v": 1,
                    "type": "assistant_animation",
                    "state": animation,
                    # Zero tells Unity to use the active character's configured
                    # crossfade instead of overriding it with a server constant.
                    "blend_seconds": 0.0,
                }),
                direction,
            )
        await self.push_frame(frame, direction)
