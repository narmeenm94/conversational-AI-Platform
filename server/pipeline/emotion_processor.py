"""Optional sentiment / emotion analysis for user utterances."""

import logging

logger = logging.getLogger(__name__)


class EmotionProcessor:
    """Analyzes user text for emotional tone and returns a context string
    that can be injected into the LLM system prompt.

    When disabled, all methods are no-ops returning neutral defaults.
    """

    def __init__(self, enabled: bool = False, model_name: str | None = None):
        self._enabled = enabled
        self._analyzer = None

        if not enabled:
            logger.info("Emotion analysis disabled.")
            return

        try:
            from transformers import pipeline as hf_pipeline

            model = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
            logger.info("Loading emotion model: %s", model)
            self._analyzer = hf_pipeline(
                "sentiment-analysis", model=model, device=0
            )
            logger.info("Emotion model loaded.")
        except Exception as e:
            logger.warning("Failed to load emotion model, disabling: %s", e)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def analyze(self, text: str) -> dict:
        """Return emotion label and confidence score.

        Returns {"label": "neutral", "score": 0.0} when disabled or on error.
        """
        if not self._enabled or self._analyzer is None:
            return {"label": "neutral", "score": 0.0}

        try:
            results = self._analyzer(text[:512])
            if results:
                return {
                    "label": results[0]["label"],
                    "score": round(results[0]["score"], 3),
                }
        except Exception as e:
            logger.warning("Emotion analysis failed: %s", e)

        return {"label": "neutral", "score": 0.0}

    def get_emotion_context(self, text: str, turn_count: int = 0) -> str:
        """Return a context block for injection into the LLM system prompt.

        Returns an empty string when disabled.
        """
        if not self._enabled:
            return ""

        emotion = self.analyze(text)
        return (
            f"\n[CONVERSATION CONTEXT]\n"
            f"- User's current emotional state: {emotion['label']} "
            f"(confidence: {emotion['score']})\n"
            f"- Conversation turn: {turn_count}\n"
            f"- Adjust your emotional tone accordingly.\n"
        )
