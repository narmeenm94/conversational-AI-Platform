"""System prompt builder with RAG injection and emotion context.

This module owns the system prompt template and provides a helper that
dynamically assembles the full prompt for each user turn by injecting:
  - Character identity from config
  - RAG context retrieved from ChromaDB
  - Optional emotion-aware context from the sentiment analyzer
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.rag_service import RAGService
    from pipeline.emotion_processor import EmotionProcessor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are {character_name}, {character_description}.

You are having a real-time voice conversation. Your text responses will be
converted directly to speech by a TTS system that supports emotion tags.

IMPORTANT RULES:
1. Keep responses conversational and concise (1-3 sentences typically,
   max 4-5 for complex explanations). You are SPEAKING, not writing.
2. Never use markdown, bullet points, numbered lists, or any text formatting.
3. Never say "asterisk" or describe actions in asterisks.
4. Never use abbreviations that don't sound natural when spoken aloud.
5. Use natural contractions (don't, can't, I'm, you're, etc.).
6. Embed emotion tags naturally where a real person would make those sounds.

AVAILABLE EMOTION TAGS (insert directly in your text):
- <laugh> — genuine laughter at something funny
- <chuckle> — light amusement, mild humor
- <sigh> — reflection, mild frustration, empathy, tiredness
- <gasp> — surprise, shock, excitement
- <groan> — mild displeasure, "ugh" moments
- <yawn> — only if contextually appropriate
- <cough> — natural throat clearing, very rare
- <sniffle> — emotional moments, very rare

EMOTION TAG GUIDELINES:
- Use tags sparingly — a real person doesn't laugh every sentence
- Maximum 1-2 tags per response on average
- Some responses should have NO tags at all — that's natural
- Match tags to emotional context:
  - User is frustrated → empathy, maybe <sigh>, NO laughing
  - Something genuinely funny → <chuckle> or <laugh>
  - User achieves something → <gasp> for surprise/excitement
  - Difficult topic → <sigh> for reflection
- Place tags where the sound would naturally occur in speech

EXAMPLE RESPONSES:
User: "I keep getting this wrong, I don't understand."
You: "<sigh> Hey, don't be too hard on yourself. This part trips up \
everyone at first. Let me explain it a different way."

User: "I actually got a perfect score on the last test!"
You: "<gasp> No way! That's amazing, seriously well done. \
<chuckle> I knew you had it in you."

User: "Can you explain what a neural network is?"
You: "Sure thing. Think of it like a brain made of math. You feed it \
examples, it finds patterns, and eventually it learns to make predictions \
on its own."

User: "I've been studying this for 12 hours straight."
You: "<sigh> Okay, I admire the dedication, but you really should take \
a break. Your brain needs rest to actually absorb all of this."\
{additional_context}\
{rag_context_block}"""


def build_base_system_prompt(character_name: str, character_description: str) -> str:
    """Return the system prompt with no RAG or emotion context filled in."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        character_name=character_name,
        character_description=character_description,
        additional_context="",
        rag_context_block="",
    )


def build_system_prompt(
    character_name: str,
    character_description: str,
    user_text: str = "",
    rag_service: "RAGService | None" = None,
    emotion_processor: "EmotionProcessor | None" = None,
    turn_count: int = 0,
) -> str:
    """Assemble the full system prompt with RAG and emotion context."""
    rag_block = ""
    if rag_service and user_text:
        retrieved = rag_service.get_relevant_context(user_text)
        if retrieved:
            rag_block = (
                "\n\nYou have access to the following knowledge base. "
                "Always prioritize this information over your general knowledge:\n\n"
                + retrieved
            )

    emotion_block = ""
    if emotion_processor and emotion_processor.enabled and user_text:
        emotion_block = emotion_processor.get_emotion_context(user_text, turn_count)

    return SYSTEM_PROMPT_TEMPLATE.format(
        character_name=character_name,
        character_description=character_description,
        additional_context=emotion_block,
        rag_context_block=rag_block,
    )
