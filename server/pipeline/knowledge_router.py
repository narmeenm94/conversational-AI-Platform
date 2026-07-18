"""Fast lexical routing for character-scoped knowledge retrieval."""

from __future__ import annotations

import re

from pipeline.speech_text import is_continuation_request


# Generic words such as "project", "research", and "website" are deliberately
# absent. They occur constantly in ordinary social conversation and previously
# caused unrelated workplace documents to leak into project brainstorming.
_EXPLICIT_KNOWLEDGE_TOPIC = re.compile(
    r"(?i)\b(?:hxrc|helsinki\s+xr|team|staff|colleagues?|coworkers?|"
    r"who\s+works|role|roles|"
    r"tiina|santeri|narmeen|janina|mikko|emmi|jussi|juho|julia|janset|"
    r"erson|henri|alicia|leevi|emilia|irina|torkkel|laaninen)\b"
)

_KNOWLEDGE_FOLLOW_UP = re.compile(
    r"(?i)^\s*(?:(?:who|anyone|anybody|someone|somebody)\s+else|"
    r"(?:what|which)\s+(?:projects?|research|work|roles?)\b)"
)


def should_retrieve_character_knowledge(
    user_text: str,
    prior_user_turns: list[str],
) -> bool:
    """Return true only for an explicit character-KB topic or its follow-up."""
    if _EXPLICIT_KNOWLEDGE_TOPIC.search(str(user_text or "")):
        return True
    prior_topic = any(
        _EXPLICIT_KNOWLEDGE_TOPIC.search(str(turn or ""))
        for turn in prior_user_turns
    )
    follow_up = bool(
        is_continuation_request(user_text)
        or _KNOWLEDGE_FOLLOW_UP.search(str(user_text or ""))
    )
    return prior_topic and follow_up
