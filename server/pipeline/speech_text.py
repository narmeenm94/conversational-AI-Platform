"""Shared text cleanup and expression cues for local TTS backends."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


ORPHEUS_TAGS = {
    "laugh", "chuckle", "sigh", "gasp", "groan", "yawn", "cough", "sniffle",
}
CHATTERBOX_CUE_MAP = {
    "laugh": "laugh",
    "chuckle": "chuckle",
    "sigh": "sigh",
    "shush": "shush",
    "cough": "cough",
    "groan": "groan",
    "sniffle": "sniff",
    "sniff": "sniff",
    "gasp": "gasp",
    "clear-throat": "clear throat",
}

# Chatterbox Turbo is non-streaming: it must finish a complete waveform before
# the first byte can be played. These short, broadly useful reactions are
# rendered once per character voice and served from a PCM cache. The LLM is
# given a very short hesitation from this cache only when a difficult turn is
# likely to take long enough to otherwise leave dead air. These are deliberately
# not complete canned acknowledgements: repeating those on every turn sounds far
# less human than a brief "hmm" or "uh".
CHATTERBOX_FAST_STARTERS = (
    "Fair point.",
    "Exactly.",
    "That tracks.",
    "Wait, seriously?",
    "You're not wrong.",
    "I know, right?",
    "Not a chance.",
    "Honestly, same.",
    "[chuckle] Okay, that's funny.",
    "[laugh] Okay, that's funny.",
    "[sigh] Fair enough.",
    "Fair enough.",
    "[gasp] No way!",
    "[groan] Oh, please.",
    "Go on.",
    "Hmm... uh... right...",
    "Uh... well... hmm...",
    "Ah... okay... hmm...",
    "Well... let's see... uh...",
)

CHATTERBOX_BACKCHANNELS = (
    "Hmm... uh... right...",
    "Uh... well... hmm...",
    "Ah... okay... hmm...",
    "Well... let's see... uh...",
)

_CONTINUATION_REQUEST_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"(?:(?:and|but|so)\s+)?(?:what|where|when|who|why|how)(?:\s+else)?\??"
    r"|(?:[\w'-]+\s+){1,3}(?:what|where|when|who|why|how)\??"
    r"|continue(?:\s+that)?"
    r"|go\s+on"
    r"|finish(?:\s+(?:that(?:\s+sentence)?|it|the\s+sentence|what\s+you\s+were\s+saying))?"
    r"|and\s+then\??"
    r"|you\s+were\s+saying(?:\s+what)?\??"
    r")\s*[.!?]*\s*$"
)


def is_continuation_request(user_text: str) -> bool:
    """Return true when the user is repairing or extending the prior sentence."""
    return bool(_CONTINUATION_REQUEST_RE.match(str(user_text or "")))


def should_use_instant_backchannel(user_text: str) -> bool:
    """Use cached hesitation only for turns likely to need real thinking time."""
    text = " ".join(str(user_text or "").split())
    if not text or is_continuation_request(text):
        return False
    words = re.findall(r"[\w']+", text, re.UNICODE)
    if len(words) < 4:
        return False
    question = "?" in text or bool(
        re.match(
            r"(?i)^(?:so\s+)?(?:who|what|where|when|why|how|which|can|could|would|do|does|is|are|tell|explain)\b",
            text,
        )
    )
    creative_request = bool(
        re.search(r"(?i)\b(?:joke|story|explain|describe|compare|opinion|think)\b", text)
    )
    return question and (len(words) >= 7 or creative_request)
ORPHEUS_CUE_MAP = {tag: tag for tag in ORPHEUS_TAGS}

_ANGLE_TAG_RE = re.compile(r"<\s*/?\s*([a-zA-Z][\w-]*)(?:\s+[^>]*)?\s*>")
_SQUARE_TAG_RE = re.compile(r"\[\s*([^\]\n]{1,80})\s*]")
_PARENTHETICAL_RE = re.compile(r"\(([^()\n]{1,100})\)")
_BOLD_MARKDOWN_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
_ITALIC_OR_ACTION_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_SPEAKABLE_RE = re.compile(r"[\w\d]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

_ABBREVIATIONS = (
    (re.compile(r"\be\s*\.\s*g\s*\.?(?=\W|$)", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\s*\.\s*e\s*\.?(?=\W|$)", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\s*\.(?=\W|$)", re.IGNORECASE), "and so on"),
    (re.compile(r"\bvs\s*\.(?=\W|$)", re.IGNORECASE), "versus"),
)

_ACTION_ALIASES = {
    "laugh": "laugh",
    "laughs": "laugh",
    "laughing": "laugh",
    "chuckle": "chuckle",
    "chuckles": "chuckle",
    "chuckling": "chuckle",
    "sigh": "sigh",
    "sighs": "sigh",
    "gasp": "gasp",
    "gasps": "gasp",
    "groan": "groan",
    "groans": "groan",
    "yawn": "yawn",
    "yawns": "yawn",
    "cough": "cough",
    "coughs": "cough",
    "sniffle": "sniffle",
    "sniffles": "sniffle",
    "sniff": "sniffle",
    "sniffs": "sniffle",
    "shush": "shush",
    "shushes": "shush",
}

_STAGE_DIRECTION_RE = re.compile(
    r"(?i)\b(?:wait(?:s|ed|ing)?|paus(?:e|es|ed|ing)|dramatic|reaction|"
    r"sarcastic\s+tone|tone|think(?:s|ing)?|smil(?:e|es|ed|ing)|"
    r"smirk(?:s|ed|ing)?|grin(?:s|ned|ning)?|nod(?:s|ded|ding)?|"
    r"shrug(?:s|ged|ging)?|whisper(?:s|ed|ing)?|shout(?:s|ed|ing)?|"
    r"roll(?:s|ed|ing)?\s+(?:my|her|his|their)?\s*eyes?|"
    r"shake(?:s|n|ing)?\s+(?:my|her|his|their)?\s*head|"
    r"stage\s+direction|beat|silence)\b"
)

_EMOJI_CUES = {
    "😂": "laugh", "🤣": "laugh", "😆": "laugh",
    "😁": "chuckle", "😄": "chuckle", "😊": "chuckle",
    "😮": "gasp", "😲": "gasp", "🤯": "gasp",
    "😔": "sigh", "😞": "sigh", "😢": "sniffle", "😭": "sniffle",
    "😤": "groan", "😠": "groan", "😡": "groan",
}
_EMOJI_CUES.update({
    "\U0001f602": "laugh", "\U0001f923": "laugh", "\U0001f606": "laugh",
    "\U0001f601": "chuckle", "\U0001f604": "chuckle", "\U0001f60a": "chuckle",
    "\U0001f62e": "gasp", "\U0001f632": "gasp", "\U0001f92f": "gasp",
    "\U0001f614": "sigh", "\U0001f61e": "sigh", "\U0001f622": "sniffle",
    "\U0001f62d": "sniffle", "\U0001f624": "groan", "\U0001f620": "groan",
    "\U0001f621": "groan",
})

_EMOTION_MAP = {
    "laugh": ("happy", 1.0),
    "chuckle": ("happy", 0.65),
    "gasp": ("surprised", 0.9),
    "sigh": ("sad", 0.45),
    "sniffle": ("sad", 0.65),
    "groan": ("annoyed", 0.65),
    "yawn": ("tired", 0.55),
    "cough": ("neutral", 0.25),
}


@dataclass(frozen=True)
class PreparedSpeech:
    text: str
    emotion: str = "neutral"
    intensity: float = 0.0

    @property
    def speakable(self) -> bool:
        return bool(_SPEAKABLE_RE.search(self.text))


def guard_verified_people(
    text: str,
    *,
    latest_user: str,
    verified_people: list[str],
    profile_name: str = "",
) -> tuple[str, list[str]]:
    """Replace a people claim containing names outside an explicit allowlist."""
    if not verified_people:
        return text, []
    people_turn = bool(re.search(
        r"(?i)\b(?:hxrc|helsinki\s+xr|team|staff|colleagues?|coworkers?|"
        r"who\s+else|anyone\s+else|who\s+works|names?|director|manager|ops)\b",
        str(latest_user or "") + " " + text,
    ))
    if not people_turn:
        return text, []

    allowed_tokens = {
        token.casefold()
        for name in list(verified_people) + [profile_name]
        for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ-]+", str(name))
    }
    ignored = {
        "and", "anyone", "center", "community", "communications",
        "developer", "development", "expert", "finnish", "generalist",
        "head", "helsinki", "i", "manager", "my", "project", "region",
        "research", "specialist", "technology", "the", "there", "unit",
        "urban", "web", "well", "we", "who", "xr",
    }
    candidates = re.findall(r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ-]{2,}\b", text)
    unknown = [
        token for token in candidates
        if token.casefold() not in allowed_tokens
        and token.casefold() not in ignored
    ]
    if not unknown:
        return text, []
    return "", unknown


def _find_tags(text: str) -> list[str]:
    tags = [_canonical_cue(m.group(1)) for m in _ANGLE_TAG_RE.finditer(text)] + [
        _canonical_cue(m.group(1)) for m in _SQUARE_TAG_RE.finditer(text)
    ]
    for match in _ITALIC_OR_ACTION_RE.finditer(text):
        action = _markdown_action(match.group(1))
        if action:
            tags.append(action)
    for match in _PARENTHETICAL_RE.finditer(text):
        action = _markdown_action(match.group(1))
        if action:
            tags.append(action)
    tags.extend(cue for emoji, cue in _EMOJI_CUES.items() if emoji in text)
    return tags


def _cue(tags: list[str]) -> tuple[str, float]:
    cues = [_EMOTION_MAP[tag] for tag in tags if tag in _EMOTION_MAP]
    return max(cues, key=lambda item: item[1]) if cues else ("neutral", 0.0)


def _markdown_action(value: str) -> str | None:
    """Return a supported stage cue, while leaving ordinary emphasis alone."""
    words = re.findall(r"[a-zA-Z]+", value.lower())
    if not words:
        return None
    for word in words:
        action = _ACTION_ALIASES.get(word)
        if action:
            return action
    lowered = value.lower()
    if "rolls eye" in lowered or "shakes head" in lowered:
        return "groan"
    if "smirk" in lowered or "grin" in lowered:
        return "chuckle"
    return None


def _canonical_cue(value: str) -> str:
    value = value.strip().lower().replace("_", "-").replace(" ", "-")
    if value in {"clear-throat", "clearthroat", "throat-clear"} or (
        "throat" in value and "clear" in value
    ):
        return "clear-throat"
    direct = _ACTION_ALIASES.get(value)
    if direct:
        return direct
    for word in re.findall(r"[a-zA-Z]+", value):
        action = _ACTION_ALIASES.get(word)
        if action:
            return action
    return value


def _format_cue(cue: str, cue_map: dict[str, str], style: str) -> str:
    output = cue_map.get(_canonical_cue(cue))
    if not output:
        return ""
    if style == "angle":
        return f"<{output}>"
    if style == "square":
        return f"[{output}]"
    return ""


def _replace_emojis(text: str, cue_map: dict[str, str], style: str) -> str:
    for emoji, cue in _EMOJI_CUES.items():
        text = text.replace(emoji, f" {_format_cue(cue, cue_map, style)} ")
    return text


def _replace_markdown(text: str, cue_map: dict[str, str], style: str) -> str:
    # Markdown bold is emphasis, not a stage direction: keep the words.
    text = _BOLD_MARKDOWN_RE.sub(lambda match: match.group(1), text)

    def replace(match: re.Match) -> str:
        action = _markdown_action(match.group(1))
        if not action:
            # Preserve one-word emphasis such as *important*, but never send an
            # unknown multiword stage direction to a voice model literally.
            words = re.findall(r"[a-zA-Z]+", match.group(1))
            return match.group(1) if len(words) <= 1 else ""
        return _format_cue(action, cue_map, style)

    return _ITALIC_OR_ACTION_RE.sub(replace, text)


def _replace_parenthetical_directions(text: str, cue_renderer) -> str:
    """Render recognized actions and drop unsupported stage directions.

    Ordinary explanatory parentheses remain intact. Only action-like text such
    as ``(laughs)`` or ``(pausing for effect)`` is treated as performance markup.
    """
    def replace(match: re.Match) -> str:
        value = match.group(1).strip()
        action = _markdown_action(value)
        if action:
            return cue_renderer(action)
        if _STAGE_DIRECTION_RE.search(value):
            return ""
        return match.group(0)

    return _PARENTHETICAL_RE.sub(replace, text)


def _normalize(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # Never pass Markdown punctuation to a speech model. Some voices pronounce
    # raw '*' as "asterisk" and backticks as literal formatting instructions.
    text = text.replace("*", "").replace("`", "")
    text = re.sub(r"(?<!\w)_{1,3}|_{1,3}(?!\w)", "", text)
    # Remove remaining pictographs/symbol emoji so a phonemizer never expands
    # their Unicode names into spoken phrases.
    text = "".join(
        character for character in text
        if unicodedata.category(character) not in {"So", "Cs"}
    )
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[,;:]\s*([.!?])", r"\1", text)
    text = re.sub(r"([!?])\.+", r"\1", text)
    return text


def _add_implicit_chatterbox_cue(text: str) -> str:
    """Add one restrained performance cue for high-signal emotional wording."""
    if any(tag in CHATTERBOX_CUE_MAP for tag in _find_tags(text)):
        return text
    lower = text.lower()
    cue = None
    if re.search(
        r"\b(joke|funny|hilarious|made me laugh|that's good|why did|knock knock|punchline)\b",
        lower,
    ):
        cue = "chuckle"
    elif re.search(r"\b(congratulations|congrats|amazing|awesome|you did it|finally works?)\b", lower):
        cue = "gasp"
    elif re.search(r"\b(i'm sorry|terrible|failed|failure|heartbreaking|that hurts|disappointed)\b", lower):
        cue = "sigh"
    elif re.search(r"\b(frustrating|frustrated|annoying|annoyed|ridiculous|angry|furious)\b", lower):
        cue = "groan"
    elif re.search(r"\b(no way|i can't believe|seriously\?)\b", lower):
        cue = "gasp"
    return f"[{cue}] {text}" if cue else text


def prepare_for_kokoro(text: str) -> PreparedSpeech:
    """Strip every stage direction; Kokoro would pronounce them literally."""
    tags = _find_tags(text)
    cleaned = _replace_emojis(text, {}, "strip")
    cleaned = _ANGLE_TAG_RE.sub("", cleaned)
    cleaned = _SQUARE_TAG_RE.sub("", cleaned)
    cleaned = _replace_markdown(cleaned, {}, "strip")
    cleaned = _replace_parenthetical_directions(cleaned, lambda _action: "")
    emotion, intensity = _cue(tags)
    return PreparedSpeech(_normalize(cleaned), emotion, intensity)


def prepare_for_pocket(text: str) -> PreparedSpeech:
    """Strip performance markup Pocket cannot execute natively.

    Cue intent still drives Unity's facial-expression event, but Pocket never
    receives a tag, action label, or synthetic substitute that it might read.
    """
    tags = _find_tags(text)
    # Pocket interprets every plain exclamation mark as a strong surprised
    # prosody instruction. Neutralize ordinary written emphasis first; explicit
    # performance cues below can still introduce an intentional "Ha!"/"Oh!".
    text = re.sub(r"!+", ".", text)

    for emoji, cue in _EMOJI_CUES.items():
        text = text.replace(emoji, " ")
    text = _ANGLE_TAG_RE.sub("", text)
    text = _SQUARE_TAG_RE.sub("", text)

    def replace_markdown(match: re.Match) -> str:
        action = _markdown_action(match.group(1))
        if action:
            return ""
        words = re.findall(r"[a-zA-Z]+", match.group(1))
        return match.group(1) if len(words) <= 1 else ""

    text = _BOLD_MARKDOWN_RE.sub(lambda match: match.group(1), text)
    text = _ITALIC_OR_ACTION_RE.sub(replace_markdown, text)
    text = _replace_parenthetical_directions(text, lambda _action: "")
    emotion, intensity = _cue(tags)
    return PreparedSpeech(_normalize(text), emotion, intensity)


def prepare_for_orpheus(text: str) -> PreparedSpeech:
    """Retain only Orpheus' documented non-verbal tokens."""
    tags = _find_tags(text)

    def replace(match: re.Match) -> str:
        return _format_cue(match.group(1), ORPHEUS_CUE_MAP, "angle")

    cleaned = _replace_emojis(text, ORPHEUS_CUE_MAP, "angle")
    cleaned = _ANGLE_TAG_RE.sub(replace, cleaned)
    cleaned = _SQUARE_TAG_RE.sub(replace, cleaned)
    cleaned = _replace_markdown(cleaned, ORPHEUS_CUE_MAP, "angle")
    cleaned = _replace_parenthetical_directions(
        cleaned,
        lambda action: _format_cue(action, ORPHEUS_CUE_MAP, "angle"),
    )
    emotion, intensity = _cue(tags)
    return PreparedSpeech(_normalize(cleaned), emotion, intensity)


def prepare_for_chatterbox(text: str) -> PreparedSpeech:
    """Convert the reliably documented Turbo tags to square-bracket form."""
    text = _add_implicit_chatterbox_cue(text)
    tags = _find_tags(text)

    def replace(match: re.Match) -> str:
        return _format_cue(match.group(1), CHATTERBOX_CUE_MAP, "square")

    cleaned = _replace_emojis(text, CHATTERBOX_CUE_MAP, "square")
    cleaned = _ANGLE_TAG_RE.sub(replace, cleaned)
    cleaned = _SQUARE_TAG_RE.sub(replace, cleaned)
    cleaned = _replace_markdown(cleaned, CHATTERBOX_CUE_MAP, "square")
    cleaned = _replace_parenthetical_directions(
        cleaned,
        lambda action: _format_cue(action, CHATTERBOX_CUE_MAP, "square"),
    )
    emotion, intensity = _cue(tags)
    return PreparedSpeech(_normalize(cleaned), emotion, intensity)


def expression_message(prepared: PreparedSpeech) -> dict | None:
    if prepared.emotion == "neutral" or prepared.intensity <= 0:
        return None
    return {
        "v": 1,
        "type": "assistant_expression",
        "emotion": prepared.emotion,
        "intensity": prepared.intensity,
    }
