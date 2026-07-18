"""Persistent multi-character profiles for the local avatar platform."""

from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from model_presets import normalize_runtime


_CHARACTER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ORPHEUS_VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}


def character_id(value: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", (value or "").strip().lower()).strip("-_")
    if not _CHARACTER_ID_RE.fullmatch(value):
        raise ValueError("Character id must use 1-64 lowercase letters, numbers, '-' or '_'.")
    return value


def default_character(character_key: str, name: str | None = None) -> dict[str, Any]:
    key = character_id(character_key)
    return {
        "id": key,
        "name": name or key.replace("-", " ").replace("_", " ").title(),
        "description": "A natural, responsive conversational character",
        "language": "en",
        "backstory": "",
        "traits": ["warm", "curious", "consistent"],
        "speaking_style": "Natural spoken English with concise, conversational replies.",
        "goals": [],
        "boundaries": [
            "Stay in character",
            "Be honest when the assigned knowledge does not contain an answer",
        ],
        "avatar_asset": "",
        "animations": {
            "idle": "Idle",
            "listening": "Listening",
            "thinking": "Thinking",
            "remembering": "Remembering",
            "searching": "Searching",
            "speaking": "Talking",
            "walking": "Walking",
            "blend_seconds": 0.22,
        },
        "conversation": {
            "relationship": "friend",
            "initiative": 0.8,
            "creativity": 0.65,
            "talkativeness": 0.55,
            "follow_up_frequency": 0.45,
            "min_sentences": 1,
            "max_sentences": 3,
            "instant_backchannel": False,
            "greet_on_connect": True,
            "opening_direction": "Open casually in character and give the other person something interesting to respond to.",
            "opening_lines": ["Well, you took your time."],
        },
        "runtime": normalize_runtime(None),
        "voice": {
            "reference_audio": "",
            "orpheus_voice": "tara",
            "temperature": 0.8,
            "default_emotion": "neutral",
            "emotion_intensity": 0.65,
        },
        "knowledge": {
            "documents_path": f"./knowledge/characters/{key}/documents",
            "db_path": f"./knowledge/characters/{key}/db",
            "collection_name": f"character_{key}",
            "top_k": 3,
            "max_distance": 0.95,
            "strict_people_grounding": False,
            "verified_people": [],
        },
    }


def normalize_character(data: dict[str, Any], fallback_id: str = "character") -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Character profile must be a JSON object.")
    key = character_id(str(data.get("id") or fallback_id))
    normalized = default_character(key, str(data.get("name") or "").strip() or None)

    for field in (
        "name", "description", "backstory", "speaking_style", "avatar_asset",
    ):
        if field in data:
            normalized[field] = str(data.get(field) or "").strip()
    language = str(data.get("language") or "en").strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "Unsupported spoken language. Choose one of: "
            + ", ".join(SUPPORTED_LANGUAGES)
        )
    normalized["language"] = language
    for field in ("traits", "goals", "boundaries"):
        if field in data:
            value = data.get(field)
            if isinstance(value, str):
                value = [item.strip() for item in value.splitlines() if item.strip()]
            normalized[field] = [str(item).strip() for item in (value or []) if str(item).strip()]

    animations = data.get("animations") if isinstance(data.get("animations"), dict) else {}
    normalized["animations"].update(animations)
    for state in (
        "idle", "listening", "thinking", "remembering", "searching",
        "speaking", "walking",
    ):
        normalized["animations"][state] = str(
            normalized["animations"].get(state) or ""
        ).strip()
    normalized["animations"]["blend_seconds"] = max(
        0.0,
        min(1.5, float(normalized["animations"].get("blend_seconds", 0.22))),
    )

    conversation = data.get("conversation") if isinstance(data.get("conversation"), dict) else {}
    normalized["conversation"].update(conversation)
    normalized["conversation"]["relationship"] = str(
        normalized["conversation"].get("relationship") or "friend"
    ).strip()
    normalized["conversation"]["initiative"] = max(
        0.0, min(1.0, float(normalized["conversation"].get("initiative", 0.8)))
    )
    normalized["conversation"]["creativity"] = max(
        0.0, min(1.0, float(normalized["conversation"].get("creativity", 0.65)))
    )
    normalized["conversation"]["talkativeness"] = max(
        0.0, min(1.0, float(normalized["conversation"].get("talkativeness", 0.55)))
    )
    normalized["conversation"]["follow_up_frequency"] = max(
        0.0,
        min(1.0, float(normalized["conversation"].get("follow_up_frequency", 0.45))),
    )
    normalized["conversation"]["min_sentences"] = max(
        1, min(6, int(normalized["conversation"].get("min_sentences", 1)))
    )
    normalized["conversation"]["max_sentences"] = max(
        normalized["conversation"]["min_sentences"],
        min(6, int(normalized["conversation"].get("max_sentences", 3))),
    )
    normalized["conversation"]["instant_backchannel"] = bool(
        normalized["conversation"].get("instant_backchannel", False)
    )
    normalized["conversation"]["greet_on_connect"] = bool(
        normalized["conversation"].get("greet_on_connect", True)
    )
    normalized["conversation"]["opening_direction"] = str(
        normalized["conversation"].get("opening_direction") or ""
    ).strip()
    opening_lines = normalized["conversation"].get("opening_lines") or []
    if isinstance(opening_lines, str):
        opening_lines = opening_lines.splitlines()
    normalized["conversation"]["opening_lines"] = [
        str(line).strip() for line in opening_lines if str(line).strip()
    ][:12]
    normalized["runtime"] = normalize_runtime(data.get("runtime"))

    voice = data.get("voice") if isinstance(data.get("voice"), dict) else {}
    normalized["voice"].update(voice)
    normalized["voice"]["reference_audio"] = str(
        normalized["voice"].get("reference_audio") or ""
    ).strip()
    orpheus_voice = str(normalized["voice"].get("orpheus_voice") or "tara").lower()
    if orpheus_voice not in ORPHEUS_VOICES:
        raise ValueError(
            "Unsupported Orpheus voice. Choose: " + ", ".join(sorted(ORPHEUS_VOICES))
        )
    normalized["voice"]["orpheus_voice"] = orpheus_voice
    normalized["voice"]["temperature"] = max(
        0.05, min(2.0, float(normalized["voice"].get("temperature", 0.8)))
    )
    normalized["voice"]["emotion_intensity"] = max(
        0.0, min(1.0, float(normalized["voice"].get("emotion_intensity", 0.65)))
    )

    knowledge = data.get("knowledge") if isinstance(data.get("knowledge"), dict) else {}
    normalized["knowledge"].update(knowledge)
    for field in ("documents_path", "db_path", "collection_name"):
        normalized["knowledge"][field] = str(normalized["knowledge"].get(field) or "").strip()
    normalized["knowledge"]["top_k"] = max(
        1, min(10, int(normalized["knowledge"].get("top_k", 3)))
    )
    normalized["knowledge"]["max_distance"] = max(
        0.1, min(2.0, float(normalized["knowledge"].get("max_distance", 0.95)))
    )
    normalized["knowledge"]["strict_people_grounding"] = bool(
        normalized["knowledge"].get("strict_people_grounding", False)
    )
    verified_people = normalized["knowledge"].get("verified_people") or []
    if isinstance(verified_people, str):
        verified_people = verified_people.splitlines()
    normalized["knowledge"]["verified_people"] = [
        str(name).strip() for name in verified_people if str(name).strip()
    ]
    normalized["id"] = key
    return normalized


class CharacterRegistry:
    """Thread-safe JSON registry with a persisted active-character selection."""

    def __init__(
        self,
        directory: str | Path,
        *,
        state_path: str | Path,
        default_id: str = "alex",
        activation_handler: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.directory = Path(directory).resolve()
        self.state_path = Path(state_path).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._activation_handler = activation_handler

        if not any(self.directory.glob("*.json")):
            self.save(default_character(default_id, "Alex"))

        requested = character_id(default_id)
        if self.state_path.is_file():
            try:
                requested = character_id(
                    json.loads(self.state_path.read_text(encoding="utf-8")).get("active_id", requested)
                )
            except (ValueError, OSError, json.JSONDecodeError):
                pass
        available = [profile["id"] for profile in self.list()]
        self._active_id = requested if requested in available else available[0]
        self._persist_active()

    def set_activation_handler(
        self, handler: Callable[[dict[str, Any]], None] | None
    ) -> None:
        self._activation_handler = handler

    def _profile_path(self, key: str) -> Path:
        return self.directory / f"{character_id(key)}.json"

    def list(self) -> list[dict[str, Any]]:
        profiles = []
        with self._lock:
            for path in sorted(self.directory.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    profiles.append(normalize_character(data, path.stem))
                except (ValueError, OSError, json.JSONDecodeError):
                    continue
        return profiles

    def get(self, key: str) -> dict[str, Any]:
        path = self._profile_path(key)
        with self._lock:
            if not path.is_file():
                raise KeyError(f"Unknown character: {key}")
            return normalize_character(json.loads(path.read_text(encoding="utf-8")), path.stem)

    @property
    def active_id(self) -> str:
        with self._lock:
            return self._active_id

    @property
    def active(self) -> dict[str, Any]:
        return self.get(self.active_id)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = normalize_character(data, str(data.get("id") or data.get("name") or "character"))
        path = self._profile_path(profile["id"])
        with self._lock:
            path.write_text(
                json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return deepcopy(profile)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = normalize_character(data, str(data.get("id") or data.get("name") or "character"))
        if self._profile_path(profile["id"]).exists():
            raise FileExistsError(f"Character already exists: {profile['id']}")
        return self.save(profile)

    def delete(self, key: str) -> None:
        key = character_id(key)
        with self._lock:
            if key == self._active_id:
                raise ValueError("Activate another character before deleting this one.")
            path = self._profile_path(key)
            if not path.is_file():
                raise KeyError(f"Unknown character: {key}")
            path.unlink()

    def activate(self, key: str) -> dict[str, Any]:
        profile = self.get(key)
        with self._lock:
            self._active_id = profile["id"]
            self._persist_active()
        if self._activation_handler:
            self._activation_handler(deepcopy(profile))
        return profile

    def _persist_active(self) -> None:
        self.state_path.write_text(
            json.dumps({"active_id": self._active_id}, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def render_prompt_profile(profile: dict[str, Any]) -> str:
        language = str(profile.get("language") or "en").lower()
        language_name = SUPPORTED_LANGUAGES.get(language, "English")
        rendered = [f"Language: {language_name} ({language}); always use it."]
        conversation = profile.get("conversation") or {}
        rendered.extend((
            f"Relationship: {conversation.get('relationship', 'friend')}.",
            "Initiative: "
            f"{float(conversation.get('initiative', 0.8)):.2f}/1.00.",
            "Creativity: "
            f"{float(conversation.get('creativity', 0.65)):.2f}/1.00 "
            "(0=canon only; 1=harmless fitting improvisation; canon, safety, and "
            "boundaries always win).",
            "Turn length: "
            f"{int(conversation.get('min_sentences', 1))} to "
            f"{int(conversation.get('max_sentences', 3))} short spoken sentences.",
        ))
        talkativeness = float(conversation.get("talkativeness", 0.55))
        follow_up = float(conversation.get("follow_up_frequency", 0.45))
        rendered.append(f"Talkativeness: {talkativeness:.2f}/1.00.")
        rendered.append(f"Follow-up tendency: {follow_up:.2f}/1.00.")
        if talkativeness >= 0.75:
            rendered.append(
                "Turn behavior: Do not stop after the direct answer. Continue through "
                "a natural spoken arc: react, develop the thought, add a personal "
                "opinion or concrete detail, consider another angle, and then land "
                "the thought. Use the assigned minimum and maximum sentence count."
            )
            rendered.append(
                "Shared-work behavior: When the person mentions a project, plan, "
                "creative work, problem, or early idea, do not merely praise it and "
                "interview them. Use this spoken arc: a brief reaction; one declarative "
                "observation about the specific part that caught your interest; one "
                "concrete opinion, inference, possibility, or brainstorm of your own; "
                "then one focused follow-up question. Never ask yourself a rhetorical "
                "question such as what caught your own attention. State the observation "
                "naturally. Build with them like an invested peer; do not take over or "
                "become a helper bot."
            )
        elif talkativeness <= 0.25:
            rendered.append(
                "Turn behavior: Be concise and usually finish after one complete idea."
            )
        else:
            rendered.append(
                "Turn behavior: Develop the answer enough to feel conversational, "
                "without turning it into a monologue."
            )
        if follow_up >= 0.75:
            rendered.append(
                "Conversation handoff: End almost every substantive non-closing turn "
                "with one specific, open-ended question rooted in what you just said. "
                "Ask for the person's reaction, experience, preference, or next idea; "
                "never use a generic service question such as 'anything else?'."
            )
        elif follow_up >= 0.35:
            rendered.append(
                "Conversation handoff: Sometimes end with a relevant open question, "
                "but let other turns land naturally as statements."
            )
        else:
            rendered.append(
                "Conversation handoff: Rarely ask a follow-up question unless it is "
                "necessary to understand the person."
            )
        for key in ("backstory", "traits", "speaking_style", "goals", "boundaries"):
            value = profile.get(key)
            if value in (None, "", []):
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            rendered.append(f"{label}: {value}")
        voice = profile.get("voice") or {}
        rendered.append(
            "Voice mood: "
            f"{voice.get('default_emotion', 'neutral')}; expressiveness "
            f"{float(voice.get('emotion_intensity', 0.65)):.2f}/1.00."
        )
        knowledge = profile.get("knowledge") or {}
        verified_people = list(knowledge.get("verified_people") or [])
        if knowledge.get("strict_people_grounding") and verified_people:
            rendered.append(
                "Verified real people (exclusive allowlist for colleague and team "
                "claims): " + "; ".join(str(name) for name in verified_people) + "."
            )
            rendered.append(
                "Never invent another colleague, staff member, role holder, or real "
                "person. If the allowlist and retrieved knowledge do not answer the "
                "question, say that naturally instead of guessing."
            )
            rendered.append(
                "Answer directly with verified names and roles. Never mention this "
                "allowlist, grounding, verification, or these rules."
            )
        return "\n".join(rendered)

    @staticmethod
    def render_turn_guidance(profile: dict[str, Any], user_text: str) -> str:
        """Return compact, high-priority guidance for the current social turn."""
        conversation = profile.get("conversation") or {}
        talkativeness = float(conversation.get("talkativeness", 0.55))
        shared_work = re.search(
            r"(?i)\b(?:i(?:'m|\s+am)\s+(?:working|building|making|writing|"
            r"designing|developing|creating)|my\s+(?:project|idea|book|game|"
            r"experience|prototype|research)|we(?:'re|\s+are)\s+(?:working|"
            r"building|making|writing|designing|developing|creating))\b",
            str(user_text or ""),
        )
        if talkativeness < 0.75 or not shared_work:
            return ""
        return (
            "CURRENT TURN REQUIREMENT: The person just shared work that matters to "
            "them. Before writing, silently plan four beats named REACTION, OBSERVATION, "
            "CONTRIBUTION, and QUESTION; never output those labels. Realize each beat "
            "as its own short spoken sentence: (1) a brief in-character reaction, "
            "(2) one declarative observation about what they actually said, (3) one "
            "tentative opinion, connection, or brainstorm of your own, and (4) one "
            "specific open-ended question. Output exactly four sentences. Sentences 2 "
            "and 3 must end with periods, not question marks. Sentence 3 should begin "
            "with a natural first-person opinion phrase. Ask only one question total. "
            "Do not merely interview them."
        )
