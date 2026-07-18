"""Conversational AI Avatar Server — Pipecat pipeline + WebSocket entry point.

Assembles a real-time voice AI pipeline:
  Audio In → Whisper STT → [RAG + Emotion] → Ollama LLM → Orpheus TTS → Audio Out

All communication with the Unity client (Quest 3 or Desktop) happens
over a single WebSocket connection carrying raw PCM audio frames.
"""

# Must run before torch/safetensors or any service importing them. Normal
# python.org installations take the no-op path; this only helps local embedded
# Python recovery environments.
from runtime_bootstrap import prepare_windows_stable_abi

prepare_windows_stable_abi()

import asyncio
from collections import deque
import io
import json
import logging
import re
import sys
import threading
import time
import wave
from pathlib import Path

from config import config

# ── VAD ──
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor

# ── Pipecat core ──
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams

# ── Transport ──
from pipecat.transports.websocket.server import (
    WebsocketServerTransport,
    WebsocketServerParams,
)

# ── Services ──
import asyncio as _stt_asyncio
import numpy as _stt_np
from pipecat.frames.frames import ErrorFrame as _ErrorFrame, TranscriptionFrame as _TranscriptionFrame
from pipecat.services.whisper.stt import WhisperSTTService as _WhisperSTTServiceBase
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.utils.time import time_now_iso8601 as _time_now_iso8601


class WhisperSTTService(_WhisperSTTServiceBase):
    """Pipecat's WhisperSTTService with verbose diagnostic logging.

    Stock Pipecat silently drops a transcription when every segment fails
    the ``no_speech_prob`` filter, which makes it impossible to tell
    whether the mic is silent, the model rejected speech, or something
    else went wrong. This subclass logs every transcription attempt so
    we always know exactly what Whisper saw.
"""


    async def run_stt(self, audio: bytes):
        if not self._model:
            yield _ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        # SegmentedSTTService supplies a WAV container. Decode it instead of
        # feeding the RIFF header to Whisper as a burst of fake speech.
        if audio[:4] == b"RIFF":
            with wave.open(io.BytesIO(audio), "rb") as source:
                audio = source.readframes(source.getnframes())
        audio_float = _stt_np.frombuffer(audio, dtype=_stt_np.int16).astype(_stt_np.float32) / 32768.0
        peak = float(_stt_np.max(_stt_np.abs(audio_float))) if audio_float.size else 0.0
        rms = float(_stt_np.sqrt(_stt_np.mean(audio_float ** 2))) if audio_float.size else 0.0

        whisper_lang = self.language_to_service_language(self._settings["language"])
        segments_iter, _info = await _stt_asyncio.to_thread(
            self._model.transcribe,
            audio_float,
            language=whisper_lang,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )

        text = ""
        seg_count = 0
        rejected = []
        for segment in segments_iter:
            seg_count += 1
            if segment.no_speech_prob < self._no_speech_prob:
                text += f"{segment.text} "
            else:
                rejected.append((segment.text, segment.no_speech_prob))

        await self.stop_processing_metrics()

        text = text.strip()
        logger.info(
            "[STT] %.2fs audio (peak=%.3f rms=%.3f) -> %d segments, kept=%r, rejected=%r",
            audio_float.size / 16000.0, peak, rms, seg_count, text or "<empty>", rejected,
        )

        if text:
            await self._handle_transcription(text, True, self._settings["language"])
            yield _TranscriptionFrame(
                text,
                self._user_id,
                _time_now_iso8601(),
                self._settings["language"],
            )

    async def warm_for_startup(self):
        """Initialize CUDA kernels before Unity's first real utterance."""
        if not self._model:
            return
        segments, _ = await _stt_asyncio.to_thread(
            self._model.transcribe,
            _stt_np.zeros(4000, dtype=_stt_np.float32),
            language=self.language_to_service_language(self._settings["language"]),
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        await _stt_asyncio.to_thread(list, segments)
        logger.info("Faster-Whisper GPU path is warm.")

# ── Context management ──
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
    LLMAssistantAggregator,
)
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# ── Frame processing ──
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# ── Local services ──
from pipeline.raw_audio_serializer import RawAudioSerializer
from pipeline.native_ollama_service import NativeOllamaLLMService
from pipeline.client_events import ClientEventProcessor
from pipeline.tts_service import OrpheusTTSService
from pipeline.rag_service import RAGService
from pipeline.knowledge_router import should_retrieve_character_knowledge
from pipeline.emotion_processor import EmotionProcessor
from pipeline.speech_text import (
    CHATTERBOX_BACKCHANNELS,
    guard_verified_people,
    is_continuation_request,
    should_use_instant_backchannel,
)
from control.character_registry import CharacterRegistry
from control.character_runtime import CharacterRuntime
from control.api import create_control_app
from model_presets import resolve_runtime
from pipeline.llm_service import (
    build_system_prompt,
    build_base_system_prompt,
    load_character_profile,
    performance_cue_guide,
    performance_fast_starter_rule,
    performance_turn_rule,
)

_runtime_log_dir = Path(__file__).resolve().parent / "runtime"
_runtime_log_dir.mkdir(parents=True, exist_ok=True)
_audit_path = _runtime_log_dir / "conversation.jsonl"
_audit_lock = threading.Lock()

POCKET_BACKCHANNELS = ("Mm.", "Oh.", "Right.", "Huh.")


def _audit(event: str, **data) -> None:
    record = {"time": time.time(), "event": event, **data}
    with _audit_lock:
        with _audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            _runtime_log_dir / "server_current.log", mode="w", encoding="utf-8"
        ),
    ],
    force=True,
)
logger = logging.getLogger("main")


async def _prewarm_ollama(model: str) -> None:
    """Load the local LLM before accepting the first Unity conversation."""
    import httpx

    started = asyncio.get_running_loop().time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                config.LLM_BASE_URL.rstrip("/") + "/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": config.LLM_KEEP_ALIVE,
                    "options": {"num_predict": 1, "num_ctx": config.LLM_CONTEXT_SIZE},
                },
            )
            response.raise_for_status()
        logger.info(
            "Ollama model pre-warmed in %.2fs.",
            asyncio.get_running_loop().time() - started,
        )
    except Exception:
        logger.exception("Ollama pre-warm failed; the first response may be slower.")


class RAGContextProcessor(FrameProcessor):
    """Enriches the LLM system prompt with RAG context before each turn.

    Sits between the user context aggregator and the LLM in the pipeline.
    When a frame passes through that indicates a new user turn, this
    processor:
      1. Reads the latest user message from the shared context
      2. Queries the RAG knowledge base
      3. Optionally runs emotion analysis
      4. Updates the system message in the shared context

    Because the LLMContext is shared by reference, the LLM will see
    the updated system prompt when it processes the same context frame.
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        character_runtime: CharacterRuntime,
        emotion_processor: EmotionProcessor,
        tts_backend: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._characters = character_runtime
        self._emotion = emotion_processor
        self._tts_backend = tts_backend
        self._turn_count = 0
        self.latest_user_text = ""

    def _get_latest_user_text(self) -> str | None:
        """Walk backwards through context to find the last user message."""
        for msg in reversed(self._context.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def _update_system_message(self, user_text: str):
        """Rebuild the system prompt with fresh RAG + emotion context."""
        self._turn_count += 1
        profile = self._characters.active
        recent_user_turns = [
            str(msg.get("content") or "").strip()
            for msg in self._context.messages
            if msg.get("role") == "user" and str(msg.get("content") or "").strip()
        ][-3:]
        rag_parts = list(recent_user_turns)
        if re.search(
            r"(?i)\b(they|them|their|those|team|staff|colleagues?|names?|roles?)\b",
            user_text,
        ):
            # Resolve conversational follow-ups such as "what are their names?"
            # against the character's established workplace/backstory instead
            # of treating the pronoun as a brand-new, context-free query.
            rag_parts.append(str(profile.get("backstory") or ""))
        needs_knowledge = should_retrieve_character_knowledge(
            user_text,
            recent_user_turns[:-1],
        )
        new_prompt = build_system_prompt(
            character_name=profile["name"],
            character_description=profile["description"],
            user_text=user_text,
            rag_query="\n".join(rag_parts),
            # Embedding every casual utterance added hundreds of milliseconds
            # and irrelevant prompt tokens. Retrieve only on a knowledge turn.
            rag_service=(
                self._characters.active_rag() if needs_knowledge else None
            ),
            emotion_processor=self._emotion,
            turn_count=self._turn_count,
            character_profile=self._characters.registry.render_prompt_profile(profile),
            emotion_cue_guide=performance_cue_guide(
                self._tts_backend, profile.get("language", "en")
            ),
            fast_starter_rule=performance_fast_starter_rule(
                self._tts_backend, profile.get("language", "en")
            ),
        )
        turn_guidance = self._characters.registry.render_turn_guidance(
            profile,
            user_text,
        )
        if turn_guidance:
            new_prompt += "\n\n" + turn_guidance
        new_prompt += "\n\n" + performance_turn_rule(
            self._tts_backend,
            profile.get("language", "en"),
        )
        if self._context.messages and self._context.messages[0].get("role") == "system":
            self._context.messages[0]["content"] = new_prompt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            user_text = self._get_latest_user_text()
            if user_text:
                self.latest_user_text = user_text
                self._update_system_message(user_text)
                logger.info("User: %s", user_text)
                profile = self._characters.active
                _audit("user", character_id=profile["id"], text=user_text)

        await self.push_frame(frame, direction)


class VoiceResponseLimiter(FrameProcessor):
    """Forward only complete, concise spoken sentences to TTS and memory."""

    def __init__(
        self,
        max_sentences: int = 2,
        profile_provider=None,
        latest_user_provider=None,
        tts_backend: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._max_sentences = max(1, max_sentences)
        self._active_max_sentences = self._max_sentences
        self._sentence_count = 0
        self._done = False
        self._profile_provider = profile_provider
        self._latest_user_provider = latest_user_provider
        self._tts_backend = tts_backend
        self._backchannel_index = 0
        self._spoken_parts: list[str] = []
        self._started_at = 0.0
        self._pending_prefix = ""
        self._interrupted = False
        self._emitted_reaction = False
        self._recent_sentences: deque[str] = deque(maxlen=16)
        self._active_profile: dict = {}
        self._active_latest_user = ""
        self._grounding_blocked = False

    @staticmethod
    def _sentence_boundaries(text: str, *, allow_terminal: bool) -> list[int]:
        lookahead = r"(?=\s|$)" if allow_terminal else r"(?=\s)"
        pattern = re.compile(
            rf"(?:\.{{2,}}|[!?]+|(?<![\d.])\.(?![\d.])){lookahead}"
        )
        return [match.end() for match in pattern.finditer(text)]

    def _remove_recent_repeats(self, text: str) -> str:
        pieces = re.findall(r".*?(?:\.{2,}|[.!?]+)(?:\s+|$)", text, re.DOTALL)
        if not pieces:
            pieces = [text]
        unique = []
        for piece in pieces:
            normalized = " ".join(
                re.findall(r"[\w']+", piece.casefold(), re.UNICODE)
            )
            if len(normalized.split()) >= 4 and normalized in self._recent_sentences:
                logger.info("Suppressing repeated spoken sentence: %r", piece.strip())
                continue
            if len(normalized.split()) >= 4:
                self._recent_sentences.append(normalized)
            unique.append(piece)
        return "".join(unique).strip()

    def _guard_verified_people(self, text: str) -> str:
        """Block invented colleague names without buffering the LLM response."""
        profile = self._active_profile or {}
        knowledge = profile.get("knowledge") or {}
        verified = [str(name) for name in knowledge.get("verified_people") or []]
        if not knowledge.get("strict_people_grounding") or not verified:
            return text

        guarded, unknown = guard_verified_people(
            text,
            latest_user=self._active_latest_user,
            verified_people=verified,
            profile_name=str(profile.get("name") or ""),
        )
        if not unknown:
            return text
        logger.warning(
            "Blocked unverified person name(s) %s in spoken sentence: %r",
            unknown,
            text,
        )
        self._grounding_blocked = True
        return guarded

    def _consume_complete(self, *, allow_terminal: bool = False) -> str:
        if not self._emitted_reaction and self._sentence_count == 0:
            reaction = re.match(
                r"^\s*(\[(?:laugh|chuckle|sigh|gasp|groan|cough|sniff|"
                r"shush|clear throat)\]|<(?:laugh|chuckle|sigh|gasp|groan|"
                r"yawn|cough|sniffle)>)",
                self._pending_prefix,
                re.IGNORECASE,
            )
            if reaction:
                self._pending_prefix = self._pending_prefix[reaction.end():]
                self._emitted_reaction = True
                # The period makes the TTS sentence aggregator release the cue
                # immediately; the Pocket sanitizer turns it into Ha!, Phew!, etc.
                return reaction.group(1) + "."

        # Small local models occasionally ignore the short-first-sentence
        # contract and begin a long sentence with a useful clause followed by a
        # comma ("I'm good, just ...").  Pocket cannot start until it receives
        # terminal punctuation, so release that already-complete opening thought
        # as a sentence while the LLM continues.  Restrict this to the first
        # clause and 2-8 words to avoid chopping ordinary prose into fragments.
        if self._sentence_count == 0 and not self._sentence_boundaries(
            self._pending_prefix, allow_terminal=allow_terminal
        ):
            comma = self._pending_prefix.find(",")
            if comma >= 0:
                clause = self._pending_prefix[:comma].strip()
                word_count = len(re.findall(r"[\w']+", clause, re.UNICODE))
                if 2 <= word_count <= 8:
                    self._pending_prefix = self._pending_prefix[comma + 1:].lstrip()
                    if self._pending_prefix:
                        self._pending_prefix = (
                            self._pending_prefix[0].upper() + self._pending_prefix[1:]
                        )
                    self._sentence_count += 1
                    return self._guard_verified_people(
                        self._remove_recent_repeats(clause + ".")
                    )

        boundaries = self._sentence_boundaries(
            self._pending_prefix, allow_terminal=allow_terminal
        )
        if not boundaries:
            return ""
        remaining = self._active_max_sentences - self._sentence_count
        count = min(len(boundaries), max(0, remaining))
        if count <= 0:
            self._done = True
            self._pending_prefix = ""
            return ""
        emit_end = boundaries[count - 1]
        candidate = self._pending_prefix[:emit_end].lstrip()
        if candidate and self._sentence_count > 0:
            candidate = re.sub(
                r"^([\"'(\[]*)([a-z])",
                lambda match: match.group(1) + match.group(2).upper(),
                candidate,
                count=1,
            )
        self._pending_prefix = self._pending_prefix[emit_end:]
        self._sentence_count += count
        if self._sentence_count >= self._active_max_sentences:
            self._done = True
            self._pending_prefix = ""
        return self._guard_verified_people(self._remove_recent_repeats(candidate))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            profile = self._profile_provider() if self._profile_provider else {}
            conversation = profile.get("conversation") or {}
            latest_user = str(
                self._latest_user_provider() if self._latest_user_provider else ""
            ).strip()
            self._active_profile = dict(profile)
            self._active_latest_user = latest_user
            self._active_max_sentences = max(
                1, min(6, int(conversation.get("max_sentences", self._max_sentences)))
            )
            talkativeness = float(conversation.get("talkativeness", 0.55))
            wants_long_answer = bool(re.search(
                r"(?i)\b(?:story|in\s+detail|explain|elaborate|team|staff|"
                r"colleagues?|coworkers?|names?|list|who\s+works)\b",
                latest_user,
            ))
            if not wants_long_answer:
                if talkativeness < 0.25:
                    self._active_max_sentences = 1
                elif talkativeness < 0.5:
                    self._active_max_sentences = min(self._active_max_sentences, 2)
                elif talkativeness < 0.75:
                    self._active_max_sentences = min(self._active_max_sentences, 3)
            # A barge-in correction replaces the cancelled turn.  Enforce a
            # one-sentence spoken repair even if a small LLM tries to resume the
            # old topic after answering the correction correctly.
            if re.match(
                r"(?i)^\s*(?:actually|no\b|wait\b|sorry\b|i\s+mean\b|correction\b)",
                latest_user,
            ):
                self._active_max_sentences = 1
            self._sentence_count = 0
            self._done = False
            self._spoken_parts = []
            self._pending_prefix = ""
            self._interrupted = False
            self._emitted_reaction = False
            self._grounding_blocked = False
            self._started_at = time.perf_counter()
            await self.push_frame(OutputTransportMessageUrgentFrame(message={
                "v": 1, "type": "assistant_response_started"
            }), direction)
            await self.push_frame(frame, direction)
            if (
                latest_user
                and self._tts_backend in {"chatterbox", "pocket"}
                and str(profile.get("language") or "en").lower() == "en"
                and conversation.get("instant_backchannel", False)
            ):
                if not should_use_instant_backchannel(latest_user):
                    return
                choices = (
                    CHATTERBOX_BACKCHANNELS
                    if self._tts_backend == "chatterbox"
                    else POCKET_BACKCHANNELS
                )
                backchannel = choices[self._backchannel_index % len(choices)]
                self._backchannel_index += 1
                self._spoken_parts.append(backchannel + " ")
                await self.push_frame(LLMTextFrame(backchannel), direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._interrupted = True
            self._pending_prefix = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if self._done:
                return
            self._pending_prefix += frame.text
            # Wait for one character of look-ahead before accepting punctuation.
            # This treats "..." as one boundary instead of three sentences and
            # avoids the clipped responses seen in Unity.
            completed = self._consume_complete()
            if completed:
                self._spoken_parts.append(completed)
                await self.push_frame(LLMTextFrame(completed + " "), direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            profile = self._profile_provider() if self._profile_provider else {}
            completed = self._consume_complete(allow_terminal=True)
            if completed:
                self._spoken_parts.append(completed)
                await self.push_frame(LLMTextFrame(completed + " "), direction)
            spoken = " ".join(part.strip() for part in self._spoken_parts).strip()
            self._pending_prefix = ""
            if not self._interrupted and not re.search(r"[\w\d]", spoken, re.UNICODE):
                # A rare model failure should sound like a human self-repair,
                # not the same canned thinking prompt on every failed turn.
                repairs = (
                    "Wait—I lost the end of that thought.",
                    "Nope, my brain just dropped that sentence.",
                    "Hold on, that came out half-finished.",
                )
                fallback = repairs[self._backchannel_index % len(repairs)]
                self._backchannel_index += 1
                knowledge = (self._active_profile or {}).get("knowledge") or {}
                verified = list(knowledge.get("verified_people") or [])
                people_turn = bool(re.search(
                    r"(?i)\b(?:team|staff|colleagues?|coworkers?|who\s+else|"
                    r"anyone\s+else|who\s+works|names?)\b",
                    self._active_latest_user,
                ))
                if (
                    knowledge.get("strict_people_grounding")
                    and people_turn
                    and verified
                ):
                    selected = " and ".join(str(name) for name in verified[:2])
                    fallback = f"{selected} are two of my colleagues."
                elif self._grounding_blocked:
                    fallback = "I don't have another verified colleague to name."
                self._spoken_parts = [fallback]
                spoken = fallback
                await self.push_frame(LLMTextFrame(fallback), direction)
            _audit(
                "assistant",
                character_id=profile.get("id", ""),
                text=spoken,
                generation_seconds=round(time.perf_counter() - self._started_at, 4),
            )
            await self.push_frame(frame, direction)
            await self.push_frame(OutputTransportMessageUrgentFrame(message={
                "v": 1, "type": "assistant_response_finished"
            }), direction)
            return

        await self.push_frame(frame, direction)


class SlidingContextWindow(FrameProcessor):
    """Keep recent turn pairs so prompt prefill latency does not grow forever."""

    def __init__(self, *, context: LLMContext, max_turns: int = 8, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._max_turns = max(1, max_turns)

    def _trim(self):
        messages = self._context.messages
        head_end = 0
        while head_end < len(messages) and messages[head_end].get("role") == "system":
            head_end += 1
        tail = list(messages[head_end:])
        if len(tail) <= self._max_turns * 2:
            return

        tail = tail[-self._max_turns * 2 :]
        while tail and tail[0].get("role") != "user":
            tail.pop(0)
        messages[:] = list(messages[:head_end]) + tail

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self._trim()
        await self.push_frame(frame, direction)


async def run_pipeline():
    """Initialize all services and run the Pipecat pipeline."""

    # ── Character platform + character-scoped RAG ──
    server_root = Path(__file__).resolve().parent
    characters_path = Path(config.CHARACTERS_DIR)
    if not characters_path.is_absolute():
        characters_path = server_root / characters_path
    registry = CharacterRegistry(
        characters_path,
        state_path=server_root / "runtime" / "active_character.json",
        default_id=config.ACTIVE_CHARACTER_ID,
    )
    character_runtime = CharacterRuntime(
        registry,
        server_root=server_root,
        embedding_model=config.RAG_EMBEDDING_MODEL,
        embedding_device=config.RAG_DEVICE,
        default_top_k=config.RAG_TOP_K,
    )
    active_character = character_runtime.active
    running_runtime = resolve_runtime(active_character)
    llm_model = running_runtime["llm_model"]
    tts_backend = running_runtime["tts_backend"]
    active_creativity = float(
        (active_character.get("conversation") or {}).get("creativity", 0.65)
    )
    llm_temperature = round(
        0.2 + 0.4 * max(0.0, min(1.0, active_creativity)), 2
    )
    _audit(
        "session_start",
        character_id=active_character["id"],
        brain=running_runtime["brain"],
        voice=running_runtime["voice"],
    )
    await _prewarm_ollama(llm_model)
    rag_service = character_runtime.active_rag()
    if rag_service.document_count > 0:
        logger.info("Warming local knowledge retrieval before opening Unity ports...")
        await asyncio.to_thread(
            rag_service.get_relevant_context,
            f"What should I know about {active_character['name']}?",
        )

    # ── Emotion Processor (optional) ──
    emotion_processor = EmotionProcessor(
        enabled=config.EMOTION_ENABLED,
        model_name=config.EMOTION_MODEL,
    )

    # ── STT (Pipecat built-in Faster Whisper) ──
    stt_backend = config.STT_BACKEND
    if stt_backend == "moonshine":
        from pipeline.moonshine_stt_service import MoonshineSTTService

        moonshine_python = Path(config.MOONSHINE_PYTHON)
        if not moonshine_python.is_absolute():
            moonshine_python = server_root / moonshine_python
        logger.info(
            "Initializing streaming STT: Moonshine %s", config.MOONSHINE_ARCH
        )
        stt = MoonshineSTTService(
            base_url=config.MOONSHINE_URL,
            worker_python=str(moonshine_python),
            worker_script=str(server_root / "moonshine_worker.py"),
            language=config.STT_LANGUAGE,
            architecture=config.MOONSHINE_ARCH,
            cache_dir=str(server_root / "runtime" / "moonshine_cache"),
            autostart=config.MOONSHINE_AUTOSTART,
            pre_roll_seconds=config.MOONSHINE_PRE_ROLL_SECONDS,
        )
        try:
            await stt.warm_for_startup()
        except Exception:
            logger.exception(
                "Moonshine startup failed; using Faster Whisper for this run."
            )
            stt_backend = "whisper"
    if stt_backend != "moonshine":
        logger.info(
            "Initializing STT: model=%s device=%s no_speech_prob=%.2f",
            config.STT_MODEL, config.STT_DEVICE, config.STT_NO_SPEECH_PROB,
        )
        stt = WhisperSTTService(
            model=config.STT_MODEL,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
            language=config.STT_LANGUAGE,
            no_speech_prob=config.STT_NO_SPEECH_PROB,
        )
        await stt.warm_for_startup()

    # ── LLM (Ollama, OpenAI-compatible API) ──
    logger.info("Initializing LLM: model=%s", llm_model)
    llm = NativeOllamaLLMService(
        model=llm_model,
        native_base_url=config.LLM_BASE_URL,
        params=BaseOpenAILLMService.InputParams(
            temperature=llm_temperature,
            top_p=config.LLM_TOP_P,
            max_tokens=config.LLM_MAX_TOKENS,
            extra={
                "extra_body": {
                    "options": {
                        "num_ctx": config.LLM_CONTEXT_SIZE,
                        "top_k": config.LLM_TOP_K,
                        "repeat_penalty": config.LLM_REPEAT_PENALTY,
                    },
                    "keep_alive": config.LLM_KEEP_ALIVE,
                }
            },
        ),
    )

    # ── TTS (Orpheus+vLLM for cloud, Kokoro for local) ──
    if tts_backend == "pocket":
        from pipeline.pocket_tts_service import PocketTTSService

        pocket_python = Path(config.POCKET_TTS_PYTHON)
        if not pocket_python.is_absolute():
            pocket_python = server_root / pocket_python
        logger.info(
            "Initializing TTS (pocket): worker=%s voice=%s",
            config.POCKET_TTS_URL,
            config.POCKET_TTS_VOICE,
        )
        tts = PocketTTSService(
            base_url=config.POCKET_TTS_URL,
            worker_python=str(pocket_python),
            worker_script=str(server_root / "pocket_worker.py"),
            default_voice=config.POCKET_TTS_VOICE,
            language=config.POCKET_TTS_LANGUAGE,
            autostart=config.POCKET_TTS_AUTOSTART,
            profile_provider=character_runtime.voice_settings,
            audit_callback=lambda payload: _audit("tts", **payload),
        )
    elif tts_backend == "kokoro":
        from pipeline.kokoro_tts_service import KokoroTTSService
        kokoro_voice = config.TTS_VOICE if config.TTS_VOICE != "tara" else "af_heart"
        kokoro_device = config.KOKORO_DEVICE or None
        logger.info(
            "Initializing TTS (kokoro): voice=%s lang=%s device=%s",
            kokoro_voice, config.KOKORO_LANG_CODE, kokoro_device or "auto",
        )
        tts = KokoroTTSService(
            voice=kokoro_voice,
            lang_code=config.KOKORO_LANG_CODE,
            speed=config.KOKORO_SPEED,
            device=kokoro_device,
        )
    elif tts_backend == "chatterbox":
        from pipeline.chatterbox_tts_service import ChatterboxTTSService

        chatterbox_device = config.CHATTERBOX_DEVICE or None
        logger.info(
            "Initializing TTS (chatterbox): device=%s reference=%s",
            chatterbox_device or "auto",
            config.CHATTERBOX_REFERENCE_AUDIO or "built-in",
        )
        tts = ChatterboxTTSService(
            device=chatterbox_device,
            reference_audio=config.CHATTERBOX_REFERENCE_AUDIO,
            temperature=config.CHATTERBOX_TEMPERATURE,
            warmup=config.CHATTERBOX_WARMUP,
            profile_provider=character_runtime.voice_settings,
            audit_callback=lambda payload: _audit("tts", **payload),
        )
    elif tts_backend == "orpheus-cpp":
        from pipeline.orpheus_cpp_tts_service import OrpheusCppTTSService

        logger.info(
            "Initializing TTS (orpheus-cpp): voice=%s prebuffer=%.2fs",
            config.TTS_VOICE,
            config.ORPHEUS_CPP_PREBUFFER_SECONDS,
        )
        tts = OrpheusCppTTSService(
            voice=config.TTS_VOICE,
            language=active_character.get("language", "en"),
            prebuffer_seconds=config.ORPHEUS_CPP_PREBUFFER_SECONDS,
            gpu_layers=config.ORPHEUS_CPP_GPU_LAYERS,
            verbose=config.ORPHEUS_CPP_VERBOSE,
            profile_provider=character_runtime.voice_settings,
        )
    elif tts_backend == "orpheus":
        logger.info(
            "Initializing TTS (orpheus): model=%s voice=%s vllm=%s",
            config.TTS_MODEL, config.TTS_VOICE, config.VLLM_BASE_URL,
        )
        tts = OrpheusTTSService(
            vllm_base_url=config.VLLM_BASE_URL,
            model_name=config.TTS_MODEL,
            voice=config.TTS_VOICE,
        )
    else:
        raise ValueError(
            f"Unsupported TTS_BACKEND={tts_backend!r}; use pocket, kokoro, "
            "chatterbox, orpheus-cpp, or orpheus"
        )

    warm_for_startup = getattr(tts, "warm_for_startup", None)
    if warm_for_startup:
        logger.info("Warming the active speech model before opening Unity ports...")
        await warm_for_startup()
        # Chatterbox and Ollama share a constrained 8 GB laptop GPU. Loading
        # Chatterbox can evict Ollama even when keep_alive is set, making the
        # first live turn pay a multi-second model reload. Make the brain the
        # final warm operation before clients are accepted.
        logger.info("Re-warming Ollama after speech-model allocation...")
        await _prewarm_ollama(llm_model)

    # ── Conversation context ──
    character_profile = registry.render_prompt_profile(active_character)
    base_prompt = build_base_system_prompt(
        active_character["name"],
        active_character["description"],
        character_profile,
        emotion_cue_guide=performance_cue_guide(
            tts_backend, active_character.get("language", "en")
        ),
        fast_starter_rule=performance_fast_starter_rule(
            tts_backend, active_character.get("language", "en")
        ),
    )
    messages = [{"role": "system", "content": base_prompt}]
    context = LLMContext(messages)
    # Moonshine finalizes the transcript before forwarding the VAD-stop frame,
    # so a second semantic "smart turn" pass only adds about one second of dead
    # air.  A tiny resume window still lets a natural micro-pause continue, but
    # hands a finalized utterance to the LLM essentially immediately.
    user_aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)]
            )
        ),
    )
    assistant_aggregator = LLMAssistantAggregator(context)

    @user_aggregator.event_handler("on_user_turn_started")
    async def cancel_stale_speech_on_user_start(_aggregator, _strategy):
        # This callback runs at VAD-start, before a busy TTS processor gets its
        # queued InterruptionFrame. Pocket can therefore release its stateful
        # generation lock while the user's correction is still being spoken.
        cancel_speech = getattr(tts, "interrupt_active_speech", None)
        if cancel_speech:
            await cancel_speech()

    # ── RAG context processor ──
    rag_processor = RAGContextProcessor(
        context=context,
        character_runtime=character_runtime,
        emotion_processor=emotion_processor,
        tts_backend=tts_backend,
    )

    def activate_character(profile):
        new_prompt = build_base_system_prompt(
            profile["name"],
            profile["description"],
            registry.render_prompt_profile(profile),
            emotion_cue_guide=performance_cue_guide(
                tts_backend, profile.get("language", "en")
            ),
            fast_starter_rule=performance_fast_starter_rule(
                tts_backend, profile.get("language", "en")
            ),
        )
        context.messages[:] = [{"role": "system", "content": new_prompt}]
        rag_processor._turn_count = 0
        async def prepare_character_assets():
            creativity = float(
                (profile.get("conversation") or {}).get("creativity", 0.65)
            )
            await llm._update_settings({
                "temperature": round(
                    0.2 + 0.4 * max(0.0, min(1.0, creativity)), 2
                )
            })
            rag = character_runtime.rag_for(profile)
            if rag.document_count > 0:
                await asyncio.to_thread(
                    rag.get_relevant_context,
                    f"What should I know about {profile['name']}?",
                )
            prepare_profile = getattr(tts, "prepare_active_profile", None)
            if prepare_profile:
                await prepare_profile()

        try:
            asyncio.get_running_loop().create_task(prepare_character_assets())
        except RuntimeError:
            logger.debug("No event loop available for character asset warm-up.")
        logger.info("Activated character: %s (%s)", profile["name"], profile["id"])

    registry.set_activation_handler(activate_character)

    context_window = SlidingContextWindow(
        context=context,
        max_turns=config.LLM_MAX_CONTEXT_TURNS,
    )
    client_events = ClientEventProcessor()
    response_limiter = VoiceResponseLimiter(
        max_sentences=3,
        profile_provider=lambda: character_runtime.active,
        latest_user_provider=lambda: rag_processor.latest_user_text,
        tts_backend=tts_backend,
    )

    # ── WebSocket transport ──
    transport = WebsocketServerTransport(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            audio_out_channels=1,
            serializer=RawAudioSerializer(sample_rate=16000, num_channels=1),
        ),
    )

    vad_processor = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=16000,
            params=VADParams(
                confidence=config.VAD_CONFIDENCE,
                start_secs=config.VAD_START_SECS,
                stop_secs=config.VAD_STOP_SECS,
                min_volume=config.VAD_MIN_VOLUME,
            ),
        )
    )

    connection_state = {
        "unity_connected": False,
        "brain_id": running_runtime["brain"],
        "voice_id": running_runtime["voice"],
        "llm_model": llm_model,
    }
    greeting_tasks: set[asyncio.Task] = set()
    opening_index = 0

    async def greet_after_connect():
        nonlocal opening_index
        # Briefly yield so the user's first speech can win, without adding a
        # noticeable pause before a proactive character opens the conversation.
        await asyncio.sleep(0.12)
        if not connection_state["unity_connected"]:
            return
        # A Wi-Fi/WebSocket reconnect must resume the conversation silently,
        # not replay a greeting and make the character seem forgetful.
        if rag_processor._turn_count > 0:
            return
        profile = character_runtime.active
        conversation = profile.get("conversation") or {}
        if not conversation.get("greet_on_connect", True):
            return
        lines = list(conversation.get("opening_lines") or [])
        if not lines:
            return
        opening = str(lines[opening_index % len(lines)]).strip()
        opening_index += 1
        if not opening:
            return
        await task.queue_frames([
            LLMFullResponseStartFrame(),
            LLMTextFrame(opening),
            LLMFullResponseEndFrame(),
        ])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        connection_state["unity_connected"] = True
        logger.info("Client connected.")
        _audit("client_connected", character_id=character_runtime.active["id"])
        greeting = asyncio.create_task(greet_after_connect())
        greeting_tasks.add(greeting)
        greeting.add_done_callback(greeting_tasks.discard)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        connection_state["unity_connected"] = False
        for greeting in tuple(greeting_tasks):
            greeting.cancel()
        logger.info("Client disconnected.")
        _audit("client_disconnected", character_id=character_runtime.active["id"])

    # ── Pipeline ──
    # Frame flow:
    #   audio → STT → user_aggregator → rag_processor → LLM → TTS → audio out → assistant_aggregator
    pipeline = Pipeline(
        [
            transport.input(),          # Raw audio from Unity client (16 kHz)
            vad_processor,              # Turn detection and interruption trigger
            stt,                        # Faster Whisper → TranscriptionFrame
            client_events,              # JSON state/transcript messages for Unity
            user_aggregator,            # Adds user msg to context, pushes context frame
            rag_processor,              # Updates system prompt with RAG + emotion
            context_window,             # Bounds prompt prefill cost
            llm,                        # Ollama → LLMTextFrame stream
            response_limiter,           # Two complete spoken sentences maximum
            tts,                        # Selected local TTS → TTSAudioRawFrame stream
            transport.output(),         # Sends audio back to Unity client (24 kHz)
            assistant_aggregator,       # Records what was spoken into context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
        idle_timeout_secs=None,
    )

    runner = PipelineRunner()

    logger.info("=" * 60)
    logger.info("Conversational AI Avatar Server")
    logger.info("=" * 60)
    logger.info("WebSocket:  ws://%s:%d", config.SERVER_HOST, config.SERVER_PORT)
    if stt_backend == "moonshine":
        logger.info(
            "STT:        moonshine %s (streaming CPU)", config.MOONSHINE_ARCH
        )
    else:
        logger.info(
            "STT:        %s (%s, %s)",
            config.STT_MODEL,
            config.STT_DEVICE,
            config.STT_COMPUTE_TYPE,
        )
    logger.info("LLM:        %s via %s", llm_model, config.LLM_BASE_URL)
    if tts_backend == "pocket":
        logger.info(
            "TTS:        pocket streaming (voice=%s, worker=%s)",
            config.POCKET_TTS_VOICE,
            config.POCKET_TTS_URL,
        )
    elif tts_backend == "kokoro":
        logger.info("TTS:        kokoro (voice=%s, lang=%s)", config.TTS_VOICE, config.KOKORO_LANG_CODE)
    elif tts_backend == "chatterbox":
        logger.info("TTS:        chatterbox-turbo (device=%s)", config.CHATTERBOX_DEVICE)
    elif tts_backend == "orpheus-cpp":
        logger.info(
            "TTS:        orpheus-cpp (voice=%s, prebuffer=%.2fs)",
            config.TTS_VOICE,
            config.ORPHEUS_CPP_PREBUFFER_SECONDS,
        )
    else:
        logger.info("TTS:        orpheus %s (voice=%s, vllm=%s)", config.TTS_MODEL, config.TTS_VOICE, config.VLLM_BASE_URL)
    active_character = character_runtime.active
    logger.info("Character:  %s — %s", active_character["name"], active_character["description"])
    if rag_service.document_count > 0:
        logger.info("Knowledge:  %d document chunks loaded", rag_service.document_count)
    else:
        logger.info("Knowledge:  empty — add docs to knowledge/documents/ and run ingest.py")
    if emotion_processor.enabled:
        logger.info("Emotion:    enabled (%s)", config.EMOTION_MODEL)
    logger.info("=" * 60)
    logger.info("Waiting for Unity client connection...")
    logger.info("Control UI: http://%s:%d", config.CONTROL_HOST, config.CONTROL_PORT)

    import uvicorn

    control_app = create_control_app(
        character_runtime,
        tts_backend=tts_backend,
        running_runtime=running_runtime,
        status_provider=lambda: dict(connection_state),
    )
    control_server = uvicorn.Server(
        uvicorn.Config(
            control_app,
            host=config.CONTROL_HOST,
            port=config.CONTROL_PORT,
            log_level="warning",
            access_log=False,
        )
    )
    control_task = asyncio.create_task(control_server.serve())
    try:
        await runner.run(task)
    finally:
        control_server.should_exit = True
        await control_task


def main():
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.info("Server shut down by user.")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
