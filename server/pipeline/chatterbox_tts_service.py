"""Pipecat adapter for local Chatterbox Turbo and Multilingual models."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable

import numpy as np
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)

from pipeline.speech_text import (
    CHATTERBOX_FAST_STARTERS,
    expression_message,
    prepare_for_chatterbox,
    prepare_for_kokoro,
)

logger = logging.getLogger(__name__)


def smooth_audio_boundaries(
    audio: np.ndarray, sample_rate: int, fade_ms: float = 8.0
) -> np.ndarray:
    """Remove DC and ramp utterance edges to prevent playback clicks."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if samples.size == 0:
        return samples
    samples -= float(np.mean(samples))
    fade_samples = min(samples.size // 2, max(1, int(sample_rate * fade_ms / 1000.0)))
    ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=True, dtype=np.float32)
    samples[:fade_samples] *= ramp
    samples[-fade_samples:] *= ramp[::-1]
    return samples


class LowLatencySpeechAggregator(BaseTextAggregator):
    """Release sentences and natural clauses for continuous low-latency speech.

    Chatterbox generates a complete waveform per request rather than streaming
    one continuous generation. Natural comma/semicolon boundaries are safe places
    to build a small playback lead without cutting a word or an arbitrary phrase.
    """

    _WORD_RE = re.compile(r"[\w']+", re.UNICODE)
    _UNSAFE_END_WORDS = {
        "a", "an", "the", "and", "or", "but", "to", "of", "in", "on",
        "with", "for", "from", "as", "at", "by", "is", "are", "was",
        "were", "be", "been", "being", "that", "which", "who", "whose",
        "how", "what", "this", "these", "those", "its", "your", "our",
        "their", "my", "some", "like", "including", "use",
    }

    def __init__(self):
        self._text = ""
        self._emitted_any = False
        self._emitted_substantive = False

    @property
    def text(self) -> Aggregation:
        return Aggregation(self._text.strip(), AggregationType.SENTENCE)

    def _word_count(self) -> int:
        return len(self._WORD_RE.findall(self._text))

    def _inside_performance_tag(self) -> bool:
        return (
            self._text.count("[") > self._text.count("]")
            or self._text.count("<") > self._text.count(">")
            or self._text.count("*") % 2 == 1
        )

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        for char in text:
            self._text += char
            if self._inside_performance_tag():
                continue
            words = self._word_count()
            # Tiny generated fragments sound robotic and finish playing before
            # the next non-streaming Chatterbox request is ready. Keep enough
            # words together for the following synthesis to finish under the
            # audio already buffered by the transport.
            normalised = " ".join(self._text.split()).casefold()
            cached_sentence = any(
                normalised == " ".join(phrase.split()).casefold()
                for phrase in CHATTERBOX_FAST_STARTERS
            )
            sentence_end = char in ".!?" and (words >= 4 or cached_sentence)
            clause_end = char in ",;:" and (
                (not self._emitted_substantive and words >= 6) or words >= 8
            )
            found_words = self._WORD_RE.findall(self._text)
            last_word = found_words[-1].casefold() if found_words else ""
            # Llama 3B occasionally ignores the requested short first sentence.
            # Bound only that first phrase at a safe word boundary so a single
            # long sentence cannot impose eight to ten seconds of TTS latency.
            first_phrase_limit = (
                not self._emitted_substantive
                and char.isspace()
                and words >= 8
                and (last_word not in self._UNSAFE_END_WORDS or words >= 11)
            )
            if sentence_end or clause_end or first_phrase_limit:
                result = self._text.strip()
                self._text = ""
                if result:
                    self._emitted_any = True
                    if not cached_sentence:
                        self._emitted_substantive = True
                    yield Aggregation(result, AggregationType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        result = self._text.strip()
        self._text = ""
        self._emitted_any = False
        self._emitted_substantive = False
        if result and result[-1:] not in ".!?":
            logger.info("Dropping incomplete Chatterbox tail: %r", result)
            return None
        return Aggregation(result, AggregationType.SENTENCE) if result else None

    async def handle_interruption(self):
        self._text = ""
        self._emitted_any = False
        self._emitted_substantive = False

    async def reset(self):
        self._text = ""
        self._emitted_any = False
        self._emitted_substantive = False


class ChatterboxTTSService(TTSService):
    """Use Turbo for English and one-at-a-time Multilingual for other languages.

    Keeping only one variant on the GPU allows an 8 GB laptop GPU to retain
    Whisper and the LLM. Switching language families warms the other local
    model in the background when a character is activated.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        reference_audio: str = "",
        temperature: float = 0.8,
        chunk_ms: int = 40,
        warmup: bool = True,
        profile_provider: Callable[[], dict[str, Any]] | None = None,
        audit_callback: Callable[[dict[str, Any]], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            sample_rate=24000,
            push_stop_frames=True,
            text_aggregator=LowLatencySpeechAggregator(),
            **kwargs,
        )
        self._device = device
        self._reference_audio = reference_audio
        self._temperature = temperature
        self._chunk_ms = chunk_ms
        self._warmup = warmup
        self._profile_provider = profile_provider
        self._audit_callback = audit_callback
        self._model = None
        self._variant = ""
        self._model_sample_rate = 24000
        self._base_conditionals = None
        self._condition_cache: dict[tuple[str, str], Any] = {}
        self._generation_lock = threading.RLock()
        self._epoch_lock = threading.Lock()
        self._generation_epoch = 0
        self._active_voice_cache_key = ""
        self._instant_pcm: dict[tuple[str, str], bytes] = {}
        self._fast_cache_dir = (
            Path(__file__).resolve().parents[1] / "runtime" / "tts_cache"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterruptionFrame):
            with self._epoch_lock:
                self._generation_epoch += 1
                epoch = self._generation_epoch
            logger.info("Cancelled stale Chatterbox work at generation epoch %d.", epoch)
        await super().process_frame(frame, direction)

    def _current_epoch(self) -> int:
        with self._epoch_lock:
            return self._generation_epoch

    def _profile_settings(self) -> dict[str, Any]:
        return dict(self._profile_provider() if self._profile_provider else {})

    @staticmethod
    def _language(settings: dict[str, Any]) -> str:
        return str(settings.get("language") or "en").strip().lower()

    @classmethod
    def _desired_variant(cls, settings: dict[str, Any]) -> str:
        return "turbo" if cls._language(settings) == "en" else "multilingual"

    async def start(self, frame):
        await super().start(frame)
        await asyncio.get_running_loop().run_in_executor(None, self._load_model)
        logger.info(
            "Chatterbox ready (variant=%s, device=%s, reference=%s).",
            self._variant,
            self._device,
            self._reference_audio or "character/built-in",
        )

    async def prepare_active_profile(self) -> None:
        """Warm a newly activated character without blocking the control API."""
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, self._prepare_profile_sync
            )
        except Exception:
            logger.exception("Could not prepare the active Chatterbox language profile.")

    async def warm_for_startup(self) -> None:
        """Load and warm the active model before opening client ports."""
        await asyncio.get_running_loop().run_in_executor(
            None, self._prepare_profile_sync
        )

    def _prepare_profile_sync(self) -> None:
        settings = self._profile_settings()
        with self._generation_lock:
            if self._desired_variant(settings) != self._variant:
                self._load_variant(settings)
            else:
                self._select_voice(settings)
            self._warm_fast_starters_sync()

    def _load_model(self) -> None:
        # Reuse an explicit startup warm-up instead of loading the model twice.
        self._prepare_profile_sync()

    def _load_variant(self, settings: dict[str, Any]) -> None:
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as exc:
            raise RuntimeError(
                "Chatterbox is not installed. Use Python 3.11 and run "
                "`pip install chatterbox-tts`."
            ) from exc

        if torch.cuda.is_available():
            # RTX 30/40-series TensorFloat-32 materially speeds the FP32
            # transformer matrices without changing the model or voice.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        desired = self._desired_variant(settings)
        language = self._language(settings)
        if self._model is not None:
            self._model = None
            self._variant = ""
            self._base_conditionals = None
            self._condition_cache.clear()
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()

        model_class = (
            ChatterboxTurboTTS if desired == "turbo" else ChatterboxMultilingualTTS
        )
        logger.info("Loading Chatterbox %s for language=%s ...", desired, language)
        self._model = model_class.from_pretrained(device=self._device)
        self._variant = desired
        self._model_sample_rate = int(self._model.sr)
        self._base_conditionals = self._model.conds

        reference = str(settings.get("reference_audio") or self._reference_audio or "")
        if not reference and self._base_conditionals is None:
            raise RuntimeError(
                "Chatterbox needs a clean 6-10 second reference WAV. "
                "Assign one in the character control interface."
            )
        self._select_voice(settings)

        if self._warmup and self._device == "cuda":
            started = time.perf_counter()
            if desired == "turbo":
                self._model.generate("Ready.", temperature=self._temperature)
            else:
                self._model.generate(
                    "Ready.",
                    language_id=language,
                    exaggeration=float(settings.get("emotion_intensity", 0.65)),
                    cfg_weight=0.4,
                    temperature=float(settings.get("temperature", self._temperature)),
                )
            logger.info(
                "Chatterbox %s warm-up completed in %.2fs.",
                desired,
                time.perf_counter() - started,
            )

    def _select_voice(self, settings: dict[str, Any]) -> None:
        reference = str(settings.get("reference_audio") or self._reference_audio or "")
        if not reference:
            self._model.conds = self._base_conditionals
            self._active_voice_cache_key = self._voice_cache_key(settings)
            return
        if not Path(reference).is_file():
            raise FileNotFoundError(f"Character voice reference does not exist: {reference}")
        key = (self._variant, reference)
        if key not in self._condition_cache:
            logger.info("Preparing and caching Chatterbox voice: %s", reference)
            if self._variant == "multilingual":
                self._model.prepare_conditionals(
                    reference,
                    exaggeration=float(settings.get("emotion_intensity", 0.65)),
                )
            else:
                self._model.prepare_conditionals(reference)
            self._condition_cache[key] = self._model.conds
        self._model.conds = self._condition_cache[key]
        self._active_voice_cache_key = self._voice_cache_key(settings)

    @staticmethod
    def _normalise_cached_text(text: str) -> str:
        return " ".join(text.split()).casefold()

    def _voice_cache_key(self, settings: dict[str, Any]) -> str:
        reference = str(settings.get("reference_audio") or self._reference_audio or "")
        reference_stamp = "built-in"
        if reference:
            path = Path(reference)
            try:
                stat = path.stat()
                reference_stamp = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                reference_stamp = reference
        identity = "|".join((
            "fast-starters-v1",
            self._variant,
            reference_stamp,
            str(settings.get("temperature", self._temperature)),
        ))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def _warm_fast_starters_sync(self) -> None:
        settings = self._profile_settings()
        if self._variant != "turbo" or self._language(settings) != "en":
            return
        phrases = tuple(dict.fromkeys(
            (*CHATTERBOX_FAST_STARTERS, *settings.get("_fast_phrases", []))
        ))
        voice_key = self._active_voice_cache_key
        cache_dir = self._fast_cache_dir / voice_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        generated = 0
        for starter in phrases:
            prepared = prepare_for_chatterbox(starter)
            normalised = self._normalise_cached_text(prepared.text)
            memory_key = (voice_key, normalised)
            if memory_key in self._instant_pcm:
                continue
            filename = hashlib.sha256(
                prepared.text.encode("utf-8")
            ).hexdigest()[:24] + ".pcm"
            cache_path = cache_dir / filename
            pcm = b""
            try:
                if cache_path.is_file():
                    pcm = cache_path.read_bytes()
            except OSError:
                logger.warning("Could not read fast speech cache: %s", cache_path)
            if len(pcm) < 2:
                pcm = self._generate(
                    prepared.text,
                    self._current_epoch(),
                    record_audit=False,
                )
                if pcm:
                    try:
                        cache_path.write_bytes(pcm)
                    except OSError:
                        logger.warning("Could not persist fast speech cache: %s", cache_path)
                    generated += 1
            if pcm:
                self._instant_pcm[memory_key] = pcm
        logger.info(
            "Fast same-voice reaction cache ready: %d phrases (%d rendered) in %.2fs.",
            len(phrases),
            generated,
            time.perf_counter() - started,
        )

    def _generate(
        self, text: str, request_epoch: int, *, record_audit: bool = True
    ) -> bytes:
        settings = self._profile_settings()
        language = self._language(settings)
        temperature = float(settings.get("temperature", self._temperature))
        intensity = float(settings.get("emotion_intensity", 0.65))
        started = time.perf_counter()
        with self._generation_lock:
            if request_epoch != self._current_epoch():
                logger.info("Skipping stale queued Chatterbox request: %r", text[:60])
                return b""
            if self._desired_variant(settings) != self._variant:
                self._load_variant(settings)
            self._select_voice(settings)
            if self._variant == "turbo":
                wav = self._model.generate(text, temperature=temperature)
            else:
                wav = self._model.generate(
                    text,
                    language_id=language,
                    exaggeration=intensity,
                    cfg_weight=0.4,
                    temperature=temperature,
                )
        if request_epoch != self._current_epoch():
            logger.info("Discarding interrupted Chatterbox audio: %r", text[:60])
            return b""
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        audio = smooth_audio_boundaries(audio, self._model_sample_rate)
        elapsed = time.perf_counter() - started
        audio_seconds = len(audio) / self._model_sample_rate
        logger.info(
            "Chatterbox %s generated %.2fs audio in %.2fs (RTF %.2f).",
            self._variant,
            audio_seconds,
            elapsed,
            elapsed / max(audio_seconds, 0.001),
        )
        if record_audit and self._audit_callback:
            self._audit_callback({
                "text": text,
                "variant": self._variant,
                "generation_seconds": round(elapsed, 4),
                "audio_seconds": round(audio_seconds, 4),
            })
        np.clip(audio, -1.0, 1.0, out=audio)
        return (audio * 32767.0).astype(np.int16).tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        settings = self._profile_settings()
        language = self._language(settings)
        prepared = (
            prepare_for_chatterbox(text or "")
            if language == "en"
            else prepare_for_kokoro(text or "")
        )
        if not prepared.speakable:
            return

        logger.info(
            "TTS [%s]: chatterbox language=%s text=%r",
            context_id,
            language,
            prepared.text[:80],
        )
        cue = expression_message(prepared)
        if cue:
            yield OutputTransportMessageUrgentFrame(message=cue)
        yield TTSStartedFrame()

        request_epoch = self._current_epoch()
        voice_key = self._active_voice_cache_key
        memory_key = (voice_key, self._normalise_cached_text(prepared.text))
        pcm = self._instant_pcm.get(memory_key, b"")
        if pcm:
            audio_seconds = len(pcm) / (self._model_sample_rate * 2)
            logger.info(
                "Instant same-voice cache hit: %.2fs audio for %r.",
                audio_seconds,
                prepared.text,
            )
            if self._audit_callback:
                self._audit_callback({
                    "text": prepared.text,
                    "variant": self._variant,
                    "generation_seconds": 0.0,
                    "audio_seconds": round(audio_seconds, 4),
                    "cache_hit": True,
                })
        else:
            pcm = await asyncio.get_running_loop().run_in_executor(
                None, self._generate, prepared.text, request_epoch
            )
        if not pcm:
            yield TTSStoppedFrame()
            return
        # This event describes audio that actually exists and is about to be
        # queued. Unity deliberately ignores Pipecat's raw bot-token transcript,
        # which can include text later dropped by a limiter or interruption.
        yield OutputTransportMessageUrgentFrame(message={
            "v": 1,
            "type": "assistant_spoken_text",
            "text": prepared.text,
        })
        bytes_per_chunk = max(
            2, int(self._model_sample_rate * (self._chunk_ms / 1000.0)) * 2
        )
        for offset in range(0, len(pcm), bytes_per_chunk):
            chunk = pcm[offset : offset + bytes_per_chunk]
            if chunk:
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self._model_sample_rate,
                    num_channels=1,
                )

        yield TTSStoppedFrame()
