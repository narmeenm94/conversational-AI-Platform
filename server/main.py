"""Conversational AI Avatar Server — Pipecat pipeline + WebSocket entry point.

Assembles a real-time voice AI pipeline:
  Audio In → Whisper STT → [RAG + Emotion] → Ollama LLM → Orpheus TTS → Audio Out

All communication with the Unity client (Quest 3 or Desktop) happens
over a single WebSocket connection carrying raw PCM audio frames.
"""

import asyncio
import logging
import sys

from config import config

# ── VAD ──
from pipecat.audio.vad.silero import SileroVADAnalyzer

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
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.ollama.llm import OLLamaLLMService

# ── Context management ──
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)

# ── Frame processing ──
from pipecat.frames.frames import Frame, TranscriptionFrame, LLMMessagesFrame
from pipecat.processors.aggregators.llm_response import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# ── Local services ──
from pipeline.raw_audio_serializer import RawAudioSerializer
from pipeline.tts_service import OrpheusTTSService
from pipeline.rag_service import RAGService
from pipeline.emotion_processor import EmotionProcessor
from pipeline.llm_service import build_system_prompt, build_base_system_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


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
        rag_service: RAGService,
        emotion_processor: EmotionProcessor,
        character_name: str,
        character_description: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._rag = rag_service
        self._emotion = emotion_processor
        self._character_name = character_name
        self._character_description = character_description
        self._turn_count = 0

    def _get_latest_user_text(self) -> str | None:
        """Walk backwards through context to find the last user message."""
        for msg in reversed(self._context.messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def _update_system_message(self, user_text: str):
        """Rebuild the system prompt with fresh RAG + emotion context."""
        self._turn_count += 1
        new_prompt = build_system_prompt(
            character_name=self._character_name,
            character_description=self._character_description,
            user_text=user_text,
            rag_service=self._rag,
            emotion_processor=self._emotion,
            turn_count=self._turn_count,
        )
        if self._context.messages and self._context.messages[0].get("role") == "system":
            self._context.messages[0]["content"] = new_prompt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (OpenAILLMContextFrame, LLMMessagesFrame)):
            user_text = self._get_latest_user_text()
            if user_text:
                self._update_system_message(user_text)
                logger.info("User: %s", user_text)

        await self.push_frame(frame, direction)


async def run_pipeline():
    """Initialize all services and run the Pipecat pipeline."""

    # ── RAG Service ──
    logger.info("Initializing RAG service...")
    rag_service = RAGService(
        db_path=config.RAG_DB_PATH,
        embedding_model=config.RAG_EMBEDDING_MODEL,
        collection_name=config.RAG_COLLECTION_NAME,
        top_k=config.RAG_TOP_K,
    )

    # ── Emotion Processor (optional) ──
    emotion_processor = EmotionProcessor(
        enabled=config.EMOTION_ENABLED,
        model_name=config.EMOTION_MODEL,
    )

    # ── STT (Pipecat built-in Faster Whisper) ──
    logger.info("Initializing STT: model=%s device=%s", config.STT_MODEL, config.STT_DEVICE)
    stt = WhisperSTTService(
        model=config.STT_MODEL,
        device=config.STT_DEVICE,
        compute_type=config.STT_COMPUTE_TYPE,
        language=config.STT_LANGUAGE,
    )

    # ── LLM (Ollama, OpenAI-compatible API) ──
    logger.info("Initializing LLM: model=%s", config.LLM_MODEL)
    llm = OLLamaLLMService(
        model=config.LLM_MODEL,
        base_url=config.LLM_BASE_URL + "/v1",
    )

    # ── TTS (Orpheus, custom Pipecat service) ──
    logger.info("Initializing TTS: model=%s voice=%s", config.TTS_MODEL, config.TTS_VOICE)
    tts = OrpheusTTSService(
        model_name=config.TTS_MODEL,
        voice=config.TTS_VOICE,
    )

    # ── Conversation context ──
    base_prompt = build_base_system_prompt(
        config.CHARACTER_NAME, config.CHARACTER_DESCRIPTION,
    )
    messages = [{"role": "system", "content": base_prompt}]
    context = LLMContext(messages)
    user_aggregator = LLMUserContextAggregator(context)
    assistant_aggregator = LLMAssistantContextAggregator(context)

    # ── RAG context processor ──
    rag_processor = RAGContextProcessor(
        context=context,
        rag_service=rag_service,
        emotion_processor=emotion_processor,
        character_name=config.CHARACTER_NAME,
        character_description=config.CHARACTER_DESCRIPTION,
    )

    # ── WebSocket transport ──
    transport = WebsocketServerTransport(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        params=WebsocketServerParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            audio_out_channels=1,
            vad_analyzer=SileroVADAnalyzer(sample_rate=16000),
            serializer=RawAudioSerializer(sample_rate=16000, num_channels=1),
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected.")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected.")

    # ── Pipeline ──
    # Frame flow:
    #   audio → STT → user_aggregator → rag_processor → LLM → TTS → audio out → assistant_aggregator
    pipeline = Pipeline(
        [
            transport.input(),          # Raw audio from Unity client (16 kHz)
            stt,                        # Faster Whisper → TranscriptionFrame
            user_aggregator,            # Adds user msg to context, pushes context frame
            rag_processor,              # Updates system prompt with RAG + emotion
            llm,                        # Ollama → LLMTextFrame stream
            tts,                        # Orpheus → TTSAudioRawFrame stream
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
    logger.info("STT:        %s (%s, %s)", config.STT_MODEL, config.STT_DEVICE, config.STT_COMPUTE_TYPE)
    logger.info("LLM:        %s via %s", config.LLM_MODEL, config.LLM_BASE_URL)
    logger.info("TTS:        %s (voice=%s)", config.TTS_MODEL, config.TTS_VOICE)
    logger.info("Character:  %s — %s", config.CHARACTER_NAME, config.CHARACTER_DESCRIPTION)
    if rag_service.document_count > 0:
        logger.info("Knowledge:  %d document chunks loaded", rag_service.document_count)
    else:
        logger.info("Knowledge:  empty — add docs to knowledge/documents/ and run ingest.py")
    if emotion_processor.enabled:
        logger.info("Emotion:    enabled (%s)", config.EMOTION_MODEL)
    logger.info("=" * 60)
    logger.info("Waiting for Unity client connection...")

    await runner.run(task)


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
