import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

_hf_token = os.getenv("HF_TOKEN", "")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
else:
    # Some Hugging Face clients treat an empty token as a literal `Bearer `
    # header instead of anonymous access.
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

_hf_home = os.getenv("HF_HOME", "")
if _hf_home:
    os.environ["HF_HOME"] = _hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_hf_home, "hub")


def _bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


class Config:
    """Centralized configuration loaded from environment / .env file."""

    # Server
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8765"))
    CONTROL_HOST: str = os.getenv("CONTROL_HOST", "0.0.0.0")
    CONTROL_PORT: int = int(os.getenv("CONTROL_PORT", "8766"))

    # Speech-to-Text. Moonshine performs recognition while the user is still
    # speaking; Faster Whisper remains the compatibility fallback.
    STT_BACKEND: str = os.getenv("STT_BACKEND", "whisper").strip().lower()
    STT_MODEL: str = os.getenv("STT_MODEL", "small.en")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")
    STT_DEVICE: str = os.getenv("STT_DEVICE", "cuda")
    STT_COMPUTE_TYPE: str = os.getenv("STT_COMPUTE_TYPE", "float16")
    # Threshold for filtering segments Whisper labels as non-speech.
    # Pipecat default is 0.4 which rejects too aggressively on short / quiet
    # utterances with the smaller models — 0.6 keeps things responsive
    # without letting noise through.
    STT_NO_SPEECH_PROB: float = float(os.getenv("STT_NO_SPEECH_PROB", "0.6"))
    MOONSHINE_URL: str = os.getenv("MOONSHINE_URL", "http://127.0.0.1:8771")
    MOONSHINE_PYTHON: str = os.getenv(
        "MOONSHINE_PYTHON", "./.venv-moonshine/Scripts/python.exe"
    )
    MOONSHINE_ARCH: str = os.getenv("MOONSHINE_ARCH", "small-streaming")
    MOONSHINE_AUTOSTART: bool = _bool(os.getenv("MOONSHINE_AUTOSTART", "true"))
    MOONSHINE_PRE_ROLL_SECONDS: float = float(
        os.getenv("MOONSHINE_PRE_ROLL_SECONDS", "0.45")
    )

    # Voice activity / turn detection. These values target natural pauses while
    # keeping end-of-turn latency below roughly half a second.
    VAD_CONFIDENCE: float = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    VAD_START_SECS: float = float(os.getenv("VAD_START_SECS", "0.2"))
    VAD_STOP_SECS: float = float(os.getenv("VAD_STOP_SECS", "0.18"))
    VAD_MIN_VOLUME: float = float(os.getenv("VAD_MIN_VOLUME", "0.6"))

    # LLM (Ollama)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "1.0"))
    LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "50"))
    LLM_REPEAT_PENALTY: float = float(os.getenv("LLM_REPEAT_PENALTY", "1.05"))
    # Enough room to finish the configured spoken sentences. The response
    # limiter still stops at the character's sentence cap, so this prevents
    # clipped endings without making ordinary turns longer.
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "160"))
    LLM_MAX_CONTEXT_TURNS: int = int(os.getenv("LLM_MAX_CONTEXT_TURNS", "4"))
    # Ollama's OpenAI-compatible endpoint currently runs this model at 4096.
    # Matching its runner context during warm-up avoids a multi-second first-
    # turn reload caused by warming a differently sized 3072-token runner.
    LLM_CONTEXT_SIZE: int = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
    LLM_KEEP_ALIVE: str = os.getenv("LLM_KEEP_ALIVE", "30m")

    # Text-to-Speech
    # TTS_BACKEND: "pocket" (CPU streaming) | "kokoro" (small/fast)
    # | "chatterbox" (expressive English) | "orpheus-cpp" (local quantized)
    # | "orpheus" (vLLM server).
    TTS_BACKEND: str = os.getenv("TTS_BACKEND", "kokoro").strip().lower()
    TTS_MODEL: str = os.getenv("TTS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "tara")
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    ORPHEUS_CPP_PREBUFFER_SECONDS: float = float(
        os.getenv("ORPHEUS_CPP_PREBUFFER_SECONDS", "0.5")
    )
    ORPHEUS_CPP_GPU_LAYERS: int = int(os.getenv("ORPHEUS_CPP_GPU_LAYERS", "-1"))
    ORPHEUS_CPP_VERBOSE: bool = _bool(os.getenv("ORPHEUS_CPP_VERBOSE", "false"))
    # Kokoro-specific (only used when TTS_BACKEND=kokoro)
    KOKORO_LANG_CODE: str = os.getenv("KOKORO_LANG_CODE", "a")
    KOKORO_SPEED: float = float(os.getenv("KOKORO_SPEED", "1.0"))
    KOKORO_DEVICE: str = os.getenv("KOKORO_DEVICE", "")  # "" = auto, "cuda", "cpu"
    # Pocket runs in a clean Python 3.12 CPU worker so its dependencies cannot
    # disturb Chatterbox or the main GPU pipeline.
    POCKET_TTS_URL: str = os.getenv("POCKET_TTS_URL", "http://127.0.0.1:8770")
    POCKET_TTS_PYTHON: str = os.getenv("POCKET_TTS_PYTHON", "./.venv-pocket/Scripts/python.exe")
    POCKET_TTS_VOICE: str = os.getenv("POCKET_TTS_VOICE", "azelma")
    POCKET_TTS_LANGUAGE: str = os.getenv("POCKET_TTS_LANGUAGE", "english")
    POCKET_TTS_AUTOSTART: bool = _bool(os.getenv("POCKET_TTS_AUTOSTART", "true"))
    # Chatterbox Turbo-specific. A clean 6-10 second reference clip is strongly
    # recommended for stable voice identity.
    CHATTERBOX_DEVICE: str = os.getenv("CHATTERBOX_DEVICE", "")
    CHATTERBOX_REFERENCE_AUDIO: str = os.getenv("CHATTERBOX_REFERENCE_AUDIO", "")
    CHATTERBOX_TEMPERATURE: float = float(os.getenv("CHATTERBOX_TEMPERATURE", "0.8"))
    CHATTERBOX_WARMUP: bool = os.getenv("CHATTERBOX_WARMUP", "true").lower() == "true"

    # RAG / Knowledge Base
    RAG_DB_PATH: str = os.getenv("RAG_DB_PATH", "./knowledge/db")
    RAG_EMBEDDING_MODEL: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    RAG_DEVICE: str = os.getenv("RAG_DEVICE", "cpu")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    RAG_COLLECTION_NAME: str = os.getenv("RAG_COLLECTION_NAME", "training_docs")

    # Character
    CHARACTER_NAME: str = os.getenv("CHARACTER_NAME", "Alex")
    CHARACTER_DESCRIPTION: str = os.getenv(
        "CHARACTER_DESCRIPTION", "a friendly and patient training instructor"
    )
    CHARACTER_PROFILE_PATH: str = os.getenv("CHARACTER_PROFILE_PATH", "")
    CHARACTERS_DIR: str = os.getenv("CHARACTERS_DIR", "./characters")
    ACTIVE_CHARACTER_ID: str = os.getenv("ACTIVE_CHARACTER_ID", "alex")

    # Emotion Analysis (optional)
    EMOTION_ENABLED: bool = _bool(os.getenv("EMOTION_ENABLED", "false"))
    EMOTION_MODEL: str = os.getenv(
        "EMOTION_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


config = Config()
