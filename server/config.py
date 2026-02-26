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

    # Speech-to-Text (Faster Whisper)
    STT_MODEL: str = os.getenv("STT_MODEL", "large-v3")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")
    STT_DEVICE: str = os.getenv("STT_DEVICE", "cuda")
    STT_COMPUTE_TYPE: str = os.getenv("STT_COMPUTE_TYPE", "float16")

    # LLM (Ollama)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.1:8b")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "150"))

    # Text-to-Speech (Orpheus via vllm + SNAC)
    TTS_MODEL: str = os.getenv("TTS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
    TTS_VOICE: str = os.getenv("TTS_VOICE", "tara")
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    # RAG / Knowledge Base
    RAG_DB_PATH: str = os.getenv("RAG_DB_PATH", "./knowledge/db")
    RAG_EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    RAG_COLLECTION_NAME: str = os.getenv("RAG_COLLECTION_NAME", "training_docs")

    # Character
    CHARACTER_NAME: str = os.getenv("CHARACTER_NAME", "Alex")
    CHARACTER_DESCRIPTION: str = os.getenv(
        "CHARACTER_DESCRIPTION", "a friendly and patient training instructor"
    )

    # Emotion Analysis (optional)
    EMOTION_ENABLED: bool = _bool(os.getenv("EMOTION_ENABLED", "false"))
    EMOTION_MODEL: str = os.getenv(
        "EMOTION_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )


config = Config()
