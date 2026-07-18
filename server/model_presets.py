"""Curated local inference choices exposed by the character control panel."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BRAINS: dict[str, dict[str, Any]] = {
    "llama3.2-3b": {
        "label": "Llama 3.2 3B · natural dialogue",
        "model": "llama3.2:3b",
        "summary": "Best character consistency that fits beside Chatterbox on this 8 GB laptop.",
        "hardware": "8 GB VRAM current minimum; 12 GB gives safer headroom",
        "warm_latency": "~0.29 s model-only first token measured here",
        "license": "Meta Llama 3.2 Community License",
    },
    "lfm-1.2b": {
        "label": "LFM 2.5 1.2B · realtime",
        "model": "LiquidAI/lfm2.5-1.2b-instruct:latest",
        "summary": "Fastest good conversational result on this laptop; the default for live dialogue.",
        "hardware": "CPU-capable; 6–8 GB VRAM system recommended with expressive TTS",
        "warm_latency": "~0.08 s model-only first token measured here",
        "license": "LFM Open License (commercial threshold applies)",
    },
    "qwen3.5-4b": {
        "label": "Qwen 3.5 4B · richer character",
        "model": "qwen3.5:4b",
        "summary": "Better character discipline and nuance, with more memory and a slower cold start.",
        "hardware": "12 GB VRAM recommended with Chatterbox; 16–24 GB with Orpheus",
        "warm_latency": "~0.28 s model-only first token measured here",
        "license": "Apache 2.0",
    },
    "granite-1b": {
        "label": "Granite 4 H 1B · permissive fallback",
        "model": "granite4:1b-h",
        "summary": "Small Apache-licensed fallback, but less reliable for natural spoken formatting.",
        "hardware": "CPU-capable; 6–8 GB VRAM system recommended with expressive TTS",
        "warm_latency": "~0.22 s model-only first token measured here",
        "license": "Apache 2.0",
    },
    "qwen3-0.6b": {
        "label": "Qwen 3 0.6B · tiny experiment",
        "model": "qwen3:0.6b",
        "summary": "Very small, but the local tests were noticeably less natural and less obedient.",
        "hardware": "CPU or low-memory edge PC",
        "warm_latency": "~0.11 s model-only first token measured here",
        "license": "Apache 2.0",
    },
    "phi4-mini": {
        "label": "Phi-4 Mini 3.8B · reasoning",
        "model": "phi4-mini:3.8b",
        "summary": "Useful for structured reasoning, but inclined to longer, less casual responses.",
        "hardware": "12 GB VRAM recommended with expressive TTS",
        "warm_latency": "~0.19 s model-only first token measured here",
        "license": "MIT",
    },
}


VOICES: dict[str, dict[str, Any]] = {
    "pocket": {
        "label": "Pocket TTS 100M · true CPU streaming",
        "backend": "pocket",
        "summary": "Fast incremental CPU speech that keeps the GPU free for the brain; it has no native laugh, sigh, or groan control tokens.",
        "hardware": "Modern CPU; two or more performance cores recommended",
        "warm_latency": "0.39–0.52 s first PCM measured through the local worker",
        "languages": "English optimized; official previews cover French, German, Italian, Spanish, and Portuguese",
        "license": "MIT code; CC BY 4.0 model weights",
    },
    "chatterbox": {
        "label": "Chatterbox Turbo · expressive realtime",
        "backend": "chatterbox",
        "summary": "Best current balance: female voice cloning, emotion sounds, and moderate GPU use.",
        "hardware": "NVIDIA GPU recommended; current 8 GB laptop profile",
        "warm_latency": "Balanced; sentence generation preserves consistent prosody",
        "languages": "English Turbo; 23-language Multilingual model",
        "license": "MIT",
    },
    "kokoro": {
        "label": "Kokoro 82M · absolute fastest",
        "backend": "kokoro",
        "summary": "Smallest and fastest local voice; natural but without genuine non-verbal emotion cues.",
        "hardware": "CPU-capable and suitable for lighter deployments",
        "warm_latency": "Lowest voice startup latency",
        "languages": "Language/voice dependent",
        "license": "Apache 2.0 model",
    },
    "orpheus-cpp": {
        "label": "Orpheus 3B CPP · natural quality experiment",
        "backend": "orpheus-cpp",
        "summary": "Human-like emotion tags and rhythm, but slower than Chatterbox in the Windows benchmark.",
        "hardware": "12 GB VRAM recommended; isolated Python environment required",
        "warm_latency": "~0.73 s first audio measured here at 0.2 s prebuffer",
        "languages": "English plus research multilingual checkpoints",
        "license": "Apache 2.0",
    },
    "orpheus-vllm": {
        "label": "Orpheus 3B vLLM · high-end streaming",
        "backend": "orpheus",
        "summary": "The intended high-throughput streaming path for a stronger dedicated inference PC.",
        "hardware": "16 GB minimum; 24 GB VRAM recommended with a quality brain",
        "warm_latency": "Official project target ~0.2 s streaming; verify on deployment hardware",
        "languages": "English plus research multilingual checkpoints",
        "license": "Apache 2.0",
    },
}


PRESETS: dict[str, dict[str, Any]] = {
    "instant-streaming": {
        "label": "Instant streaming · recommended",
        "brain": "llama3.2-3b",
        "voice": "pocket",
        "summary": "The best tested laptop balance: stronger character continuity on GPU and genuinely streaming cloned speech on CPU.",
        "hardware": "8 GB NVIDIA GPU plus a modern multi-core CPU",
        "expected_turn": "Target sub-second warm reaction onset; measured Pocket first PCM is 0.39–0.52 s",
    },
    "instant-expressive": {
        "label": "Instant expressive · laptop",
        "brain": "lfm-1.2b",
        "voice": "chatterbox",
        "summary": "Fast 1.2B dialogue brain plus a cached same-voice thinking phrase while expressive speech renders.",
        "hardware": "8 GB VRAM current laptop profile",
        "expected_turn": "Immediate natural acknowledgement; substantive speech follows from the local expressive model",
    },
    "balanced-realtime": {
        "label": "Realtime expressive · recommended",
        "brain": "llama3.2-3b",
        "voice": "chatterbox",
        "summary": "Current 8 GB profile: character-consistent brain, cloned expressive voice, barge-in enabled.",
        "hardware": "8 GB VRAM / 16 GB system RAM minimum",
        "expected_turn": "Roughly 0.9–1.5 s warm end-of-speech to audio; room and microphone affect VAD",
    },
    "fastest": {
        "label": "Fastest possible · light",
        "brain": "lfm-1.2b",
        "voice": "kokoro",
        "summary": "Choose when reaction speed matters more than laughs, sighs, and voice cloning.",
        "hardware": "Modern CPU; discrete GPU optional",
        "expected_turn": "Roughly 0.7–1.2 s warm end-of-speech to audio",
    },
    "richer-character": {
        "label": "Richer character · stronger PC",
        "brain": "qwen3.5-4b",
        "voice": "chatterbox",
        "summary": "More nuanced dialogue while keeping the proven expressive voice path.",
        "hardware": "12 GB VRAM recommended",
        "expected_turn": "Roughly 1.0–1.8 s warm end-of-speech to audio",
    },
    "orpheus-natural": {
        "label": "Natural Orpheus · experimental",
        "brain": "lfm-1.2b",
        "voice": "orpheus-cpp",
        "summary": "Prioritizes natural vocal performance over minimum response latency.",
        "hardware": "12 GB VRAM recommended; isolated Orpheus runtime",
        "expected_turn": "Roughly 1.3–2.0 s on the tested laptop; benchmark the target machine",
    },
    "premium-streaming": {
        "label": "Premium streaming · workstation",
        "brain": "qwen3.5-4b",
        "voice": "orpheus-vllm",
        "summary": "Higher-quality brain and Orpheus' intended streaming server path.",
        "hardware": "24 GB VRAM recommended, or separate LLM and TTS GPUs",
        "expected_turn": "Hardware-dependent; designed for sub-second optimized streaming",
    },
}


DEFAULT_RUNTIME = {
    "preset": "instant-streaming",
    "brain": "llama3.2-3b",
    "voice": "pocket",
}


def normalize_runtime(value: Any) -> dict[str, str]:
    runtime = dict(DEFAULT_RUNTIME)
    if isinstance(value, dict):
        runtime.update({key: str(value.get(key) or runtime[key]) for key in runtime})
    if runtime["preset"] not in PRESETS:
        raise ValueError("Unknown performance preset.")
    if runtime["brain"] not in BRAINS:
        raise ValueError("Unknown brain engine.")
    if runtime["voice"] not in VOICES:
        raise ValueError("Unknown voice engine.")
    return runtime


def resolve_runtime(profile: dict[str, Any]) -> dict[str, Any]:
    selected = normalize_runtime(profile.get("runtime"))
    return {
        **selected,
        "llm_model": BRAINS[selected["brain"]]["model"],
        "tts_backend": VOICES[selected["voice"]]["backend"],
    }


def catalog() -> dict[str, Any]:
    return {
        "default": deepcopy(DEFAULT_RUNTIME),
        "presets": deepcopy(PRESETS),
        "brains": deepcopy(BRAINS),
        "voices": deepcopy(VOICES),
    }
