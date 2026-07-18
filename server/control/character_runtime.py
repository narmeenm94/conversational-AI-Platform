"""Active character, per-character RAG, and knowledge document operations."""

from __future__ import annotations

import hashlib
import logging
import wave
from pathlib import Path
from typing import Any

from control.character_registry import CharacterRegistry
from knowledge.ingest import READERS, SUPPORTED_EXTENSIONS, chunk_text, discover_files
from pipeline.rag_service import RAGService

logger = logging.getLogger(__name__)


class CharacterRuntime:
    def __init__(
        self,
        registry: CharacterRegistry,
        *,
        server_root: str | Path,
        embedding_model: str,
        embedding_device: str,
        default_top_k: int = 3,
    ):
        self.registry = registry
        self.server_root = Path(server_root).resolve()
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.default_top_k = default_top_k
        self._rag_services: dict[str, RAGService] = {}

    @property
    def active(self) -> dict[str, Any]:
        return self.registry.active

    def resolve_path(self, value: str, *, require_local: bool = False) -> Path:
        path = Path(value or ".")
        if not path.is_absolute():
            path = self.server_root / path
        path = path.resolve()
        if require_local:
            try:
                path.relative_to(self.server_root)
            except ValueError as exc:
                raise ValueError("Knowledge paths must stay inside the local server directory.") from exc
        return path

    def rag_for(self, profile: dict[str, Any] | None = None) -> RAGService:
        profile = profile or self.active
        key = profile["id"]
        if key in self._rag_services:
            return self._rag_services[key]
        knowledge = profile["knowledge"]
        rag = RAGService(
            db_path=str(self.resolve_path(knowledge["db_path"], require_local=True)),
            embedding_model=self.embedding_model,
            device=self.embedding_device,
            collection_name=knowledge["collection_name"],
            top_k=int(knowledge.get("top_k", self.default_top_k)),
            max_distance=float(knowledge.get("max_distance", 0.95)),
        )
        self._rag_services[key] = rag
        return rag

    def active_rag(self) -> RAGService:
        return self.rag_for(self.active)

    def voice_settings(self) -> dict[str, Any]:
        active = self.active
        settings = dict(active.get("voice") or {})
        settings["language"] = str(active.get("language") or "en").lower()
        # Internal-only same-voice phrases that should be ready before Unity
        # connects. Keeping proactive openers here avoids a live Chatterbox GPU
        # generation displacing the already-warm Ollama model.
        settings["_fast_phrases"] = list(
            (active.get("conversation") or {}).get("opening_lines") or []
        )
        reference = str(settings.get("reference_audio") or "").strip()
        if reference:
            settings["reference_audio"] = str(self.resolve_path(reference))
        return settings

    def voice_directory(self) -> Path:
        directory = self.server_root / "voices"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def list_voices(self) -> list[dict[str, Any]]:
        root = self.voice_directory()
        voices = []
        for path in sorted(root.rglob("*.wav")):
            try:
                with wave.open(str(path), "rb") as wav:
                    rate = wav.getframerate()
                    duration = wav.getnframes() / max(rate, 1)
                    channels = wav.getnchannels()
            except (OSError, wave.Error):
                continue
            relative = path.relative_to(root).as_posix()
            voices.append({
                "name": path.stem.replace("_", " ").replace("-", " ").title(),
                "path": f"./voices/{relative}",
                "duration_seconds": round(duration, 2),
                "sample_rate": rate,
                "channels": channels,
                "bytes": path.stat().st_size,
            })
        return voices

    def save_voice_file(self, filename: str, content: bytes) -> dict[str, Any]:
        root = self.voice_directory()
        relative = Path(filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid voice filename.")
        if relative.suffix.lower() != ".wav":
            raise ValueError("Chatterbox reference voices must be WAV files.")
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("Voice file must stay inside the local voice library.") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        try:
            with wave.open(str(target), "rb") as wav:
                duration = wav.getnframes() / max(wav.getframerate(), 1)
        except (OSError, wave.Error) as exc:
            target.unlink(missing_ok=True)
            raise ValueError("The uploaded file is not a readable PCM WAV.") from exc
        if duration <= 5.0:
            target.unlink(missing_ok=True)
            raise ValueError("Voice reference must be longer than 5 seconds; 6-10 is ideal.")
        expected = f"./voices/{relative.as_posix()}"
        return next(voice for voice in self.list_voices() if voice["path"] == expected)

    def knowledge_directory(self, profile: dict[str, Any]) -> Path:
        directory = self.resolve_path(
            profile["knowledge"]["documents_path"], require_local=True
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def list_knowledge_files(self, key: str) -> list[dict[str, Any]]:
        profile = self.registry.get(key)
        root = self.knowledge_directory(profile)
        items = []
        for path in discover_files(root):
            items.append({
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
            })
        return items

    def save_knowledge_file(self, key: str, filename: str, content: bytes) -> dict[str, Any]:
        profile = self.registry.get(key)
        root = self.knowledge_directory(profile)
        relative = Path(filename.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid knowledge filename.")
        if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Supported knowledge files are: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("Knowledge file must stay inside the character directory.") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"name": relative.as_posix(), "bytes": len(content)}

    def delete_knowledge_file(self, key: str, filename: str) -> None:
        profile = self.registry.get(key)
        root = self.knowledge_directory(profile)
        target = (root / filename.replace("\\", "/")).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("Invalid knowledge filename.") from exc
        if not target.is_file():
            raise FileNotFoundError(filename)
        target.unlink()

    def reindex(self, key: str) -> dict[str, Any]:
        profile = self.registry.get(key)
        root = self.knowledge_directory(profile)
        rag = self.rag_for(profile)
        rag.reset()

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        files = discover_files(root)
        for path in files:
            reader = READERS.get(path.suffix.lower())
            if reader is None:
                continue
            text = reader(path)
            chunks = chunk_text(text)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            for index, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({
                    "source": path.relative_to(root).as_posix(),
                    "chunk_index": index,
                    "character_id": profile["id"],
                })
                ids.append(f"{profile['id']}_{digest}_{index}")
        rag.add_documents(documents, metadatas=metadatas, ids=ids)
        logger.info(
            "Indexed character=%s files=%d chunks=%d",
            profile["id"], len(files), len(documents),
        )
        return {"files": len(files), "chunks": len(documents)}

    def status(self, key: str) -> dict[str, Any]:
        profile = self.registry.get(key)
        return {
            "character_id": key,
            "files": self.list_knowledge_files(key),
            "chunks": self.rag_for(profile).document_count,
        }
