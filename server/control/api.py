"""Local FastAPI control surface for character and knowledge management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from control.character_runtime import CharacterRuntime
from model_presets import catalog, resolve_runtime


def create_control_app(
    runtime: CharacterRuntime,
    *,
    tts_backend: str,
    running_runtime: dict[str, Any],
    status_provider: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    app = FastAPI(title="Conversational AI Character Control", version="0.1.0")
    ui_path = Path(__file__).resolve().parent / "static" / "index.html"
    logo_path = Path(__file__).resolve().parents[2] / "metropolia_s_oranssi_en.png"

    def fail(exc: Exception):
        if isinstance(exc, KeyError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, FileExistsError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, (ValueError, FileNotFoundError)):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise exc

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return ui_path.read_text(encoding="utf-8")

    @app.get("/assets/metropolia-logo.png", response_class=FileResponse)
    async def metropolia_logo():
        return FileResponse(logo_path, media_type="image/png")

    @app.get("/api/status")
    async def status():
        active = runtime.active
        language = str(active.get("language") or "en").lower()
        active_backend = tts_backend
        if tts_backend == "chatterbox":
            active_backend = (
                "chatterbox-turbo" if language == "en" else "chatterbox-multilingual"
            )
        payload = {
            "active_character_id": active["id"],
            "active_character_name": active["name"],
            "language": language,
            "tts_backend": active_backend,
            "knowledge_chunks": runtime.active_rag().document_count,
            "runtime": dict(running_runtime),
        }
        requested = resolve_runtime(active)
        payload["requested_runtime"] = requested
        payload["restart_required"] = any(
            requested[key] != running_runtime[key] for key in ("brain", "voice")
        )
        if status_provider:
            payload.update(status_provider())
        return payload

    @app.get("/api/model-presets")
    async def model_presets():
        return catalog()

    @app.get("/api/characters")
    async def list_characters():
        return {
            "active_id": runtime.registry.active_id,
            "characters": runtime.registry.list(),
        }

    @app.get("/api/voices")
    async def list_voices():
        return {"voices": runtime.list_voices()}

    @app.put("/api/voices/{filename:path}")
    async def upload_voice(filename: str, request: Request):
        try:
            return runtime.save_voice_file(filename, await request.body())
        except Exception as exc:
            fail(exc)

    @app.get("/api/characters/{key}")
    async def get_character(key: str):
        try:
            return runtime.registry.get(key)
        except Exception as exc:
            fail(exc)

    @app.post("/api/characters", status_code=201)
    async def create_character(payload: dict[str, Any] = Body(...)):
        try:
            return runtime.registry.create(payload)
        except Exception as exc:
            fail(exc)

    @app.put("/api/characters/{key}")
    async def update_character(key: str, payload: dict[str, Any] = Body(...)):
        try:
            runtime.registry.get(key)
            payload["id"] = key
            saved = runtime.registry.save(payload)
            if runtime.registry.active_id == key:
                runtime.registry.activate(key)
            return saved
        except Exception as exc:
            fail(exc)

    @app.delete("/api/characters/{key}", status_code=204)
    async def delete_character(key: str):
        try:
            runtime.registry.delete(key)
        except Exception as exc:
            fail(exc)

    @app.post("/api/characters/{key}/activate")
    async def activate_character(key: str):
        try:
            return runtime.registry.activate(key)
        except Exception as exc:
            fail(exc)

    @app.get("/api/characters/{key}/knowledge")
    async def knowledge_status(key: str):
        try:
            return runtime.status(key)
        except Exception as exc:
            fail(exc)

    @app.put("/api/characters/{key}/knowledge/files/{filename:path}")
    async def upload_knowledge(key: str, filename: str, request: Request):
        try:
            return runtime.save_knowledge_file(key, filename, await request.body())
        except Exception as exc:
            fail(exc)

    @app.delete("/api/characters/{key}/knowledge/files/{filename:path}", status_code=204)
    async def delete_knowledge(key: str, filename: str):
        try:
            runtime.delete_knowledge_file(key, filename)
        except Exception as exc:
            fail(exc)

    @app.post("/api/characters/{key}/knowledge/reindex")
    async def reindex_knowledge(key: str):
        try:
            return runtime.reindex(key)
        except Exception as exc:
            fail(exc)

    return app
