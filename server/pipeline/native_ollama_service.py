"""Pipecat Ollama service using the native streaming API.

Ollama's OpenAI-compatible endpoint currently ignores ``think=false`` for
Qwen 3.5 and can spend the entire completion budget on hidden reasoning. The
native endpoint honors it, producing the spoken response immediately.
"""

from __future__ import annotations

import json
import logging

import httpx
from pipecat.services.ollama.llm import OLLamaLLMService


logger = logging.getLogger(__name__)


class NativeOllamaLLMService(OLLamaLLMService):
    def __init__(self, *, native_base_url: str, **kwargs):
        self._native_base_url = native_base_url.rstrip("/")
        self._native_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=None, write=10.0, pool=3.0)
        )
        super().__init__(base_url=self._native_base_url + "/v1", **kwargs)

    def _native_options(self) -> dict:
        extra = dict(self._settings.get("extra") or {})
        extra_body = dict(extra.get("extra_body") or {})
        options = dict(extra_body.get("options") or {})
        options.update({
            "temperature": self._settings.get("temperature"),
            "top_p": self._settings.get("top_p"),
            "num_predict": self._settings.get("max_tokens"),
        })
        return {key: value for key, value in options.items() if value is not None}

    async def _process_context(self, context):
        payload = {
            "model": self.model_name,
            "messages": context.messages,
            "stream": True,
            "think": False,
            "keep_alive": (
                (self._settings.get("extra") or {})
                .get("extra_body", {})
                .get("keep_alive", "30m")
            ),
            "options": self._native_options(),
        }
        await self.start_ttfb_metrics()
        first_text = True
        async with self._native_client.stream(
            "POST", self._native_base_url + "/api/chat", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                text = str((chunk.get("message") or {}).get("content") or "")
                if text:
                    if first_text:
                        first_text = False
                        await self.stop_ttfb_metrics()
                    await self._push_llm_text(text)
                if chunk.get("done"):
                    break
        if first_text:
            await self.stop_ttfb_metrics()

    async def stop(self, frame):
        await self._native_client.aclose()
        await super().stop(frame)

