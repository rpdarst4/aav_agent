"""Chat with a local model through Ollama, returning plain dicts."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Remove reasoning blocks some models emit inline."""
    return _THINK.sub("", text or "").strip()


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_response(resp: Any) -> dict:
    """{'content': str, 'tool_calls': [{'name': str, 'arguments': dict | str}]} from an Ollama response."""
    msg = _get(resp, "message")
    calls = []
    for tc in _get(msg, "tool_calls") or []:
        fn = _get(tc, "function")
        args = _get(fn, "arguments")
        if args is None:
            args = {}
        elif not isinstance(args, dict):
            try:
                args = json.loads(args)
            except (TypeError, json.JSONDecodeError):
                pass  # left as-is; the tools node reports it back to the model
        calls.append({"name": _get(fn, "name"), "arguments": args})
    return {"content": strip_think(_get(msg, "content") or ""), "tool_calls": calls}


class OllamaChat:
    """Async chat client. `host` defaults to $OLLAMA_HOST (handled by the ollama library)."""

    def __init__(self, model: str = "qwen3:8b", host: str | None = None, num_ctx: int = 8192,
                 temperature: float = 0.0, think: bool = False, client=None):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        if client is None:
            from ollama import AsyncClient

            client = AsyncClient(host=host) if host else AsyncClient()
        self._client = client

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = await self._client.chat(
            model=self.model, messages=messages, tools=tools, think=self.think,
            options={"num_ctx": self.num_ctx, "temperature": self.temperature},
        )
        return normalize_response(resp)
