"""Chat with a local model through Ollama, returning plain dicts."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

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


OLLAMA_CLOUD_HOST = "https://ollama.com"


def client_kwargs(host: str | None, api_key: str | None) -> dict:
    """Arguments for ollama.AsyncClient. The API key is sent only to ollama.com, never to another host
    (for example the PC on your network), and an API key with no host means Ollama's cloud."""
    if api_key and not host:
        host = OLLAMA_CLOUD_HOST
    kwargs: dict = {}
    if host:
        kwargs["host"] = host
        if api_key and (urlparse(host if "//" in host else f"//{host}").hostname or "").endswith("ollama.com"):
            kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return kwargs


class OllamaChat:
    """Async chat client for a local Ollama server or Ollama's cloud.

    `host` defaults to $OLLAMA_HOST (handled by the ollama library). For the cloud, set $OLLAMA_API_KEY
    (or pass api_key) and use a cloud model such as gpt-oss:120b.
    """

    def __init__(self, model: str = "qwen3:8b", host: str | None = None, num_ctx: int = 8192,
                 temperature: float = 0.0, think: bool = False, client=None, api_key: str | None = None,
                 client_cls=None):
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        if client is None:
            if client_cls is None:
                from ollama import AsyncClient as client_cls
            key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY")
            client = client_cls(**client_kwargs(host, key))
        self._client = client

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        resp = await self._client.chat(
            model=self.model, messages=messages, tools=tools, think=self.think,
            options={"num_ctx": self.num_ctx, "temperature": self.temperature},
        )
        return normalize_response(resp)
