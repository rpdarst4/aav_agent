"""A small async client for the AAV MCP server (stdio)."""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_tool_result(result: Any) -> Any:
    """The Python value a tool returned.

    Prefers the structured result; otherwise parses JSON from the text blocks (several blocks mean a
    list, one JSON value per block). Errors raised by the server become {"status": "error", ...}.
    """
    blocks = _get(result, "content") or []
    texts = [_get(b, "text") for b in blocks if _get(b, "text") is not None]
    if _get(result, "isError", False):
        return {"status": "error", "message": "\n".join(texts) or "the tool raised an error"}

    structured = _get(result, "structuredContent")
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:  # wrapper for non-dict returns
            return structured["result"]
        return structured

    parsed = []
    for t in texts:
        try:
            parsed.append(json.loads(t))
        except json.JSONDecodeError:
            parsed.append(t)
    if not parsed:
        return None
    return parsed[0] if len(parsed) == 1 else parsed


class McpToolClient:
    """Starts the MCP server as a subprocess and calls its tools.

    The server is given its settings explicitly (--settings) and the full environment, because MCP
    launchers pass only a small safe subset of the environment by default.
    """

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None, cwd: str | None = None):
        self._params = dict(command=command, args=args, env=env, cwd=cwd)
        self._stack: AsyncExitStack | None = None
        self._session = None
        self.tools: list = []

    async def __aenter__(self) -> McpToolClient:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(
                stdio_client(StdioServerParameters(**self._params))
            )
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            initialize = getattr(self._session, "initialize", None)  # absent or optional in newer protocol revisions
            if initialize is not None:
                await initialize()
            self.tools = list((await self._session.list_tools()).tools)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def call(self, name: str, arguments: dict) -> Any:
        return parse_tool_result(await self._session.call_tool(name, arguments))
