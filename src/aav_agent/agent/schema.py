"""Turn MCP tool definitions into the tool format Ollama expects, with simpler JSON schemas.

Small models handle flat schemas better than the `anyOf: [X, null]` unions that Pydantic produces
for optional arguments, so those are collapsed to plain X; titles and null defaults are dropped.
"""

from __future__ import annotations

from typing import Any


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def simplify_schema(schema: dict) -> dict:
    node = dict(schema)
    node.pop("title", None)

    if "anyOf" in node:
        options = [o for o in node["anyOf"] if not (isinstance(o, dict) and o.get("type") == "null")]
        if len(options) == 1:
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            node = {**rest, **simplify_schema(options[0])}
        else:
            node["anyOf"] = [simplify_schema(o) for o in options]

    if isinstance(node.get("type"), list):
        kinds = [t for t in node["type"] if t != "null"]
        node["type"] = kinds[0] if len(kinds) == 1 else kinds

    if node.get("default", 0) is None:
        node.pop("default")

    if isinstance(node.get("properties"), dict):  # property *names* must not be treated as keywords
        node["properties"] = {name: simplify_schema(sub) for name, sub in node["properties"].items()}
    if isinstance(node.get("items"), dict):
        node["items"] = simplify_schema(node["items"])
    return node


def to_ollama_tools(mcp_tools, hidden: frozenset[str] | set[str] = frozenset()) -> list[dict]:
    """MCP tool objects (or dicts) to Ollama function-tool dicts, skipping `hidden` tool names."""
    out = []
    for tool in mcp_tools:
        name = _get(tool, "name")
        if name in hidden:
            continue
        schema = _get(tool, "inputSchema") or _get(tool, "input_schema") or {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (_get(tool, "description") or "").strip(),
                "parameters": simplify_schema(schema),
            },
        })
    return out
