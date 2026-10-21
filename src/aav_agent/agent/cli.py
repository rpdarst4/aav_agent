"""Command line: run the agent once on a prompt, or interactively.

    python -m aav_agent.agent --settings config/settings.yaml "run laava on ..."
    python -m aav_agent.agent --settings config/settings.yaml          # interactive

The model endpoint comes from --ollama-host or $OLLAMA_HOST, the model from --model or $AAV_LLM_MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from .graph import build_graph
from .llm import OllamaChat
from .mcp_client import McpToolClient
from .nodes import HIDDEN_TOOLS, compact, make_nodes
from .prompt import SYSTEM_PROMPT
from .schema import to_ollama_tools


def _status(result) -> str:
    if isinstance(result, dict):
        return str(result.get("status", "ok"))
    if isinstance(result, list):
        return f"ok ({len(result)} items)"
    return "ok"


def print_event(kind: str, payload: dict, verbose: bool = False) -> None:
    """Progress on stderr, so stdout carries only the answer. With verbose, full tool results too."""
    if kind == "call":
        print(f"[tool] {payload['name']} {payload['arguments']}", file=sys.stderr)
    elif kind == "waiting":
        print(f"[wait] {payload.get('run_id')} {payload.get('screen_id') or ''} -> {payload.get('status')} "
              f"({payload.get('waited_s', 0)} s)", file=sys.stderr)
    elif kind == "result":
        print(f"[tool] {payload['name']} -> {_status(payload['result'])}", file=sys.stderr)
        if verbose:
            print(f"[result] {compact(payload['result'], max_chars=2000)}", file=sys.stderr)
    elif kind == "assistant" and verbose and payload.get("content"):
        print(f"[assistant] {payload['content']}", file=sys.stderr)


class Transcript:
    """Append every event to a JSON-lines file, for debugging a conversation afterwards."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self, kind: str, payload: dict) -> None:
        line = json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "kind": kind, "payload": payload}, default=str, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as fh:  # opened per event: nothing to close, nothing lost
            fh.write(line + "\n")


def carry_over(history: list[dict], final: str) -> list[dict]:
    """Conversation memory between turns: user messages and final answers only (no tool traffic)."""
    kept = [m for m in history if m["role"] == "user" or (m["role"] == "assistant" and not m.get("tool_calls"))]
    return kept + [{"role": "assistant", "content": final}]


async def run(args: argparse.Namespace) -> int:
    settings = args.settings or os.environ.get("AAV_SETTINGS")
    if not settings:
        print("Pass --settings /abs/path/settings.yaml or set AAV_SETTINGS.", file=sys.stderr)
        return 2
    server_args = ["-m", "aav_agent.mcp_server", "--settings", os.path.abspath(settings)]
    async with McpToolClient(sys.executable, server_args, env=dict(os.environ)) as mcp:
        tools = to_ollama_tools(mcp.tools, hidden=HIDDEN_TOOLS)
        llm = OllamaChat(model=args.model, host=args.ollama_host, num_ctx=args.num_ctx)
        transcript = Transcript(args.transcript) if args.transcript else None

        def on_event(kind: str, payload: dict) -> None:
            print_event(kind, payload, verbose=args.verbose)
            if transcript:
                transcript(kind, payload)

        nodes = make_nodes(llm, mcp, tools, SYSTEM_PROMPT, max_steps=args.max_steps, on_event=on_event)
        app = build_graph(nodes)

        history: list[dict] = []
        prompts = [args.prompt] if args.prompt else None
        while True:
            if prompts is not None:
                if not prompts:
                    return 0
                prompt = prompts.pop(0)
            else:
                try:
                    prompt = input("you> ").strip()
                except EOFError:
                    return 0
                if not prompt or prompt in ("exit", "quit"):
                    return 0
            state = await app.ainvoke(
                {"messages": history + [{"role": "user", "content": prompt}], "steps": 0, "artifacts": {}},
                {"recursion_limit": args.max_steps * 3 + 10},
            )
            print(state["final"])
            history = carry_over(history + [{"role": "user", "content": prompt}], state["final"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Chat with the AAV analysis agent (local model + MCP tools).")
    p.add_argument("prompt", nargs="?", help="One-shot prompt; omit for an interactive session")
    p.add_argument("--settings", help="settings.yaml (default: $AAV_SETTINGS)")
    p.add_argument("--model", default=os.environ.get("AAV_LLM_MODEL", "qwen3:8b"))
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST"))
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("-v", "--verbose", action="store_true", help="Also print each tool result and the model's text")
    p.add_argument("--transcript", help="Append every call and result to this JSON-lines file")
    sys.exit(asyncio.run(run(p.parse_args(argv))))
