"""The agent's graph nodes as plain async functions, so they can be tested without LangGraph.

Design choices for a small local model:
  * Polling is hidden. When run_laava or screen_contaminants returns "queued", the tools node itself
    waits (via wait_for_job) and hands the model the finished result.
  * Numbers are not left to the model. `finalize` appends the run and screen results, taken straight
    from the tool outputs, after whatever the model says.
"""

from __future__ import annotations

import json
import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

HIDDEN_TOOLS = frozenset({"wait_for_job"})
ACTIVE = ("queued", "running")


class AgentState(TypedDict, total=False):
    messages: Annotated[list, operator.add]  # nodes return only the new messages
    steps: int
    artifacts: dict
    final: str


@dataclass
class Nodes:
    llm: Callable[[dict], Awaitable[dict]]
    tools: Callable[[dict], Awaitable[dict]]
    finalize: Callable[[dict], dict]
    route: Callable[[dict], str]


# Size limits for what a single tool result may put into the model's context. Roughly 4 characters per
# token. The model runs with num_ctx=8192 (see cli.py), and the tool definitions plus system prompt already
# take about 2.5k of that, so one result must stay well below the rest: 6000 characters is ~1500 tokens.
MAX_LOG_CHARS = 1200  # ~300 tokens; the end of a failed run's log, where the error is
MAX_RESULT_CHARS = 6000  # ~1500 tokens
# Successively harder structural cuts: (max list items, max string length), tried until the JSON fits.
_SHRINK_STEPS = ((20, 500), (10, 300), (5, 150), (3, 80))
# If even that does not fit, these fields are kept whole: they say what happened and where.
_ESSENTIAL_KEYS = ("status", "error", "message", "run_id", "screen_id", "log_tail")


def _trim_log(text: str, max_chars: int) -> str:
    """The end of a log, starting at a line boundary, with a marker saying lines were dropped."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    if text[-max_chars - 1] != "\n" and "\n" in tail:  # the cut landed mid-line: drop the partial line
        tail = tail.split("\n", 1)[1]
    return "[earlier lines omitted]\n" + tail


def _shrink(value: Any, max_items: int, max_str: int) -> Any:
    """Cut long lists and strings, saying what was left out, so the result is still valid JSON."""
    if isinstance(value, dict):
        # log_tail is already bounded by _trim_log and is the most useful part of a failure: keep it whole
        return {k: (v if k == "log_tail" else _shrink(v, max_items, max_str)) for k, v in value.items()}
    if isinstance(value, list):
        kept = [_shrink(v, max_items, max_str) for v in value[:max_items]]
        if len(value) > max_items:
            kept.append(f"... {len(value) - max_items} more items omitted")
        return kept
    if isinstance(value, str) and len(value) > max_str:
        return value[:max_str] + f"... ({len(value) - max_str} more characters omitted)"
    return value


def compact(result: Any, max_chars: int = MAX_RESULT_CHARS, max_log_chars: int = MAX_LOG_CHARS) -> str:
    """A tool result as JSON for the model, kept valid and under max_chars.

    Logs are cut first (from the front, at a line boundary) because their useful part is the end. If the
    JSON is still too long, lists and strings are shortened with explicit "omitted" notes; only if that
    cannot make it fit are the essential fields (status, error, log, ids) kept whole with a text preview of
    the rest.
    """
    if isinstance(result, dict) and isinstance(result.get("log_tail"), str):
        result = {**result, "log_tail": _trim_log(result["log_tail"], max_log_chars)}

    def dump(v: Any) -> str:
        return json.dumps(v, default=str, ensure_ascii=False)

    text = dump(result)
    if len(text) <= max_chars:
        return text
    for max_items, max_str in _SHRINK_STEPS:
        text = dump(_shrink(result, max_items, max_str))
        if len(text) <= max_chars:
            return text
    return _last_resort(result, max_chars)


def _last_resort(result: Any, max_chars: int) -> str:
    """Valid JSON within max_chars: the essential fields whole, plus a preview of whatever else fits."""
    if isinstance(result, dict):
        essentials = {k: result[k] for k in _ESSENTIAL_KEYS if k in result}
        rest: Any = {k: v for k, v in result.items() if k not in essentials}
    else:
        essentials, rest = {}, result
    head = {"truncated": True, **essentials}
    if len(json.dumps(head, default=str, ensure_ascii=False)) > max_chars:
        head, rest = {"truncated": True}, essentials or rest  # even the essentials are too big: preview them
    preview = json.dumps(rest, default=str, ensure_ascii=False)
    text = json.dumps({**head, "preview": preview}, default=str, ensure_ascii=False)
    while len(text) > max_chars and preview:  # escaping can lengthen the preview, so shrink until it fits
        preview = preview[: int(len(preview) * 0.8)]
        text = json.dumps({**head, "preview": preview}, default=str, ensure_ascii=False)
    return text


def _count(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def format_results(artifacts: dict) -> str:
    """Human-readable results block built from tool outputs only."""
    lines: list[str] = []
    run = artifacts.get("run")
    screen = artifacts.get("screen")
    if run:
        s = run.get("summary") or {}
        total, unmapped = s.get("total_reads"), s.get("unmapped_reads")
        head = f"LAAVA run {run.get('run_id')} (sample {run.get('sample_id')})"
        if isinstance(total, int) and isinstance(unmapped, int) and total:
            head += f": {_count(total)} reads, {_count(unmapped)} unmapped ({unmapped / total:.1%})."
        lines.append(head)
        for w in run.get("warnings") or []:
            lines.append(f"  Warning: {w}")
        for title, key in (("By reference", "by_reference_label"), ("By assigned type", "by_assigned_type")):
            counts = s.get(key) or {}
            if counts:
                lines.append(f"  {title}: " + "; ".join(f"{k} {_count(v)}" for k, v in counts.items()))
    if screen:
        buckets = screen.get("buckets") or {}
        n = sum(buckets.values())
        rec = screen.get("database_recorded") or {}
        label = ", ".join(x for x in (rec.get("db_title"), rec.get("db_date")) if x) or str(screen.get("database", ""))
        lines.append(f"Contaminant screen {screen.get('screen_id')} against {label}: {_count(n)} unmapped reads classified.")
        for bucket, count in buckets.items():
            lines.append(f"  {bucket}: {_count(count)} ({count / n:.1%})" if n else f"  {bucket}: {_count(count)}")
        v = screen.get("validation")
        if v:
            lines.append(f"  Validation on spiked reads: {v.get('correct')} of {v.get('spiked_reads')} correct "
                         f"({v.get('leaked_reads')} never reached BLAST).")
        lines.append(f"Report with contaminant section: {screen.get('report_html')}")
    elif run and (run.get("outputs") or {}).get("report_html"):
        lines.append(f"LAAVA report: {run['outputs']['report_html']}")
    return "\n".join(lines)


def make_nodes(
    llm,
    mcp,
    tools: list[dict],
    system_prompt: str,
    max_steps: int = 12,
    wait_s: float = 45.0,
    max_job_s: float = 3600.0,
    on_event: Callable[[str, dict], None] | None = None,
    max_result_chars: int = MAX_RESULT_CHARS,
    max_log_chars: int = MAX_LOG_CHARS,
) -> Nodes:
    """llm: has `async chat(messages, tools)`; mcp: has `async call(name, arguments)`; tools: Ollama tool dicts."""
    emit = on_event or (lambda kind, payload: None)
    visible = {t["function"]["name"] for t in tools}
    schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}

    def check_args(name: str, args: dict) -> str | None:
        """Reject arguments the tool does not have. The server would silently ignore them, and a model that
        sent `folder` to a tool without one would believe it had searched that folder."""
        props = set(schemas[name].get("properties") or {})
        unknown = sorted(set(args) - props)
        missing = [r for r in schemas[name].get("required") or [] if r not in args]
        problems = []
        if unknown:
            problems.append(f"unknown argument(s) {unknown}; valid arguments are {sorted(props)}")
        if missing:
            problems.append(f"missing required argument(s) {missing}")
        return f"{name}: " + "; ".join(problems) if problems else None

    async def llm_node(state: dict) -> dict:
        messages = [{"role": "system", "content": system_prompt}] + list(state["messages"])
        reply = await llm.chat(messages, tools)
        assistant: dict = {"role": "assistant", "content": reply["content"]}
        if reply["tool_calls"]:
            assistant["tool_calls"] = [
                {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in reply["tool_calls"]
            ]
        emit("assistant", assistant)
        return {"messages": [assistant], "steps": state.get("steps", 0) + 1}

    async def await_job(started: dict) -> dict:
        run_id, screen_id = started["run_id"], started.get("screen_id")
        deadline = time.monotonic() + max_job_s
        while True:
            args = {"run_id": run_id, "timeout_s": wait_s}
            if screen_id:
                args["screen_id"] = screen_id
            status = await mcp.call("wait_for_job", args)
            if not isinstance(status, dict):
                return {"status": "error", "message": f"unexpected reply while waiting: {status!r}"}
            emit("waiting", status)
            if status.get("status") not in ACTIVE:
                return status
            if time.monotonic() >= deadline:
                return {"status": "timeout", "run_id": run_id, "screen_id": screen_id,
                        "message": f"still {status['status']} after {int(max_job_s)} s; it keeps running in the "
                                   "background. Check it later with get_run."}

    async def tools_node(state: dict) -> dict:
        last = state["messages"][-1]
        artifacts = dict(state.get("artifacts") or {})
        out = []
        for call in last.get("tool_calls", []):
            name, args = call["function"]["name"], call["function"]["arguments"]
            emit("call", {"name": name, "arguments": args})
            if name not in visible:
                result: Any = {"status": "error", "message": f"unknown tool {name!r}; available: {sorted(visible)}"}
            elif not isinstance(args, dict):
                result = {"status": "error", "message": "tool arguments must be a JSON object"}
            elif (problem := check_args(name, args)) is not None:
                result = {"status": "error", "message": problem}
            else:
                try:
                    result = await mcp.call(name, args)
                    if isinstance(result, dict) and result.get("status") == "queued" and result.get("run_id"):
                        started = result
                        result = await await_job(started)
                        if started.get("warnings") and isinstance(result, dict):  # keep the start-up warnings
                            merged = list(dict.fromkeys(started["warnings"] + (result.get("warnings") or [])))
                            result = {**result, "warnings": merged}
                except Exception as e:  # a crashed server or a transport error must reach the model as text
                    result = {"status": "error", "message": f"{type(e).__name__}: {e}"}
            if isinstance(result, dict) and result.get("status") == "succeeded":
                if name == "run_laava":
                    artifacts["run"] = result
                elif name == "screen_contaminants":
                    artifacts["screen"] = result
            emit("result", {"name": name, "result": result})
            out.append({"role": "tool", "tool_name": name,
                        "content": compact(result, max_result_chars, max_log_chars)})
        return {"messages": out, "artifacts": artifacts}

    def route(state: dict) -> str:
        last = state["messages"][-1]
        return "tools" if last.get("tool_calls") and state.get("steps", 0) < max_steps else "finalize"

    def finalize_node(state: dict) -> dict:
        last = state["messages"][-1]
        text = (last.get("content") or "").strip()
        note = f"(Stopped after {max_steps} steps before finishing.)" if last.get("tool_calls") else ""
        block = format_results(state.get("artifacts") or {})
        return {"final": "\n\n".join(p for p in (text, note, block) if p) or "(no answer)"}

    return Nodes(llm=llm_node, tools=tools_node, finalize=finalize_node, route=route)
