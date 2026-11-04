"""Offline tests for the agent. A scripted fake model and a fake MCP client stand in for Ollama and the
server, and the LangGraph wiring is checked through an injected fake of its API, so none of `ollama`,
`mcp` or `langgraph` needs to be installed to run these. Real wiring is covered by trying the agent."""

import asyncio
import copy
import inspect
import json
from types import SimpleNamespace

import pytest

from aav_agent.agent import cli
from aav_agent.agent.graph import build_graph
from aav_agent.agent.llm import OllamaChat, normalize_response, strip_think
from aav_agent.agent.mcp_client import parse_tool_result
from aav_agent.agent.nodes import HIDDEN_TOOLS, compact, format_results, make_nodes
from aav_agent.agent.prompt import SYSTEM_PROMPT
from aav_agent.agent.schema import simplify_schema, to_ollama_tools
from aav_agent.mcp_server.server import make_tools

START, END = "__start__", "__end__"


# ---------------------------------------------------------------- fakes


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def chat(self, messages, tools):
        self.seen.append((copy.deepcopy(messages), tools))
        return self.replies.pop(0)


class FakeMcp:
    """handlers: tool name -> callable(arguments) or a list of results returned in order."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        h = self.handlers[name]
        return h(arguments) if callable(h) else h.pop(0)


class FakeStateGraph:
    """Records the wiring and runs it the way LangGraph would (lists in `messages` accumulate)."""

    def __init__(self, state_cls):
        self.nodes, self.edges, self.cond = {}, {}, {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, src, route, mapping):
        self.cond[src] = (route, mapping)

    def compile(self):
        g = self

        class App:
            async def ainvoke(self, state, config=None):
                state = dict(state)
                cur = g.edges[START]
                for _ in range(200):
                    if cur == END:
                        return state
                    update = g.nodes[cur](state)
                    if asyncio.iscoroutine(update):
                        update = await update
                    for k, v in update.items():
                        state[k] = state.get(k, []) + v if k == "messages" else v
                    cur = g.cond[cur][1][g.cond[cur][0](state)] if cur in g.cond else g.edges[cur]
                raise RuntimeError("graph did not terminate")

        return App()


FAKE_API = SimpleNamespace(StateGraph=FakeStateGraph, START=START, END=END)

def _tool_definitions():
    """Tool schemas built from the real MCP tool signatures, so agent tests use the real argument names."""
    out = []
    for name, fn in make_tools(object()).items():
        if name in HIDDEN_TOOLS:
            continue
        props, required = {}, []
        for pname, param in inspect.signature(fn).parameters.items():
            ann = str(param.annotation)
            kind = "number" if "float" in ann else "integer" if "int" in ann else "boolean" if "bool" in ann else "string"
            props[pname] = {"type": kind}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        out.append({"type": "function", "function": {
            "name": name, "description": fn.__doc__ or "",
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return out


TOOLS = _tool_definitions()

RUN_OK = {
    "status": "succeeded", "run_id": "run-1", "sample_id": "mcp.test",
    "outputs": {"report_html": "/data/runs/run-1/output/report/mcp.test_AAV_report.html"},
    "summary": {"total_reads": 3627, "unmapped_reads": 130,
                "by_reference_label": {"vector": 3414, "(unmapped)": 130},
                "by_assigned_type": {"ssAAV": 2032, "NA": 213}},
}
SCREEN_OK = {
    "status": "succeeded", "run_id": "run-1", "screen_id": "screen-001", "database": "demo",
    "buckets": {"soil bacteria": 50, "insect": 24, "enterobacteria": 22, "no hit": 14, "plant": 10, "baculovirus": 10},
    "report_html": "/data/runs/run-1/screens/screen-001/mcp.test/report/mcp.test_AAV_report.contaminants.html",
    "database_recorded": {"db_title": "AAV demo contaminants", "db_date": "Oct 7, 2026"},
}


def call(name, **args):
    return {"content": "", "tool_calls": [{"name": name, "arguments": args}]}


def say(text):
    return {"content": text, "tool_calls": []}


def run_agent(llm, mcp, max_steps=12, events=None, prompt="go"):
    nodes = make_nodes(llm, mcp, TOOLS, "SYSTEM", max_steps=max_steps, wait_s=0.01, max_job_s=5,
                       on_event=(lambda k, p: events.append(k)) if events is not None else None)
    app = build_graph(nodes, api=FAKE_API)
    return asyncio.run(app.ainvoke({"messages": [{"role": "user", "content": prompt}], "steps": 0, "artifacts": {}}))


# ---------------------------------------------------------------- schema


def test_simplify_schema_collapses_optional_unions_and_keeps_property_names():
    schema = {
        "title": "run_laavaArguments", "type": "object", "required": ["reads"],
        "properties": {
            "reads": {"title": "Reads", "type": "string"},
            "host_fa": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Host Fa"},
            "min_cov": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": None},
            "mode": {"type": ["string", "null"]},
            "title": {"type": "string", "title": "Title"},  # a property literally named "title" must survive
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
            "n": {"type": "integer", "default": 50},
        },
    }
    out = simplify_schema(schema)
    props = out["properties"]
    assert "title" not in out and out["required"] == ["reads"]
    assert props["host_fa"] == {"type": "string"} and props["min_cov"] == {"type": "number"}
    assert props["mode"] == {"type": "string"} and "title" in props and props["title"] == {"type": "string"}
    assert props["either"]["anyOf"] == [{"type": "string"}, {"type": "integer"}]
    assert props["n"]["default"] == 50
    assert "anyOf" in schema["properties"]["host_fa"]  # the input was not modified


def test_to_ollama_tools_hides_polling_and_accepts_objects_or_dicts():
    mcp_tools = [
        SimpleNamespace(name="run_laava", description=" Start a run. ", inputSchema={"type": "object", "properties": {}}),
        {"name": "wait_for_job", "description": "poll", "inputSchema": {"type": "object"}},
        {"name": "get_run", "description": "get", "inputSchema": {"type": "object", "properties": {"run_id": {"type": "string"}}}},
    ]
    tools = to_ollama_tools(mcp_tools, hidden=HIDDEN_TOOLS)
    assert [t["function"]["name"] for t in tools] == ["run_laava", "get_run"]
    assert tools[0]["type"] == "function" and tools[0]["function"]["description"] == "Start a run."


# ---------------------------------------------------------------- MCP result parsing


def test_parse_tool_result_variants():
    text = lambda s: SimpleNamespace(text=s)  # noqa: E731
    assert parse_tool_result(SimpleNamespace(isError=False, content=[], structuredContent={"result": [1, 2]})) == [1, 2]
    assert parse_tool_result(SimpleNamespace(isError=False, content=[], structuredContent={"status": "ok"})) == {"status": "ok"}
    assert parse_tool_result(SimpleNamespace(isError=False, content=[text('{"a": 1}')], structuredContent=None)) == {"a": 1}
    assert parse_tool_result(SimpleNamespace(isError=False, content=[text('{"a": 1}'), text('{"a": 2}')],
                                             structuredContent=None)) == [{"a": 1}, {"a": 2}]
    assert parse_tool_result(SimpleNamespace(isError=False, content=[text("plain words")], structuredContent=None)) == "plain words"
    err = parse_tool_result(SimpleNamespace(isError=True, content=[text("boom")], structuredContent=None))
    assert err == {"status": "error", "message": "boom"}
    assert parse_tool_result({"content": [{"text": '{"x": 1}'}], "isError": False}) == {"x": 1}  # dict form too
    assert parse_tool_result(SimpleNamespace(isError=False, content=[], structuredContent=None)) is None


# ---------------------------------------------------------------- LLM wrapper


def test_normalize_response_handles_objects_dicts_json_strings_and_think_blocks():
    obj = SimpleNamespace(message=SimpleNamespace(
        content="<think>hmm</think>Hello", tool_calls=[
            SimpleNamespace(function=SimpleNamespace(name="get_run", arguments={"run_id": "r1"})),
            SimpleNamespace(function=SimpleNamespace(name="list_runs", arguments='{"limit": 3}')),
            SimpleNamespace(function=SimpleNamespace(name="list_databases", arguments=None)),
        ]))
    out = normalize_response(obj)
    assert out["content"] == "Hello"
    assert out["tool_calls"] == [{"name": "get_run", "arguments": {"run_id": "r1"}},
                                 {"name": "list_runs", "arguments": {"limit": 3}},
                                 {"name": "list_databases", "arguments": {}}]
    as_dict = {"message": {"content": "x", "tool_calls": [{"function": {"name": "n", "arguments": "not json"}}]}}
    assert normalize_response(as_dict)["tool_calls"][0]["arguments"] == "not json"  # left for the tools node to reject
    assert normalize_response({"message": {"content": None}}) == {"content": "", "tool_calls": []}
    assert strip_think("<think>a\nb</think>\n\nanswer") == "answer"


def test_ollama_chat_sends_context_temperature_and_disables_thinking():
    sent = {}

    class Client:
        async def chat(self, **kw):
            sent.update(kw)
            return {"message": {"content": "ok"}}

    llm = OllamaChat(model="qwen3:8b", num_ctx=4096, client=Client())
    out = asyncio.run(llm.chat([{"role": "user", "content": "hi"}], TOOLS))
    assert out == {"content": "ok", "tool_calls": []}
    assert sent["model"] == "qwen3:8b" and sent["think"] is False and sent["tools"] == TOOLS
    assert sent["options"] == {"num_ctx": 4096, "temperature": 0.0}


# ---------------------------------------------------------------- nodes and the whole loop


def test_full_conversation_hides_polling_and_reports_exact_numbers():
    llm = FakeLLM([
        call("list_input_files", pattern="*.fastq.gz"),
        call("run_laava", reads="/fx/reads.fastq.gz", vector_fa="/fx/v.fa", vector_bed="/fx/v.bed", vector_type="ss"),
        call("screen_contaminants", run_id="run-1"),
        say("I analysed the reads and screened the unmapped ones."),
    ])
    mcp = FakeMcp({
        "list_input_files": lambda a: {"files": ["/fx/reads.fastq.gz"], "truncated": False},
        "run_laava": lambda a: {"status": "queued", "run_id": "run-1", "sample_id": "mcp.test"},
        "screen_contaminants": lambda a: {"status": "queued", "run_id": "run-1", "screen_id": "screen-001"},
        "wait_for_job": [
            {"status": "running", "run_id": "run-1", "waited_s": 0.0},
            RUN_OK,
            SCREEN_OK,
        ],
    })
    events = []
    state = run_agent(llm, mcp, events=events)

    assert [c[0] for c in mcp.calls] == ["list_input_files", "run_laava", "wait_for_job", "wait_for_job",
                                         "screen_contaminants", "wait_for_job"]
    assert mcp.calls[2][1] == {"run_id": "run-1", "timeout_s": 0.01}  # run: no screen_id
    assert mcp.calls[5][1] == {"run_id": "run-1", "timeout_s": 0.01, "screen_id": "screen-001"}
    # the model never sees "queued": polling happened inside the tools node
    tool_messages = [m["content"] for seen, _ in llm.seen for m in seen if m["role"] == "tool"]
    assert tool_messages and not any('"queued"' in t for t in tool_messages)
    assert state["artifacts"]["run"] == RUN_OK and state["artifacts"]["screen"] == SCREEN_OK

    final = state["final"]
    assert final.startswith("I analysed the reads")
    for expected in ("3,627 reads, 130 unmapped (3.6%)", "soil bacteria: 50 (38.5%)", "no hit: 14 (10.8%)",
                     "AAV demo contaminants, Oct 7, 2026", SCREEN_OK["report_html"], "LAAVA run run-1"):
        assert expected in final, expected
    assert state["steps"] == 4
    assert {"assistant", "call", "waiting", "result"} <= set(events)


def test_model_can_recover_from_invalid_arguments():
    llm = FakeLLM([
        call("run_laava", reads="/nope.fastq", vector_fa="/fx/v.fa", vector_bed="/fx/v.bed"),
        call("run_laava", reads="/fx/reads.fastq.gz", vector_fa="/fx/v.fa", vector_bed="/fx/v.bed"),
        say("Done."),
    ])
    mcp = FakeMcp({
        "run_laava": [{"status": "invalid", "errors": ["reads: file not found: /nope.fastq"], "warnings": []},
                      {"status": "queued", "run_id": "run-1"}],
        "wait_for_job": [RUN_OK],
    })
    state = run_agent(llm, mcp)
    second_turn_messages, _ = llm.seen[1]
    assert any("file not found" in m["content"] for m in second_turn_messages if m["role"] == "tool")
    assert state["artifacts"] == {"run": RUN_OK}
    assert "LAAVA report: /data/runs/run-1/output/report/mcp.test_AAV_report.html" in state["final"]  # no screen yet


def test_bad_calls_come_back_as_errors_not_crashes():
    llm = FakeLLM([
        {"content": "", "tool_calls": [
            {"name": "delete_everything", "arguments": {}},
            {"name": "get_run", "arguments": "not json"},
            {"name": "list_runs", "arguments": {}},
        ]},
        say("Sorry about that."),
    ])

    def boom(args):
        raise ConnectionError("server went away")

    mcp = FakeMcp({"list_runs": boom})
    state = run_agent(llm, mcp)
    tool_msgs = {m["tool_name"]: m["content"] for m in state["messages"] if m["role"] == "tool"}
    assert "unknown tool" in tool_msgs["delete_everything"] and "list_databases" in tool_msgs["delete_everything"]
    assert "must be a JSON object" in tool_msgs["get_run"]
    assert "ConnectionError: server went away" in tool_msgs["list_runs"]
    assert mcp.calls == [("list_runs", {})]  # unknown tool and malformed arguments never reached the server


def test_polling_gives_up_after_the_time_limit_and_says_so():
    llm = FakeLLM([call("run_laava", reads="r", vector_fa="v", vector_bed="b"), say("It is still running.")])
    mcp = FakeMcp({
        "run_laava": lambda a: {"status": "queued", "run_id": "run-9"},
        "wait_for_job": lambda a: {"status": "running", "run_id": "run-9"},
    })
    nodes = make_nodes(llm, mcp, TOOLS, "SYSTEM", wait_s=0.01, max_job_s=0.05)
    app = build_graph(nodes, api=FAKE_API)
    state = asyncio.run(app.ainvoke({"messages": [{"role": "user", "content": "go"}], "steps": 0, "artifacts": {}}))
    result = json.loads(next(m["content"] for m in state["messages"] if m["role"] == "tool"))
    assert result["status"] == "timeout" and "get_run" in result["message"] and "run" not in state["artifacts"]


def test_step_limit_stops_a_looping_model():
    llm = FakeLLM([call("list_runs")] * 10)
    mcp = FakeMcp({"list_runs": lambda a: []})
    state = run_agent(llm, mcp, max_steps=3)
    assert state["steps"] == 3 and len(mcp.calls) == 2
    assert "Stopped after 3 steps" in state["final"]


def test_a_plain_answer_needs_no_tools():
    state = run_agent(FakeLLM([say("Which fastq file did you mean?")]), FakeMcp({}))
    assert state["final"] == "Which fastq file did you mean?" and state["steps"] == 1


def test_the_system_prompt_is_sent_each_turn_and_polling_tool_is_not_offered():
    llm = FakeLLM([say("hi")])
    run_agent(llm, FakeMcp({}))
    messages, tools = llm.seen[0]
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert "wait_for_job" not in {t["function"]["name"] for t in tools}


def test_compact_logs_are_cut_at_a_line_boundary_and_say_so():
    lines = [f"line number {i:04d} of the log" for i in range(500)]
    out = json.loads(compact({"status": "failed", "log_tail": "\n".join(lines)}, max_log_chars=1200))
    tail = out["log_tail"]
    assert tail.startswith("[earlier lines omitted]\n") and len(tail) <= 1200 + 30
    kept = tail.split("\n")[1:]
    assert kept and all(line in lines for line in kept)  # no line is cut in half
    assert kept[-1] == lines[-1]  # the end of the log, where the error is, survives
    # a short log is left alone, and so is one that is exactly at the limit
    assert json.loads(compact({"log_tail": "short"}))["log_tail"] == "short"
    exact = "x" * 1200
    assert json.loads(compact({"log_tail": exact}, max_log_chars=1200))["log_tail"] == exact
    # one enormous line cannot be split at a boundary: keep its end
    huge = json.loads(compact({"log_tail": "y" * 5000}, max_log_chars=1200))["log_tail"]
    assert huge.startswith("[earlier lines omitted]") and huge.endswith("y" * 1200)


def test_compact_shrinks_lists_structurally_and_stays_valid_json():
    files = [f"/data/some/long/folder/name/sample_{i:03d}.fastq.gz" for i in range(200)]
    text = compact({"files": files, "truncated": False}, max_chars=1500)
    out = json.loads(text)  # still valid JSON, unlike slicing the string
    assert len(text) <= 1500
    assert out["files"][0] == files[0] and out["files"][1] == files[1]  # the head is intact
    assert out["files"][-1].endswith("more items omitted")
    kept = len(out["files"]) - 1
    assert out["files"][-1] == f"... {200 - kept} more items omitted"  # the count of what was dropped is exact
    assert compact(files[:5]) == json.dumps(files[:5])  # a result that fits is passed through unchanged


def test_compact_handles_a_file_listing_with_descriptions():
    files = [{"path": f"/data/some/long/folder/name/sample_{i:03d}.fa",
              "what": "FASTA, 2 records (chr19, chrM), 24.6 Mb; looks like a genome"} for i in range(60)]
    text = compact({"status": "ok", "files": files, "truncated": False}, max_chars=3000)
    out = json.loads(text)
    assert len(text) <= 3000 and out["status"] == "ok"
    assert out["files"][0] == files[0] and out["files"][-1].endswith("more items omitted")  # entries stay whole


def test_compact_keeps_the_failure_log_even_when_other_fields_are_huge():
    big = {f"label_{i}": "z" * 400 for i in range(40)}
    result = {"status": "failed", "error": "nextflow exited with code 1", "log_tail": "ERROR: missing file",
              "summary": big}
    # moderately too big: structural shrinking is enough, and the log is untouched
    out = json.loads(compact(result, max_chars=8000))
    assert out["log_tail"] == "ERROR: missing file" and out["error"] == "nextflow exited with code 1"
    assert all(len(v) < 400 for v in out["summary"].values()) and "truncated" not in out
    # far too big: the essentials are kept whole and the rest becomes a text preview
    text = compact(result, max_chars=2000)
    out = json.loads(text)
    assert len(text) <= 2000 and out["truncated"] is True and out["status"] == "failed"
    assert out["log_tail"] == "ERROR: missing file" and out["error"] == "nextflow exited with code 1"
    assert isinstance(out["preview"], str) and "label_0" in out["preview"] and "summary" not in out


def test_compact_last_resort_is_valid_json_within_the_limit_even_for_awkward_input():
    for limit in (50, 120, 400):
        text = compact({"a": 'q"' * 100, "b": list(range(1000))}, max_chars=limit)
        parsed = json.loads(text)  # valid JSON whichever path produced it
        assert len(text) <= limit
        assert parsed.get("truncated") is True or limit == 400  # at 400 the structural cuts are enough
    text = compact({"status": "failed", "log_tail": "w" * 100}, max_chars=120, max_log_chars=100)
    assert len(text) <= 120 and json.loads(text)["truncated"] is True  # essentials alone are too big: previewed
    assert json.loads(compact("a plain string " * 100, max_chars=100))["truncated"] is True


def test_make_nodes_passes_the_limits_to_tool_messages():
    llm = FakeLLM([call("list_runs"), say("ok")])
    mcp = FakeMcp({"list_runs": lambda a: [f"run-{i:04d}" for i in range(500)]})
    nodes = make_nodes(llm, mcp, TOOLS, "SYSTEM", max_result_chars=400)
    app = build_graph(nodes, api=FAKE_API)
    state = asyncio.run(app.ainvoke({"messages": [{"role": "user", "content": "go"}], "steps": 0, "artifacts": {}}))
    content = next(m["content"] for m in state["messages"] if m["role"] == "tool")
    assert len(content) <= 400 and "more items omitted" in content and isinstance(json.loads(content), list)


def test_format_results_handles_missing_pieces():
    assert format_results({}) == ""
    only_screen = format_results({"screen": {"screen_id": "s1", "buckets": {}, "database": "demo", "report_html": "/r"}})
    assert "Contaminant screen s1 against demo" in only_screen and "/r" in only_screen
    with_validation = format_results({"screen": {**SCREEN_OK, "validation": {"correct": 109, "spiked_reads": 110, "leaked_reads": 0}}})
    assert "109 of 110 correct" in with_validation


# ---------------------------------------------------------------- the prompt matches the real tools, and CLI helpers


def test_the_prompt_only_mentions_tools_that_exist_and_does_not_mention_the_hidden_one():
    real = set(make_tools(object()))
    mentioned = {t for t in real | {"wait_for_job"} if t in SYSTEM_PROMPT}
    assert mentioned <= real and "wait_for_job" not in SYSTEM_PROMPT
    assert {"list_input_files", "describe_input", "run_laava", "screen_contaminants"} <= mentioned


def test_history_carry_over_keeps_only_user_and_final_messages():
    history = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_laava", "arguments": {}}}]},
        {"role": "tool", "tool_name": "run_laava", "content": "{}"},
        {"role": "assistant", "content": "Done."},
    ]
    kept = cli.carry_over(history, "FINAL")
    assert kept == [{"role": "user", "content": "run it"}, {"role": "assistant", "content": "Done."},
                    {"role": "assistant", "content": "FINAL"}]


def test_progress_events_print_to_stderr_not_stdout(capsys):
    cli.print_event("call", {"name": "run_laava", "arguments": {"reads": "x"}})
    cli.print_event("waiting", {"run_id": "r", "status": "running", "waited_s": 3})
    cli.print_event("result", {"name": "run_laava", "result": {"status": "succeeded"}})
    cli.print_event("assistant", {})  # unknown kinds are ignored
    out = capsys.readouterr()
    assert out.out == "" and "[tool] run_laava" in out.err and "[wait] r" in out.err


def test_results_without_a_status_field_print_as_ok_not_none(capsys):
    cli.print_event("result", {"name": "list_input_files", "result": {"files": ["a"], "truncated": False}})
    cli.print_event("result", {"name": "list_runs", "result": [1, 2]})
    cli.print_event("result", {"name": "get_run", "result": {"status": "error", "message": "no such run"}})
    err = capsys.readouterr().err
    assert "list_input_files -> ok" in err and "list_runs -> ok (2 items)" in err and "get_run -> error" in err
    assert "None" not in err and "[result]" not in err  # full results only with --verbose


def test_verbose_shows_results_and_model_text(capsys):
    cli.print_event("result", {"name": "run_laava", "result": {"status": "invalid", "errors": ["reads: file not found"]}},
                    verbose=True)
    cli.print_event("assistant", {"content": "Let me look for the files."}, verbose=True)
    err = capsys.readouterr().err
    assert "[result]" in err and "reads: file not found" in err and "[assistant] Let me look" in err


def test_transcript_appends_one_json_line_per_event(tmp_path):
    path = tmp_path / "t" / "chat.jsonl"
    path.parent.mkdir()
    transcript = cli.Transcript(str(path))
    transcript("call", {"name": "get_run", "arguments": {"run_id": "r1"}})
    transcript("result", {"name": "get_run", "result": {"path": tmp_path}})  # non-JSON values are stringified
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["call", "result"] and rows[0]["payload"]["arguments"] == {"run_id": "r1"}
    assert rows[1]["time"].endswith("+00:00") and str(tmp_path) in rows[1]["payload"]["result"]["path"]


# ---------------------------------------------------------------- argument checking and warnings


def test_arguments_the_tool_does_not_have_are_rejected_before_reaching_the_server():
    llm = FakeLLM([
        call("get_run", run_id="r1", verbose=True, folder="/somewhere"),  # invented arguments
        call("get_run"),  # missing the required run_id
        call("get_run", run_id="r1"),
        say("ok"),
    ])
    mcp = FakeMcp({"get_run": lambda a: {"status": "succeeded", "run_id": a["run_id"]}})
    state = run_agent(llm, mcp)
    msgs = [json.loads(m["content"]) for m in state["messages"] if m["role"] == "tool"]
    assert "unknown argument(s) ['folder', 'verbose']" in msgs[0]["message"] and "run_id" in msgs[0]["message"]
    assert "missing required argument(s) ['run_id']" in msgs[1]["message"]
    assert msgs[2]["status"] == "succeeded"
    assert mcp.calls == [("get_run", {"run_id": "r1"})]  # only the valid call was made


def test_start_up_warnings_reach_the_model_and_the_final_answer():
    llm = FakeLLM([call("run_laava", reads="r", vector_fa="v", vector_bed="b", itr_label_1="ITR1"), say("Done.")])
    warning = "itr_label_1='ITR1' is not a feature name in ss.annotation.bed (names: ITR)"
    mcp = FakeMcp({
        "run_laava": lambda a: {"status": "queued", "run_id": "run-1", "warnings": [warning]},
        "wait_for_job": [RUN_OK],  # the finished status does not repeat the warnings
    })
    state = run_agent(llm, mcp)
    seen_by_model = json.loads(next(m["content"] for m in state["messages"] if m["role"] == "tool"))
    assert seen_by_model["warnings"] == [warning]
    assert f"Warning: {warning}" in state["final"]


def test_warnings_from_start_and_finish_are_merged_without_duplicates():
    llm = FakeLLM([call("run_laava", reads="r", vector_fa="v", vector_bed="b"), say("Done.")])
    start_warning, data_warning = "a start-up warning", "98% of reads were assigned other-vector"
    mcp = FakeMcp({
        "run_laava": lambda a: {"status": "queued", "run_id": "run-1", "warnings": [start_warning]},
        "wait_for_job": [{**RUN_OK, "warnings": [start_warning, data_warning]}],  # the record holds both
    })
    state = run_agent(llm, mcp)
    seen = json.loads(next(m["content"] for m in state["messages"] if m["role"] == "tool"))
    assert seen["warnings"] == [start_warning, data_warning]
    assert state["final"].count(start_warning) == 1 and f"Warning: {data_warning}" in state["final"]


# ---------------------------------------------------------------- local vs. cloud Ollama client setup


def test_client_kwargs_send_the_api_key_only_to_ollama_dot_com():
    from aav_agent.agent.llm import client_kwargs

    assert client_kwargs(None, None) == {}  # default local server
    assert client_kwargs("http://192.168.1.20:11434", None) == {"host": "http://192.168.1.20:11434"}
    # a key with no host means Ollama's cloud
    assert client_kwargs(None, "k") == {"host": "https://ollama.com", "headers": {"Authorization": "Bearer k"}}
    assert client_kwargs("https://ollama.com", "k")["headers"] == {"Authorization": "Bearer k"}
    assert client_kwargs("ollama.com", "k")["headers"] == {"Authorization": "Bearer k"}  # no scheme
    # the key must never be sent to the PC on your network, or to a look-alike host
    assert "headers" not in client_kwargs("http://192.168.1.20:11434", "k")
    assert "headers" not in client_kwargs("https://ollama.com.evil.example", "k")
    assert "headers" not in client_kwargs("https://notollama.com.example", "k")


def test_ollama_chat_builds_its_client_from_host_and_key(monkeypatch):
    import os

    made = []

    class FakeClient:
        def __init__(self, **kwargs):
            made.append(kwargs)

    OllamaChat(host="http://pc:11434", client_cls=FakeClient, api_key="")
    OllamaChat(model="gpt-oss:120b", client_cls=FakeClient, api_key="secret")
    monkeypatch.setitem(os.environ, "OLLAMA_API_KEY", "from-env")  # used when no key is passed
    OllamaChat(client_cls=FakeClient)
    assert made[0] == {"host": "http://pc:11434"}
    assert made[1] == {"host": "https://ollama.com", "headers": {"Authorization": "Bearer secret"}}
    assert made[2]["headers"] == {"Authorization": "Bearer from-env"}
