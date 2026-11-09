"""Tests for the MCP layer. The official SDK is replaced by a tiny fake, so these run without `mcp`
installed; they check the tool surface (names, signatures, descriptions) and that arguments reach the
service correctly. Wiring to the real SDK is covered by trying the server in the MCP inspector."""

import asyncio
import inspect
import json
import sys
import types
from pathlib import Path

import pytest
from test_service import GOLDEN_UNMAPPED, FakeRunner, env, spec_dict, store  # noqa: F401  (fixtures)

from aav_agent.lims.store import FileRunStore
from aav_agent.mcp_server import server as mcp_server

EXPECTED_TOOLS = {"list_databases", "list_input_files", "describe_input", "run_laava", "screen_contaminants",
                  "wait_for_job", "get_run", "list_runs"}


class FakeFastMCP:
    def __init__(self, name, instructions=None):
        self.name, self.instructions, self.registered, self.ran = name, instructions, {}, None

    def tool(self, name=None, description=None):
        def deco(fn):
            self.registered[name] = {"fn": fn, "description": description}
            return fn
        return deco

    def run(self, transport=None):
        self.ran = transport


def paths_of(listing):
    return [f["path"] for f in listing["files"]]


def call(tool, **kwargs):
    return asyncio.run(tool(**kwargs))


def test_tool_surface_is_complete_and_well_described(env):
    tools = mcp_server.make_tools(env.svc)
    assert set(tools) == EXPECTED_TOOLS
    for name, fn in tools.items():
        assert inspect.iscoroutinefunction(fn), name
        assert fn.__doc__ and len(fn.__doc__) > 80, f"{name} needs a description the model can use"
        sig = inspect.signature(fn)
        assert all(p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in sig.parameters.values()), name
        assert all(p.annotation is not inspect.Parameter.empty for p in sig.parameters.values()), name
        assert sig.return_annotation is not inspect.Signature.empty, name
    required = {n for n, p in inspect.signature(tools["run_laava"]).parameters.items() if p.default is p.empty}
    assert required == {"reads", "vector_fa", "vector_bed"}


def test_server_registers_every_tool_with_instructions(env):
    fake = mcp_server.create_server(env.svc, server_cls=FakeFastMCP)
    assert fake.name == "aav-agent" and "wait_for_job" in fake.instructions
    assert set(fake.registered) == EXPECTED_TOOLS
    assert all(r["description"] for r in fake.registered.values())


def test_run_laava_tool_starts_a_job_and_wait_for_job_finishes_it(env):
    tools = mcp_server.make_tools(env.svc)
    started = call(tools["run_laava"], reads=str(env.fx / "ss.spikein.fastq.gz"),
                   vector_fa=str(env.fx / "vector.fa"), vector_bed=str(env.fx / "vector.bed"),
                   vector_type="ss", packaging_fa=str(env.fx / "packaging.fa"), host_fa=str(env.fx / "host.fa"))
    assert started["status"] == "queued"
    done = call(tools["wait_for_job"], run_id=started["run_id"], timeout_s=10)
    assert done["status"] == "succeeded" and done["summary"]["unmapped_reads"] == GOLDEN_UNMAPPED
    # unset optional arguments must not reach LAAVA as None
    params = json.loads((env.settings.runs_dir / started["run_id"] / "params.json").read_text())
    assert params["packaging_fa"].endswith("packaging.fa") and None not in params.values()
    # the packaging records keep their own names: nothing is passed for repcap/helper/lambda
    assert params["repcap_name"] == params["helper_name"] == params["lambda_name"] == ""

    screened = call(tools["screen_contaminants"], run_id=started["run_id"])
    assert screened["status"] == "queued" and screened["screen_id"] == "screen-001"
    result = call(tools["wait_for_job"], run_id=started["run_id"], screen_id="screen-001", timeout_s=10)
    assert result["status"] == "succeeded" and result["buckets"] == {"enterobacteria": 1}
    assert call(tools["get_run"], run_id=started["run_id"])["screens"][0]["status"] == "succeeded"
    assert call(tools["list_runs"])[0]["run_id"] == started["run_id"]


def test_bad_input_comes_back_as_readable_data_not_an_exception(env):
    tools = mcp_server.make_tools(env.svc)
    res = call(tools["run_laava"], reads="/etc/passwd", vector_fa=str(env.fx / "vector.fa"),
               vector_bed=str(env.fx / "vector.bed"), vector_type="triple")
    assert res["status"] == "invalid"
    assert any("outside the folders" in e for e in res["errors"]) and any("vector_type" in e for e in res["errors"])
    assert call(tools["screen_contaminants"], run_id="nope")["status"] == "error"


def test_wait_for_job_caps_the_timeout(env, monkeypatch):
    seen = {}
    monkeypatch.setattr(env.svc, "wait_for_job", lambda rid, sid, timeout: seen.update(timeout=timeout) or {"status": "x"})
    tools = mcp_server.make_tools(env.svc)
    call(tools["wait_for_job"], run_id="r", timeout_s=9999)
    assert seen["timeout"] == mcp_server.MAX_WAIT_S
    call(tools["wait_for_job"], run_id="r", timeout_s=-5)
    assert seen["timeout"] == 0.0


def test_list_databases_and_input_files_tools(env):
    tools = mcp_server.make_tools(env.svc)
    assert {d["name"] for d in call(tools["list_databases"])} == {"demo", "nt"}
    files = call(tools["list_input_files"], pattern="*.fa")
    assert any(f["path"].endswith("vector.fa") for f in files["files"])


def test_build_service_from_environment(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        f"data_dir: {tmp_path / 'data'}\nlaava_dir: {tmp_path / 'laava'}\n"
        f"screen_pipeline: {tmp_path / 'p.nf'}\ntaxdump_dir: {tmp_path / 'tax'}\ncategories: {tmp_path / 'c.yaml'}\n"
        "default_database: demo\ndatabases:\n  demo: {dir: " + str(tmp_path / "db") + ", db_name: demo}\n"
    )
    # a run left "running" by a previous server process
    old = FileRunStore(tmp_path / "data" / "lims")
    old.create_run({"_id": "stale", "type": "laava", "status": "running", "created_utc": "2026-01-01T00:00:00Z"})
    svc = mcp_server.build_service({"AAV_SETTINGS": str(cfg)})
    assert isinstance(svc.store, FileRunStore)
    assert svc.store.get_run("stale")["status"] == "failed"
    with pytest.raises(SystemExit) as e:
        mcp_server.build_service({})
    assert "--settings" in str(e.value) and "-e flag" in str(e.value)  # tells the user how to fix it
    # the explicit argument works with an empty environment, as when a launcher drops the variable
    assert mcp_server.build_service({}, settings_path=str(cfg)).store.get_run("stale")["status"] == "failed"


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


class _V1Server: ...


class _V2Server: ...


def test_loader_prefers_sdk_v2_then_falls_back_to_v1_then_explains(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", _module("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", _module("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", _module("mcp.server.mcpserver", MCPServer=_V2Server))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", _module("mcp.server.fastmcp", FastMCP=_V1Server))
    assert mcp_server._load_server_class() is _V2Server  # both installed: v2 wins

    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)  # None blocks the import, as if absent
    assert mcp_server._load_server_class() is _V1Server  # SDK v1 only

    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with pytest.raises(RuntimeError) as e:
        mcp_server._load_server_class()
    assert "Install the `mcp` package" in str(e.value)


def test_main_parses_settings_argument(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(mcp_server, "build_service", lambda settings_path=None: seen.update(path=settings_path) or "svc")

    class Server:
        def run(self, transport=None):
            seen["transport"] = transport

    monkeypatch.setattr(mcp_server, "create_server", lambda svc: Server())
    mcp_server.main(["--settings", "/x/settings.yaml"])
    assert seen == {"path": "/x/settings.yaml", "transport": "stdio"}


def test_file_tools_pass_kind_folder_and_path_through(env):
    tools = mcp_server.make_tools(env.svc)
    listed = call(tools["list_input_files"], kind="fasta", folder=str(env.fx))
    assert listed["status"] == "ok" and any(f["path"].endswith("packaging.fa") for f in listed["files"])
    assert not any(f["path"].endswith("vector.bed") for f in listed["files"])
    described = call(tools["describe_input"], path=str(env.fx / "packaging.fa"))
    assert [r["id"] for r in described["records"]] == ["pRep2Cap9", "pHelper", "Lambda"]
    odd = call(tools["list_input_files"], kind="nonsense", folder=str(env.fx))
    assert odd["status"] == "ok" and "not recognised" in odd["note"]  # tolerated, with a note
    assert paths_of(call(tools["list_input_files"], kind="fastq", folder=str(env.fx)))  # "fastq" means reads


def test_the_screen_tool_names_the_available_databases(env):
    tools = mcp_server.make_tools(env.svc)
    assert "Available databases: demo (default)" in tools["screen_contaminants"].__doc__
    assert "nt" not in tools["screen_contaminants"].__doc__.split("Available databases:")[1]  # not offered: unavailable
    again = mcp_server.make_tools(env.svc)  # rebuilding does not pile up text
    assert again["screen_contaminants"].__doc__.count("Available databases") == 1


def test_list_input_files_tool_takes_the_users_words(env):
    tools = mcp_server.make_tools(env.svc)
    res = call(tools["list_input_files"], query="spike-in fastq")
    assert Path(res["files"][0]["path"]).name == "ss.spikein.fastq.gz" and res["files"][0]["matches"]
    params = list(inspect.signature(tools["list_input_files"]).parameters)
    assert params[0] == "query"  # the first argument a model fills in is the one that works best


def test_the_agent_cannot_set_the_packaging_renaming_options(env):
    names = set(inspect.signature(mcp_server.make_tools(env.svc)["run_laava"]).parameters)
    assert not names & {"repcap_name", "helper_name", "lambda_name"}
    assert {"packaging_fa", "host_fa", "itr_label_1", "mitr_label"} <= names  # the rest of the interface is intact
