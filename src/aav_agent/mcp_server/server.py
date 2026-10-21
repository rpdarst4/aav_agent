"""MCP server exposing the AAV tool layer (stdio transport).

Run with:   python -m aav_agent.mcp_server --settings /abs/path/to/settings.yaml
            (or set AAV_SETTINGS instead of passing --settings)

stdout carries the MCP protocol, so this module never prints; logging goes to stderr.
Long jobs (LAAVA, BLAST) run in the background: run_laava and screen_contaminants return at once and
wait_for_job polls, which keeps every call short enough for client timeouts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable, Mapping

from ..config import Settings
from ..lims.store import FileRunStore, MongoRunStore
from ..service import AavService

logger = logging.getLogger(__name__)

MAX_WAIT_S = 55.0  # stay under typical client tool-call timeouts (often 60 s)

INSTRUCTIONS = """\
Tools for analysing AAV vector long-read sequencing with LAAVA and screening for contaminants.

Typical workflow:
1. list_input_files to find real file paths (reads, vector FASTA/BED, packaging, host). Never invent paths.
   describe_input to read record names from a FASTA and feature names from a BED. Never invent names.
2. run_laava starts a LAAVA run and returns a run_id at once.
3. wait_for_job(run_id) until status is succeeded or failed; repeat while it says still in progress.
4. screen_contaminants(run_id) BLASTs the reads LAAVA could not map, classifies them by organism type,
   and adds a pie chart section to the LAAVA HTML report. Use list_databases to see which databases exist.
5. wait_for_job(run_id, screen_id) until it finishes, then report the buckets and the report_html path.
If a call returns status "invalid" or "error", read the messages, fix the arguments, and try again.
"""


def _clean(values: Mapping) -> dict:
    """Drop arguments the caller left unset."""
    return {k: v for k, v in values.items() if v is not None}


def make_tools(service: AavService) -> dict[str, Callable]:
    """The tool functions, as plain async callables with flat, typed signatures.

    Blocking work runs in a worker thread so the server stays responsive.
    """

    async def list_databases() -> list[dict]:
        """List the BLAST databases available for contaminant screening. Each entry has a name,
        a description, whether its files are present (available), and whether it is the default.
        Use a database's name as the `database` argument of screen_contaminants."""
        return await asyncio.to_thread(service.list_databases)

    async def list_input_files(
        pattern: str | None = None, folder: str | None = None, kind: str | None = None, limit: int = 50
    ) -> dict:
        """Find files the tools are allowed to read. kind is "reads" (FASTQ/BAM), "fasta" (.fa, .fasta, .fna),
        "bed" or "bam"; use kind instead of guessing extensions. pattern is an optional name filter such as
        "*packaging*" (comma-separate several). folder limits the search to one directory, for example
        "laava/test" or a full path. Returns full paths; use them exactly. If truncated is true, narrow the
        search with folder or pattern. Do not guess paths."""
        return await asyncio.to_thread(service.list_input_files, pattern, limit, folder, kind)

    async def describe_input(path: str) -> dict:
        """Look inside an input file. For a FASTA it lists the record names and lengths (use these for
        repcap_name, helper_name and lambda_name); for a BED it lists the feature names (use these for
        itr_label_1, itr_label_2 and mitr_label); for reads it gives the format and size. Use it rather than
        guessing names."""
        return await asyncio.to_thread(service.describe_input, path)

    async def run_laava(
        reads: str,
        vector_fa: str,
        vector_bed: str,
        vector_type: str = "unspecified",
        packaging_fa: str | None = None,
        host_fa: str | None = None,
        repcap_name: str | None = None,
        helper_name: str | None = None,
        lambda_name: str | None = None,
        itr_label_1: str | None = None,
        itr_label_2: str | None = None,
        mitr_label: str | None = None,
        flipflop_name: str | None = None,
        flipflop_fa: str | None = None,
        sample_id: str | None = None,
        target_gap_threshold: int | None = None,
        max_allowed_outside_vector: int | None = None,
        max_allowed_missing_flanking: int | None = None,
        min_supp_joint_coverage: float | None = None,
    ) -> dict:
        """Start a LAAVA analysis of PacBio AAV reads. Returns immediately with a run_id (status
        "queued"); then call wait_for_job. Returns status "invalid" with a list of errors if any input
        is wrong; fix them and call again.

        reads: BAM or FASTQ(.gz) of PacBio reads. vector_fa: the AAV vector plasmid FASTA.
        vector_bed: LAAVA annotation BED for the vector. vector_type: "ss", "sc" or "unspecified".
        packaging_fa: FASTA with the RepCap and helper plasmids (and lambda control). host_fa: host cell
        genome FASTA. repcap_name, helper_name, lambda_name: record names inside packaging_fa.
        itr_label_1/itr_label_2/mitr_label: ITR labels in the BED. flipflop_name and flipflop_fa:
        flip/flop reference. sample_id: defaults to the reads file name. The remaining arguments are
        LAAVA classification thresholds; leave them unset unless asked.
        Paths must come from list_input_files or the user, and must lie in the allowed folders."""
        spec = _clean({
            "reads": reads, "vector_fa": vector_fa, "vector_bed": vector_bed, "vector_type": vector_type,
            "packaging_fa": packaging_fa, "host_fa": host_fa, "repcap_name": repcap_name,
            "helper_name": helper_name, "lambda_name": lambda_name, "itr_label_1": itr_label_1,
            "itr_label_2": itr_label_2, "mitr_label": mitr_label, "flipflop_name": flipflop_name,
            "flipflop_fa": flipflop_fa, "sample_id": sample_id,
            "target_gap_threshold": target_gap_threshold,
            "max_allowed_outside_vector": max_allowed_outside_vector,
            "max_allowed_missing_flanking": max_allowed_missing_flanking,
            "min_supp_joint_coverage": min_supp_joint_coverage,
        })
        return await asyncio.to_thread(service.start_laava, spec)

    async def screen_contaminants(run_id: str, database: str | None = None, truth: str | None = None) -> dict:
        """Start a contaminant screen of a finished LAAVA run: the reads LAAVA could not map are BLASTed,
        classified by organism type (e.g. plant, insect, enterobacteria, human), and a pie chart section is
        added to a copy of the LAAVA HTML report. Returns immediately with a screen_id; then call
        wait_for_job with both run_id and screen_id. The run must have status succeeded.
        database: a name from list_databases (default if omitted). truth: only for demonstration data
        with a spike-in truth table; adds a validation table."""
        return await asyncio.to_thread(service.start_screen, run_id, database, truth)

    async def wait_for_job(run_id: str, screen_id: str | None = None, timeout_s: float = 45.0) -> dict:
        """Wait up to timeout_s seconds (maximum 55) for a LAAVA run, or for one screen of it when
        screen_id is given. Returns the current status either way. If the status is queued or running,
        call this again. When it is succeeded, the result includes the summary or buckets and file paths;
        when failed, it includes the error and the end of the log."""
        timeout = min(max(float(timeout_s), 0.0), MAX_WAIT_S)
        return await asyncio.to_thread(service.wait_for_job, run_id, screen_id, timeout)

    async def get_run(run_id: str, detail: bool = False) -> dict:
        """Get a LAAVA run's record: status, sample id, output paths, read counts by LAAVA assignment,
        and its contaminant screens. detail=true adds the inputs with checksums, parameters and the
        provenance (code, image and tool versions)."""
        return await asyncio.to_thread(service.get_run, run_id, detail)

    async def list_runs(limit: int = 10) -> list[dict]:
        """List recent runs, newest first, with status, sample id and unmapped read count. Use it to find
        a run_id when the user refers to an earlier run."""
        return await asyncio.to_thread(service.list_runs, limit)

    tools = [list_databases, list_input_files, describe_input, run_laava, screen_contaminants, wait_for_job, get_run, list_runs]
    return {t.__name__: t for t in tools}


def _load_server_class():
    """The SDK's high-level server class: `MCPServer` in SDK v2, `FastMCP` in SDK v1."""
    try:
        from mcp.server.mcpserver import MCPServer  # SDK v2 (2.0.0 removed mcp.server.fastmcp)

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # SDK v1

        return FastMCP
    except ImportError as e:
        raise RuntimeError(
            "Could not import the MCP server class. Install the `mcp` package "
            "(v2: mcp.server.mcpserver.MCPServer; v1: mcp.server.fastmcp.FastMCP)."
        ) from e


def create_server(service: AavService, server_cls=None):
    """Build the MCP server. `server_cls` lets tests substitute a fake; by default the SDK's own class."""
    server_cls = server_cls or _load_server_class()
    server = server_cls("aav-agent", instructions=INSTRUCTIONS)
    for name, fn in make_tools(service).items():
        server.tool(name=name, description=fn.__doc__)(fn)
    return server


def build_service(environ: Mapping[str, str] | None = None, settings_path: str | None = None) -> AavService:
    """Settings from --settings or $AAV_SETTINGS; run records in MongoDB ($AAV_MONGO_URI) or JSON files.

    MCP launchers (the inspector, SDK clients) deliberately pass only a small, safe subset of the
    parent environment to the server, so an exported AAV_SETTINGS may never arrive. The --settings
    argument is the reliable way to say where the settings live.
    """
    environ = os.environ if environ is None else environ
    path = settings_path or environ.get("AAV_SETTINGS")
    if not path:
        raise SystemExit(
            "Say where the settings file is: pass --settings /abs/path/settings.yaml, or set AAV_SETTINGS. "
            "(MCP launchers such as the inspector do not forward your shell's environment; use their -e flag.)"
        )
    settings = Settings.from_yaml(path)
    uri = environ.get("AAV_MONGO_URI")
    store = (MongoRunStore.from_uri(uri, environ.get("AAV_MONGO_DB", "aav_agent")) if uri
             else FileRunStore(settings.data_dir / "lims"))
    service = AavService(settings, store)
    stale = service.recover_interrupted()
    if stale:
        logger.warning("marked %d interrupted run(s)/screen(s) from a previous session as failed", stale)
    return service


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MCP server for the AAV agent tools (stdio).")
    parser.add_argument("--settings", help="Path to settings.yaml (overrides $AAV_SETTINGS)")
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    create_server(build_service(settings_path=args.settings)).run(transport="stdio")
