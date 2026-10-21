"""The tool layer: plain Python functions the MCP server will expose.

Every method returns a JSON-serialisable dict and reports expected problems (bad input,
missing database, failed run) as data with a "status" field instead of raising, so a language
model can read the message and correct itself. Unexpected exceptions still propagate, after the
run record has been marked failed.
"""

from __future__ import annotations

import csv
import fnmatch
import gzip
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import PathNotAllowed, Settings
from .lims.store import RunStore
from .provenance import collect_provenance, utcnow
from .runner import CommandRunner, SubprocessRunner, tail
from .spec import RunSpec, SpecError, build_laava_params, validate_run_spec

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_record(role: str, path: Path, max_hash_bytes: int) -> dict:
    size = path.stat().st_size
    if size > max_hash_bytes:
        return {"role": role, "path": str(path), "size": size, "sha256": None,
                "note": f"not checksummed (larger than {max_hash_bytes} bytes)"}
    return {"role": role, "path": str(path), "size": size, "sha256": sha256_file(path)}


def summarize_per_read(path: Path) -> dict:
    """Counts from LAAVA's per-read table."""
    opener = gzip.open if str(path).endswith(".gz") else open
    by_label: Counter = Counter()
    by_type: Counter = Counter()
    with opener(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            by_label[row.get("reference_label", "")] += 1
            by_type[row.get("assigned_type", "")] += 1
    return {
        "total_reads": sum(by_label.values()),
        "unmapped_reads": by_label.get("(unmapped)", 0),
        "by_reference_label": dict(by_label.most_common()),
        "by_assigned_type": dict(by_type.most_common()),
    }


OTHER_VECTOR_WARN_FRACTION = 0.9
UNMAPPED_WARN_FRACTION = 0.5


def assess_summary(summary: dict) -> list[str]:
    """Warnings about a finished run's read assignments that suggest mismatched inputs.

    Heuristics, not LAAVA errors: LAAVA happily reports whatever it finds. When reads come from a different
    construct than the one supplied, nearly all of them end up "other-vector" (they map, but do not look like
    ITR-to-ITR genomes) or unmapped.
    """
    total = summary.get("total_reads") or 0
    if not total:
        return ["LAAVA found no reads in the input."]
    out = []
    unmapped = summary.get("unmapped_reads", 0)
    if unmapped / total >= UNMAPPED_WARN_FRACTION:
        out.append(f"{unmapped:,} of {total:,} reads ({unmapped / total:.0%}) did not map to any supplied reference. "
                   "Check that the reads belong to this construct and that the references are the right ones.")
    other = (summary.get("by_assigned_type") or {}).get("other-vector", 0)
    if other / total >= OTHER_VECTOR_WARN_FRACTION:
        out.append(f"{other:,} of {total:,} reads ({other / total:.0%}) were assigned 'other-vector': they map to the "
                   "vector but do not look like AAV genomes between the ITRs. This is typical when the reads come "
                   "from a different construct, sample or annotation than the one supplied; check that the reads, "
                   "vector FASTA and annotation BED belong together.")
    return out


def _default_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _public(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    out = {k: v for k, v in doc.items() if k != "_id"}
    out["id"] = doc["_id"]
    return out


@dataclass
class _LaavaJob:
    run_id: str
    sample_id: str
    run_dir: Path
    cmd: list
    log: Path
    outputs: dict
    warnings: list


@dataclass
class _ScreenJob:
    run_id: str
    screen_id: str
    doc_id: str
    sample_id: str
    outdir: Path
    cmd: list
    log: Path
    report: Path
    summary_json: Path
    database_name: str


class AavService:
    def __init__(
        self,
        settings: Settings,
        store: RunStore,
        runner: CommandRunner | None = None,
        run_id_factory: Callable[[], str] = _default_run_id,
        max_workers: int = 1,
    ):
        self.settings = settings
        self.store = store
        self.runner = runner or SubprocessRunner()
        self._new_run_id = run_id_factory
        self._lock = threading.Lock()  # serialises record creation (screen numbering)
        # One job at a time by default: LAAVA and BLAST are heavy on a laptop. Later jobs wait as "queued".
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aav-job")

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------ helpers

    def _nextflow_env(self) -> dict[str, str]:
        env = {}
        if self.settings.nxf_syntax_parser:
            env["NXF_SYNTAX_PARSER"] = self.settings.nxf_syntax_parser
        if self.settings.nxf_version:
            env["NXF_VER"] = self.settings.nxf_version
        return env

    def _error(self, message: str, **extra) -> dict:
        return {"status": "error", "message": message, **extra}

    # ------------------------------------------------------------------ tools

    def list_databases(self) -> list[dict]:
        return [
            {"name": c.name, "description": c.description, "available": c.available(),
             "default": c.name == self.settings.default_database}
            for c in self.settings.databases.values()
        ]

    # ---------------------------------------------------------------- LAAVA

    def _prepare_laava(self, spec: RunSpec | dict) -> dict | _LaavaJob:
        """Validate, create the run folder and record (status "queued"). Returns an error dict or a job."""
        s = self.settings
        if isinstance(spec, dict):
            try:
                spec = RunSpec.from_dict(spec)
            except SpecError as e:
                return {"status": "invalid", "errors": e.errors, "warnings": []}
        v = validate_run_spec(spec, s)
        if v.errors:
            return {"status": "invalid", "errors": v.errors, "warnings": v.warnings}

        run_id = self._new_run_id()
        run_dir = s.runs_dir / run_id
        for sub in ("output", "work", "logs"):
            (run_dir / sub).mkdir(parents=True)
        params = build_laava_params(spec, v)
        params_path = run_dir / "params.json"
        params_path.write_text(json.dumps(params, indent=2) + "\n")
        (run_dir / "spec.json").write_text(json.dumps(asdict(spec), indent=2) + "\n")

        self.store.create_run({
            "_id": run_id, "type": "laava", "status": "queued", "sample_id": v.sample_id,
            "created_utc": utcnow(), "run_dir": str(run_dir), "spec": asdict(spec), "params": params,
            "warnings": v.warnings,
            "inputs": [file_record(role, p, s.max_hash_bytes) for role, p in sorted(v.paths.items())],
            "provenance": collect_provenance(s, self.runner, include_laava=True),
        })
        cmd = [
            s.nextflow, "run", "-profile", "local", str(s.laava_dir / "main.nf"),
            "-params-file", str(params_path),
            "--output", str(run_dir / "output"), "-w", str(run_dir / "work"),
            "--cpus", str(s.cpus), "--memory", s.memory,
            "--container_version", s.laava_container_version,
        ]
        sid = v.sample_id
        outputs = {
            "name_sorted_bam": run_dir / "output" / "alignment" / f"{sid}.sort_by_name.bam",
            "per_read_tsv": run_dir / "output" / "report" / f"{sid}.per_read.tsv.gz",
            "report_html": run_dir / "output" / "report" / f"{sid}_AAV_report.html",
        }
        return _LaavaJob(run_id, sid, run_dir, cmd, run_dir / "logs" / "laava.log", outputs, v.warnings)

    def _execute_laava(self, job: _LaavaJob) -> dict:
        self.store.update_run(job.run_id, {"status": "running", "started_utc": utcnow()})
        try:
            result = self.runner.run(job.cmd, cwd=job.run_dir, env=self._nextflow_env(), log_path=job.log)
            missing = [k for k, p in job.outputs.items() if not p.exists()]
            if result.returncode != 0:
                error = f"nextflow exited with code {result.returncode}"
            elif missing:
                error = f"LAAVA finished but these expected outputs are missing: {', '.join(missing)}"
            else:
                error = None
        except BaseException as e:
            self.store.update_run(job.run_id, {"status": "failed", "finished_utc": utcnow(), "error": repr(e)})
            raise

        if error:
            log_tail = tail(job.log)
            self.store.update_run(job.run_id, {"status": "failed", "finished_utc": utcnow(),
                                               "error": error, "log_tail": log_tail})
            return {"status": "failed", "run_id": job.run_id, "error": error, "log_tail": log_tail,
                    "log": str(job.log), "warnings": job.warnings}

        summary = summarize_per_read(job.outputs["per_read_tsv"])
        warnings = job.warnings + assess_summary(summary)
        out_paths = {k: str(p) for k, p in job.outputs.items()}
        out_paths["laava_output"] = str(job.run_dir / "output")
        self.store.update_run(job.run_id, {"status": "succeeded", "finished_utc": utcnow(), "outputs": out_paths,
                                           "summary": summary, "warnings": warnings})
        return {"status": "succeeded", "run_id": job.run_id, "sample_id": job.sample_id, "outputs": out_paths,
                "summary": summary, "warnings": warnings}

    _SKIP_DIRS = {"work", ".git", "node_modules", "__pycache__", ".nextflow"}

    def _resolve_folder(self, folder: str) -> Path | None:
        """A directory under the allowed folders. Relative names are tried against each allowed folder and
        its parent, so "laava/test" finds <projects>/laava/test."""
        p = Path(folder).expanduser()
        if p.is_absolute():
            candidates = [p]
        else:
            roots = self.settings.allowed_input_roots
            candidates = [r / p for r in roots] + [r.parent / p for r in roots]
        for c in candidates:
            try:
                resolved = self.settings.resolve_input(str(c))
            except PathNotAllowed:
                continue
            if resolved.is_dir():
                return resolved
        return None

    def list_input_files(self, pattern: str | None = None, limit: int = 50, folder: str | None = None,
                         kind: str | None = None) -> dict:
        """Files the tools may read, found by `kind` ("reads", "fasta", "bed", "bam") and/or name pattern(s).

        `kind` and `pattern` together mean "files of this kind whose name matches"; either alone is enough.
        Patterns are shell-style globs, comma-separated ("*packaging*,*host*"). `folder` limits the search to
        one directory. Lets an agent find real paths instead of guessing them; Nextflow work folders and
        previous run outputs are skipped.
        """
        from .inputs import KIND_PATTERNS

        if kind and kind not in KIND_PATTERNS:
            return self._error(f"unknown kind {kind!r}", valid_kinds=sorted(KIND_PATTERNS))
        # kind and pattern narrow each other: a file must have the kind's extension AND match a pattern
        kind_patterns = [p.lower() for p in KIND_PATTERNS[kind]] if kind else ["*"]
        name_patterns = [p.strip().lower() for p in (pattern or "").split(",") if p.strip()] or ["*"]
        patterns = {"kind": kind_patterns if kind else None, "pattern": name_patterns if pattern else None}

        def wanted(name: str) -> bool:
            lowered = name.lower()
            return (any(fnmatch.fnmatchcase(lowered, k) for k in kind_patterns)
                    and any(fnmatch.fnmatchcase(lowered, n) for n in name_patterns))

        if folder:
            found_dir = self._resolve_folder(folder)
            if found_dir is None:
                return self._error(f"folder not found under the allowed folders: {folder!r}",
                                   folders_searched=[str(r) for r in self.settings.allowed_input_roots])
            search = [found_dir]
        else:
            search = [r for r in self.settings.allowed_input_roots if r.is_dir()]

        runs_dir = self.settings.runs_dir.resolve()
        found: list[str] = []
        truncated = False
        for root in search:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames
                                     if d not in self._SKIP_DIRS and (Path(dirpath) / d).resolve() != runs_dir)
                for name in sorted(filenames):
                    if wanted(name):
                        if len(found) >= limit:
                            truncated = True
                            break
                        found.append(str(Path(dirpath) / name))
                if truncated:
                    break
            if truncated:
                break
        return {"status": "ok", "patterns": patterns, "files": found, "truncated": truncated,
                "folders_searched": [str(r) for r in search]}

    def describe_input(self, path: str) -> dict:
        """What is inside an input file: record names and lengths of a FASTA, feature names of a BED,
        or the type and size of a reads file. Use it to take names from the file instead of guessing."""
        from .inputs import READ_SUFFIXES, bed_features, fasta_records, is_fasta

        try:
            p = self.settings.resolve_input(path)
        except PathNotAllowed as e:
            return self._error(str(e))
        if not p.is_file():
            return self._error(f"file not found: {p}")
        base = {"status": "ok", "path": str(p), "size_bytes": p.stat().st_size}
        name = p.name.lower()
        if is_fasta(name):
            return {**base, "type": "fasta", **fasta_records(p)}
        if name.endswith(".bed"):
            return {**base, "type": "bed", **bed_features(p)}
        if name.endswith(READ_SUFFIXES):
            return {**base, "type": "reads", "format": "bam" if name.endswith(".bam") else "fastq"}
        return {**base, "type": "other"}

    def run_laava(self, spec: RunSpec | dict) -> dict:
        """Run LAAVA and wait for it to finish (blocking). Tests and scripts use this."""
        job = self._prepare_laava(spec)
        return job if isinstance(job, dict) else self._execute_laava(job)

    def start_laava(self, spec: RunSpec | dict) -> dict:
        """Validate and queue a LAAVA run; returns immediately. Poll with wait_for_job."""
        job = self._prepare_laava(spec)
        if isinstance(job, dict):
            return job
        self._submit(self._execute_laava, job)
        return {"status": "queued", "run_id": job.run_id, "sample_id": job.sample_id,
                "warnings": job.warnings,
                "next": "Call wait_for_job with this run_id until the status is succeeded or failed."}

    # ---------------------------------------------------------------- screening

    def _prepare_screen(self, run_id: str, database: str | None, truth: str | None) -> dict | _ScreenJob:
        s = self.settings
        run = self.store.get_run(run_id)
        if run is None or run.get("type") != "laava":
            recent = [r["_id"] for r in self.store.list_runs(5)]
            return self._error(f"no such run: {run_id}", recent_runs=recent)
        if run["status"] != "succeeded":
            return self._error(f"run {run_id} has status {run['status']!r}; only succeeded runs can be screened")

        name = database or s.default_database
        cfg = s.databases.get(name)
        if cfg is None:
            return self._error(f"unknown database {name!r}", available_databases=sorted(s.databases))
        if not cfg.available():
            return self._error(f"database {name!r} is configured but no BLAST files were found at "
                               f"{cfg.dir}/{cfg.db_name}.*", available_databases=[
                                   d["name"] for d in self.list_databases() if d["available"]])
        for label, p in (("taxdump", s.taxdump_dir / "nodes.dmp"), ("categories file", s.categories),
                         ("screening pipeline", s.screen_pipeline)):
            if not p.exists():
                return self._error(f"{label} not found: {p}")

        truth_path = None
        if truth:
            try:
                truth_path = s.resolve_input(truth)
            except PathNotAllowed as e:
                return self._error(f"truth: {e}")
            if not truth_path.is_file():
                return self._error(f"truth: file not found: {truth_path}")

        sid = run["sample_id"]
        run_dir = Path(run["run_dir"])
        prov = collect_provenance(s, self.runner, include_laava=False)
        prov["categories_sha256"] = sha256_file(s.categories)
        with self._lock:  # numbering and creation must be atomic
            n = len(self.store.list_screens(run_id)) + 1
            screen_id = f"screen-{n:03d}"
            doc_id = f"{run_id}:{screen_id}"
            outdir = run_dir / "screens" / screen_id
            (outdir / "work").mkdir(parents=True)
            self.store.create_screen({
                "_id": doc_id, "type": "screen", "run_id": run_id, "screen_id": screen_id, "status": "queued",
                "created_utc": utcnow(), "database": {"name": name, "dir": str(cfg.dir), "db_name": cfg.db_name},
                "truth": str(truth_path) if truth_path else None, "provenance": prov, "outdir": str(outdir),
            })

        cmd = [
            s.nextflow, "run", "-profile", "local", str(s.screen_pipeline),
            "-w", str(outdir / "work"),
            "--laava_output", run["outputs"]["laava_output"], "--sample_id", sid,
            "--blast_db_dir", str(cfg.dir), "--blast_db_name", cfg.db_name,
            "--taxdump_dir", str(s.taxdump_dir), "--categories", str(s.categories),
            "--outdir", str(outdir), "--tools_image", s.tools_image,
        ]
        if truth_path:
            cmd += ["--truth", str(truth_path), "--show_validation"]
        return _ScreenJob(
            run_id, screen_id, doc_id, sid, outdir, cmd, outdir / "screen.log",
            outdir / sid / "report" / f"{sid}_AAV_report.contaminants.html",
            outdir / sid / "classification" / f"{sid}.classified.json", name,
        )

    def _execute_screen(self, job: _ScreenJob) -> dict:
        self.store.update_screen(job.doc_id, {"status": "running", "started_utc": utcnow()})
        try:
            result = self.runner.run(job.cmd, cwd=job.outdir, env=self._nextflow_env(), log_path=job.log)
            if result.returncode != 0:
                error = f"nextflow exited with code {result.returncode}"
            elif not (job.report.exists() and job.summary_json.exists()):
                error = "the screen finished but its report or summary file is missing"
            else:
                error = None
        except BaseException as e:
            self.store.update_screen(job.doc_id, {"status": "failed", "finished_utc": utcnow(), "error": repr(e)})
            raise

        if error:
            log_tail = tail(job.log)
            self.store.update_screen(job.doc_id, {"status": "failed", "finished_utc": utcnow(),
                                                  "error": error, "log_tail": log_tail})
            return {"status": "failed", "run_id": job.run_id, "screen_id": job.screen_id, "error": error,
                    "log_tail": log_tail, "log": str(job.log)}

        summary = json.loads(job.summary_json.read_text())
        buckets = {b: v["reads"] for b, v in summary["buckets"].items()}
        db_meta = summary.get("database") or {}
        result_doc = {
            "report_html": str(job.report), "classified_json": str(job.summary_json),
            "unmapped_reads_screened": summary["total_reads"], "buckets": buckets,
            "database_recorded": {k: db_meta.get(k) for k in ("db_title", "db_date", "db_sequences", "blastn")},
        }
        if "validation" in summary:
            result_doc["validation"] = {k: summary["validation"][k] for k in
                                        ("spiked_reads", "leaked_reads", "correct", "recovery_of_classifiable")}
        self.store.update_screen(job.doc_id, {"status": "succeeded", "finished_utc": utcnow(), **result_doc})
        self.store.update_run(job.run_id, {"latest_screen": job.screen_id})
        return {"status": "succeeded", "run_id": job.run_id, "screen_id": job.screen_id,
                "database": job.database_name, **result_doc}

    def screen_contaminants(self, run_id: str, database: str | None = None, truth: str | None = None) -> dict:
        """BLAST the reads LAAVA could not map, classify them by organism type, and add a report section
        (blocking). Tests and scripts use this."""
        job = self._prepare_screen(run_id, database, truth)
        return job if isinstance(job, dict) else self._execute_screen(job)

    def start_screen(self, run_id: str, database: str | None = None, truth: str | None = None) -> dict:
        """Validate and queue a contaminant screen; returns immediately. Poll with wait_for_job."""
        job = self._prepare_screen(run_id, database, truth)
        if isinstance(job, dict):
            return job
        self._submit(self._execute_screen, job)
        return {"status": "queued", "run_id": job.run_id, "screen_id": job.screen_id,
                "next": "Call wait_for_job with this run_id and screen_id until the status is succeeded or failed."}

    # ---------------------------------------------------------------- background jobs

    def _submit(self, fn: Callable, job) -> Future:
        fut = self._executor.submit(fn, job)

        def _log_failure(f: Future) -> None:
            exc = f.exception()
            if exc is not None:  # the record was already marked failed by the executor function
                logger.error("background job %s crashed: %r", getattr(job, "run_id", "?"), exc)

        fut.add_done_callback(_log_failure)
        return fut

    _JOB_FIELDS = ("status", "error", "log_tail", "outputs", "summary", "warnings", "buckets", "report_html",
                   "unmapped_reads_screened", "database_recorded", "validation")

    def job_status(self, run_id: str, screen_id: str | None = None) -> dict:
        """Current state of a LAAVA run, or of one screen of it when screen_id is given."""
        if screen_id:
            docs = {d["screen_id"]: d for d in self.store.list_screens(run_id)}
            doc = docs.get(screen_id)
            if doc is None:
                return self._error(f"no screen {screen_id!r} for run {run_id!r}", screens=sorted(docs))
            out = {"run_id": run_id, "screen_id": screen_id, "database": doc["database"]["name"]}
        else:
            doc = self.store.get_run(run_id)
            if doc is None:
                return self._error(f"no such run: {run_id}",
                                   recent_runs=[r["_id"] for r in self.store.list_runs(5)])
            out = {"run_id": run_id, "sample_id": doc.get("sample_id")}
        out.update({k: doc[k] for k in self._JOB_FIELDS if k in doc})
        return out

    def wait_for_job(self, run_id: str, screen_id: str | None = None, timeout_s: float = 45.0,
                     poll_s: float = 1.0) -> dict:
        """Wait up to timeout_s for a job to finish. Returns its status either way, so callers can loop."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        start = time.monotonic()
        while True:
            status = self.job_status(run_id, screen_id)
            if status.get("status") not in ACTIVE:
                break
            if time.monotonic() >= deadline:
                status["message"] = "Still in progress. Call wait_for_job again."
                break
            time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))
        status["waited_s"] = round(time.monotonic() - start, 1)
        return status

    def recover_interrupted(self) -> int:
        """Mark records left queued/running by a previous server process as failed. Returns how many."""
        n = 0
        for run in self.store.list_runs(100_000):
            if run["status"] in ACTIVE:
                self.store.update_run(run["_id"], {"status": "failed", "finished_utc": utcnow(),
                                                   "error": "interrupted: the server stopped while this run was in progress"})
                n += 1
            for sc in self.store.list_screens(run["_id"]):
                if sc["status"] in ACTIVE:
                    self.store.update_screen(sc["_id"], {"status": "failed", "finished_utc": utcnow(),
                                                         "error": "interrupted: the server stopped while this screen was in progress"})
                    n += 1
        return n

    def get_run(self, run_id: str, detail: bool = False) -> dict:
        doc = self.store.get_run(run_id)
        if doc is None:
            return self._error(f"no such run: {run_id}", recent_runs=[r["_id"] for r in self.store.list_runs(5)])
        public = _public(doc)
        screens = self.store.list_screens(run_id)
        public["screens"] = [
            {"screen_id": sc["screen_id"], "status": sc["status"], "database": sc["database"]["name"],
             "buckets": sc.get("buckets"), "report_html": sc.get("report_html")}
            for sc in screens
        ]
        if not detail:
            for bulky in ("params", "inputs", "provenance", "spec", "log_tail"):
                public.pop(bulky, None)
        return public

    def list_runs(self, limit: int = 20) -> list[dict]:
        rows = []
        for d in self.store.list_runs(limit):
            rows.append({
                "run_id": d["_id"], "sample_id": d.get("sample_id"), "status": d["status"],
                "created_utc": d["created_utc"],
                "unmapped_reads": (d.get("summary") or {}).get("unmapped_reads"),
                "latest_screen": d.get("latest_screen"),
            })
        return rows
