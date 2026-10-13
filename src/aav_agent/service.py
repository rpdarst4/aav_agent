"""The tool layer: plain Python functions the MCP server will expose.

Every method returns a JSON-serialisable dict and reports expected problems (bad input,
missing database, failed run) as data with a "status" field instead of raising, so a language
model can read the message and correct itself. Unexpected exceptions still propagate, after the
run record has been marked failed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import secrets
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from aav_agent.config import PathNotAllowed, Settings
from aav_agent.lims.store import RunStore
from aav_agent.provenance import collect_provenance, utcnow
from aav_agent.runner import CommandRunner, SubprocessRunner, tail
from aav_agent.spec import RunSpec, SpecError, build_laava_params, validate_run_spec


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


def _default_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def _public(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    out = {k: v for k, v in doc.items() if k != "_id"}
    out["id"] = doc["_id"]
    return out


class AavService:
    def __init__(
        self,
        settings: Settings,
        store: RunStore,
        runner: CommandRunner | None = None,
        run_id_factory: Callable[[], str] = _default_run_id,
    ):
        self.settings = settings
        self.store = store
        self.runner = runner or SubprocessRunner()
        self._new_run_id = run_id_factory

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

    def run_laava(self, spec: RunSpec | dict) -> dict:
        """Run LAAVA on a set of reads. Creates a run folder and a record, and returns the outputs."""
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

        now = utcnow()
        self.store.create_run({
            "_id": run_id, "type": "laava", "status": "running", "sample_id": v.sample_id,
            "created_utc": now, "run_dir": str(run_dir), "spec": asdict(spec), "params": params,
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
        log = run_dir / "logs" / "laava.log"
        sid = v.sample_id
        outputs = {
            "name_sorted_bam": run_dir / "output" / "alignment" / f"{sid}.sort_by_name.bam",
            "per_read_tsv": run_dir / "output" / "report" / f"{sid}.per_read.tsv.gz",
            "report_html": run_dir / "output" / "report" / f"{sid}_AAV_report.html",
        }
        try:
            result = self.runner.run(cmd, cwd=run_dir, env=self._nextflow_env(), log_path=log)
            missing = [k for k, p in outputs.items() if not p.exists()]
            if result.returncode != 0:
                error = f"nextflow exited with code {result.returncode}"
            elif missing:
                error = f"LAAVA finished but these expected outputs are missing: {', '.join(missing)}"
            else:
                error = None
        except BaseException as e:
            self.store.update_run(run_id, {"status": "failed", "finished_utc": utcnow(), "error": repr(e)})
            raise

        if error:
            log_tail = tail(log)
            self.store.update_run(run_id, {"status": "failed", "finished_utc": utcnow(),
                                           "error": error, "log_tail": log_tail})
            return {"status": "failed", "run_id": run_id, "error": error, "log_tail": log_tail,
                    "log": str(log), "warnings": v.warnings}

        summary = summarize_per_read(outputs["per_read_tsv"])
        out_paths = {k: str(p) for k, p in outputs.items()}
        out_paths["laava_output"] = str(run_dir / "output")
        self.store.update_run(run_id, {"status": "succeeded", "finished_utc": utcnow(),
                                       "outputs": out_paths, "summary": summary})
        return {"status": "succeeded", "run_id": run_id, "sample_id": sid, "outputs": out_paths,
                "summary": summary, "warnings": v.warnings}

    def screen_contaminants(self, run_id: str, database: str | None = None, truth: str | None = None) -> dict:
        """BLAST the reads LAAVA could not map, classify them by organism type, and add a section to the report."""
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
        n = len(self.store.list_screens(run_id)) + 1
        screen_id = f"screen-{n:03d}"
        doc_id = f"{run_id}:{screen_id}"
        outdir = run_dir / "screens" / screen_id
        (outdir / "work").mkdir(parents=True)

        prov = collect_provenance(s, self.runner, include_laava=False)
        prov["categories_sha256"] = sha256_file(s.categories)
        self.store.create_screen({
            "_id": doc_id, "type": "screen", "run_id": run_id, "screen_id": screen_id, "status": "running",
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
        log = outdir / "screen.log"
        report = outdir / sid / "report" / f"{sid}_AAV_report.contaminants.html"
        summary_json = outdir / sid / "classification" / f"{sid}.classified.json"
        try:
            result = self.runner.run(cmd, cwd=outdir, env=self._nextflow_env(), log_path=log)
            if result.returncode != 0:
                error = f"nextflow exited with code {result.returncode}"
            elif not (report.exists() and summary_json.exists()):
                error = "the screen finished but its report or summary file is missing"
            else:
                error = None
        except BaseException as e:
            self.store.update_screen(doc_id, {"status": "failed", "finished_utc": utcnow(), "error": repr(e)})
            raise

        if error:
            log_tail = tail(log)
            self.store.update_screen(doc_id, {"status": "failed", "finished_utc": utcnow(),
                                              "error": error, "log_tail": log_tail})
            return {"status": "failed", "run_id": run_id, "screen_id": screen_id, "error": error,
                    "log_tail": log_tail, "log": str(log)}

        summary = json.loads(summary_json.read_text())
        buckets = {b: v["reads"] for b, v in summary["buckets"].items()}
        db_meta = summary.get("database") or {}
        result_doc = {
            "report_html": str(report), "classified_json": str(summary_json),
            "unmapped_reads_screened": summary["total_reads"], "buckets": buckets,
            "database_recorded": {k: db_meta.get(k) for k in ("db_title", "db_date", "db_sequences", "blastn")},
        }
        if "validation" in summary:
            result_doc["validation"] = {k: summary["validation"][k] for k in
                                        ("spiked_reads", "leaked_reads", "correct", "recovery_of_classifiable")}
        self.store.update_screen(doc_id, {"status": "succeeded", "finished_utc": utcnow(), **result_doc})
        self.store.update_run(run_id, {"latest_screen": screen_id})
        return {"status": "succeeded", "run_id": run_id, "screen_id": screen_id, "database": name, **result_doc}

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
