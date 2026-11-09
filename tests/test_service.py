"""Tests for the tool layer. No Nextflow, Docker, Mongo or LLM needed: commands go through a
FakeRunner that creates the files the real pipelines would, and MongoRunStore is exercised
against a small fake collection."""

import copy
import gzip
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pytest

from aav_agent.config import DatabaseConfig, PathNotAllowed, Settings
from aav_agent.lims.store import FileRunStore, InMemoryRunStore, MongoRunStore
from aav_agent.runner import CommandResult
from aav_agent.service import AavService
from aav_agent.spec import RunSpec, SpecError, validate_run_spec
from aav_agent.tools import add_contaminants_to_report as report_tools

# Real LAAVA output (sc.subsample005, 1687 reads, 1 unmapped) and a structural copy of a real LAAVA
# HTML report. The fake LAAVA below copies these, so the code under test parses genuine output.
FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_BAM = FIXTURES / "sc.subsample005.sort_by_name.bam"
GOLDEN_PER_READ = FIXTURES / "sc.subsample005.per_read.tsv.gz"
GOLDEN_REPORT = FIXTURES / "laava_report_skeleton.html"
GOLDEN_TOTAL_READS, GOLDEN_UNMAPPED = 1687, 1


class FakeRunner:
    """Stands in for subprocess: answers provenance queries and fakes the two Nextflow pipelines."""

    def __init__(self, nf_returncode=0, laava_outputs=True, screen_outputs=True, git_ok=True,
                 docker_ok=True, raise_on_nextflow=None):
        self.calls = []
        self.nf_returncode = nf_returncode
        self.laava_outputs = laava_outputs
        self.screen_outputs = screen_outputs
        self.git_ok = git_ok
        self.docker_ok = docker_ok
        self.raise_on_nextflow = raise_on_nextflow
        self.per_read_text = None  # replaces the golden per-read table (to simulate odd LAAVA results)
        self.hold = None  # a threading.Event: pipeline runs block until it is set
        self.active = 0
        self.max_active = 0
        self._count_lock = threading.Lock()

    def nextflow_calls(self):
        return [c for c in self.calls if c["args"][0] == "nextflow" and "run" in c["args"]]

    def run(self, args, *, cwd=None, env=None, log_path=None, timeout=None):
        self.calls.append({"args": list(args), "cwd": cwd, "env": env, "log_path": log_path})
        exe = args[0]
        if exe == "git":
            if not self.git_ok:
                return CommandResult(128, "", "fatal: not a git repository")
            if "--abbrev-ref" in args:
                return CommandResult(0, "main\n")
            if "rev-parse" in args:
                return CommandResult(0, "abc123\n")
            return CommandResult(0, "")  # clean tree
        if exe == "docker":
            if not self.docker_ok:
                return CommandResult(127, "", "command not found: docker")
            if args[-1] == "{{.Id}}":
                return CommandResult(0, "sha256:deadbeef\n")
            return CommandResult(0, '["registry/img@sha256:1111"]\n')
        if exe == "nextflow" and "-version" in args:
            return CommandResult(0, "\n  N E X T F L O W\n  version 25.10.0 build 5000\n")
        if exe == "nextflow" and "run" in args:
            with self._count_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                if self.hold is not None:
                    self.hold.wait(timeout=10)
                return self._run_pipeline(args, log_path)
            finally:
                with self._count_lock:
                    self.active -= 1
        raise AssertionError(f"unexpected command: {args}")

    def _run_pipeline(self, args, log_path):
        if self.raise_on_nextflow:
            raise self.raise_on_nextflow
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("line 1\nline 2\nsomething broke\n" if self.nf_returncode else "ok\n")
        if self.nf_returncode == 0:
            script = next(a for a in args if a.endswith(".nf"))
            (self._fake_laava if script.endswith("main.nf") else self._fake_screen)(args)
        return CommandResult(self.nf_returncode)

    @staticmethod
    def _opt(args, name):
        return args[args.index(name) + 1]

    def _fake_laava(self, args):
        if not self.laava_outputs:
            return
        out = Path(self._opt(args, "--output"))
        sid = json.loads(Path(self._opt(args, "-params-file")).read_text())["sample_unique_id"]
        (out / "alignment").mkdir(parents=True)
        (out / "report").mkdir(parents=True)
        shutil.copy(GOLDEN_BAM, out / "alignment" / f"{sid}.sort_by_name.bam")
        if self.per_read_text is None:
            shutil.copy(GOLDEN_PER_READ, out / "report" / f"{sid}.per_read.tsv.gz")
        else:
            with gzip.open(out / "report" / f"{sid}.per_read.tsv.gz", "wt") as fh:
                fh.write(self.per_read_text)
        shutil.copy(GOLDEN_REPORT, out / "report" / f"{sid}_AAV_report.html")

    def _fake_screen(self, args):
        if not self.screen_outputs:
            return
        outdir = Path(self._opt(args, "--outdir"))
        sid = self._opt(args, "--sample_id")
        (outdir / sid / "report").mkdir(parents=True)
        (outdir / sid / "classification").mkdir(parents=True)
        summary = {
            "total_reads": 1,
            "buckets": {"enterobacteria": {"reads": 1, "read_fraction": 1.0, "bases": 100, "base_fraction": 1.0}},
            "database": {"db_title": "AAV demo contaminants", "db_date": "Oct 7, 2026", "db_sequences": 8,
                         "blastn": "blastn: 2.16.0+"},
        }
        if "--truth" in args:
            summary["validation"] = {"spiked_reads": 1, "leaked_reads": 0, "correct": 1,
                                     "recovery_of_classifiable": 1.0, "confusion": []}
        (outdir / sid / "classification" / f"{sid}.classified.json").write_text(json.dumps(summary))
        section = report_tools.build_section(summary, [], show_validation="--truth" in args)
        spliced = report_tools.splice(GOLDEN_REPORT.read_text(encoding="utf-8"), section)
        (outdir / sid / "report" / f"{sid}_AAV_report.contaminants.html").write_text(spliced, encoding="utf-8")


class FakeCursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, key, direction):
        self.docs = sorted(self.docs, key=lambda d: d[key], reverse=direction < 0)
        return self

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(copy.deepcopy(self.docs))


class FakeCollection:
    """The slice of the pymongo collection API that MongoRunStore uses."""

    def __init__(self):
        self.docs = {}

    def insert_one(self, doc):
        if doc["_id"] in self.docs:
            raise ValueError("duplicate _id")
        self.docs[doc["_id"]] = copy.deepcopy(doc)

    def find_one(self, flt):
        d = self.docs.get(flt["_id"])
        return copy.deepcopy(d) if d else None

    def update_one(self, flt, update):
        self.docs[flt["_id"]].update(copy.deepcopy(update["$set"]))

    def find(self, flt):
        return FakeCursor([d for d in self.docs.values() if all(d.get(k) == v for k, v in flt.items())])


class FakeMongoDB(dict):
    def __missing__(self, name):
        self[name] = FakeCollection()
        return self[name]


@pytest.fixture(params=["memory", "mongo", "file"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryRunStore()
    if request.param == "mongo":
        return MongoRunStore(FakeMongoDB())
    return FileRunStore(tmp_path / "lims")


@pytest.fixture
def env(tmp_path, store):
    fx = tmp_path / "fixtures"
    fx.mkdir()
    (fx / "ss.spikein.fastq.gz").write_bytes(b"reads")
    (fx / "vector.fa").write_text(">vec\nACGT\n")
    (fx / "vector.bed").write_text("vec\t0\t4\tITR\n")
    (fx / "packaging.fa").write_text(">pRep2Cap9 desc\nACGT\n>pHelper\nACGT\n>Lambda 48502\nACGT\n")
    (fx / "host.fa").write_text(">chr19\nACGT\n")
    (fx / "truth.tsv").write_text("read_id\ttaxid\n")
    laava = tmp_path / "laava"
    laava.mkdir()
    (laava / "main.nf").write_text("// stub")
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "contaminant_screen.nf").write_text("// stub")
    tax = tmp_path / "tax"
    tax.mkdir()
    (tax / "nodes.dmp").write_text("1\t|\t1\t|\tno rank\t|\n")
    (tmp_path / "cats.yaml").write_text("categories: []\n")
    demo = tmp_path / "blastdb"
    demo.mkdir()
    (demo / "demo_contaminants.nsq").write_bytes(b"x")
    (tmp_path / "nt").mkdir()  # configured but empty: unavailable
    settings = Settings(
        data_dir=tmp_path / "data", laava_dir=laava, screen_pipeline=pipeline / "contaminant_screen.nf",
        taxdump_dir=tax, categories=tmp_path / "cats.yaml",
        databases={
            "demo": DatabaseConfig("demo", demo, "demo_contaminants", "demo database"),
            "nt": DatabaseConfig("nt", tmp_path / "nt", "core_nt", "NCBI core_nt"),
        },
        default_database="demo", allowed_input_roots=[tmp_path / "data", laava, fx],
        nxf_syntax_parser="v1",
    )
    settings.data_dir.mkdir()
    runner = FakeRunner()
    ids = iter(f"run-{i:03d}" for i in range(1, 50))
    svc = AavService(settings, store, runner, run_id_factory=lambda: next(ids))
    return type("Env", (), {"settings": settings, "store": store, "runner": runner, "svc": svc, "fx": fx,
                            "tmp": tmp_path})()


def paths(listing):
    """File paths from a list_input_files result (entries are {"path", "what"} dicts)."""
    return [f["path"] for f in listing["files"]]


def spec_dict(env, **over):
    d = {"reads": str(env.fx / "ss.spikein.fastq.gz"), "vector_fa": str(env.fx / "vector.fa"),
         "vector_bed": str(env.fx / "vector.bed"), "vector_type": "ss",
         "packaging_fa": str(env.fx / "packaging.fa"), "host_fa": str(env.fx / "host.fa"),
         "repcap_name": "pRep2Cap9", "helper_name": "pHelper", "lambda_name": "Lambda"}
    d.update(over)
    return d


# ---------------------------------------------------------------- validation


def test_invalid_spec_reports_every_problem_and_creates_nothing(env):
    res = env.svc.run_laava(spec_dict(env, reads=str(env.fx / "missing.fastq"), vector_type="double"))
    assert res["status"] == "invalid"
    assert any("reads: file not found" in e for e in res["errors"])
    assert any("vector_type" in e for e in res["errors"])
    assert env.svc.list_runs() == []
    assert not env.settings.runs_dir.exists()
    assert env.runner.nextflow_calls() == []


def test_paths_outside_allowed_roots_are_refused(env, tmp_path):
    (tmp_path / "secret.txt").write_text("x")
    for bad in ("/etc/passwd", "../../etc/passwd", str(tmp_path / "secret.txt")):
        res = env.svc.run_laava(spec_dict(env, vector_fa=bad))
        assert res["status"] == "invalid"
        assert any("outside the folders" in e for e in res["errors"]), bad
    link = env.fx / "link.fa"
    link.symlink_to(tmp_path / "secret.txt")  # a symlink must not smuggle a file out
    res = env.svc.run_laava(spec_dict(env, vector_fa=str(link)))
    assert any("outside the folders" in e for e in res["errors"])


def test_relative_paths_resolve_against_data_dir(env):
    (env.settings.data_dir / "reads.fastq").write_text("x")
    assert env.settings.resolve_input("reads.fastq") == (env.settings.data_dir / "reads.fastq").resolve()
    with pytest.raises(PathNotAllowed):
        env.settings.resolve_input("../elsewhere/reads.fastq")  # climbs out of data_dir into no allowed root
    # ...whereas climbing into a different allowed root is fine
    assert env.settings.resolve_input("../fixtures/ss.spikein.fastq.gz") == (env.fx / "ss.spikein.fastq.gz").resolve()


def test_sample_id_default_and_checks(env):
    spec = RunSpec.from_dict(spec_dict(env))
    assert validate_run_spec(spec, env.settings).sample_id == "ss.spikein"
    bad = RunSpec.from_dict(spec_dict(env, sample_id="../escape"))
    assert any("sample_id" in e for e in validate_run_spec(bad, env.settings).errors)


def test_from_dict_rejects_unknown_fields_and_coerces_numbers(env):
    with pytest.raises(SpecError) as e:
        RunSpec.from_dict(spec_dict(env, colour="red"))
    assert "unknown field" in str(e.value)
    spec = RunSpec.from_dict(spec_dict(env, target_gap_threshold="250", min_supp_joint_coverage="0.9"))
    assert spec.target_gap_threshold == 250 and spec.min_supp_joint_coverage == 0.9
    with pytest.raises(SpecError):
        RunSpec.from_dict(spec_dict(env, target_gap_threshold="lots"))
    res = env.svc.run_laava({"reads": "x"})  # missing required fields comes back as data, not an exception
    assert res["status"] == "invalid" and any("vector_fa is required" in m for m in res["errors"])


def test_invented_names_are_errors_that_list_the_real_ones(env):
    spec = RunSpec.from_dict(spec_dict(env, repcap_name="Repcap", helper_name="Helper", lambda_name="Lambda"))
    v = validate_run_spec(spec, env.settings)
    assert len([e for e in v.errors if "does not match any record" in e]) == 2  # Lambda is real, the others are not
    assert any("repcap_name='Repcap'" in e and "pRep2Cap9" in e and "pHelper" in e for e in v.errors)
    assert any("helper_name='Helper'" in e for e in v.errors)  # "Helper" is not "pHelper", though it appears inside it
    res = env.svc.run_laava(spec_dict(env, helper_name="pAV_CMV_GFP"))
    assert res["status"] == "invalid" and env.svc.list_runs() == []  # nothing ran


def test_name_matching_accepts_ids_prefixes_and_header_words_only():
    from aav_agent.inputs import name_matches_header

    assert name_matches_header("pHelper", "pHelper")  # exact id
    assert name_matches_header("pHelper", "pHelper_v2 adenovirus helper")  # leading part of the id
    assert name_matches_header("Lambda", "NC_001416.1 Lambda phage 48502")  # another word of the header
    assert not name_matches_header("Helper", "pHelper")  # merely inside a longer word
    assert not name_matches_header("Repcap", "pRep2Cap9")
    assert not name_matches_header("x", "")


def test_names_without_a_packaging_file_and_flipflop_mismatch_only_warn(env):
    v = validate_run_spec(RunSpec.from_dict(spec_dict(env, packaging_fa=None, host_fa=None, flipflop_name="AAV2")), env.settings)
    assert v.errors == [] and len(v.warnings) == 2


def test_bed_label_mismatch_is_an_error_listing_the_real_names(env):
    v = validate_run_spec(RunSpec.from_dict(spec_dict(env, itr_label_1="ITR1", mitr_label="MITR")), env.settings)
    assert any("itr_label_1='ITR1'" in e and "names: ITR" in e for e in v.errors)
    assert any("mitr_label='MITR'" in e for e in v.errors)
    assert not any("itr_label_2" in e for e in v.errors)  # the default "ITR" is a real name in this BED
    res = env.svc.run_laava(spec_dict(env, mitr_label="MITR"))
    assert res["status"] == "invalid" and env.svc.list_runs() == []  # nothing ran


def test_the_annotation_must_describe_the_vector_construct(env):
    (env.fx / "other_construct.fa").write_text(">pAV_CMV_GFP\nACGT\n")
    v = validate_run_spec(RunSpec.from_dict(spec_dict(env, vector_fa=str(env.fx / "other_construct.fa"))), env.settings)
    assert any("describes vec" in e and "pAV_CMV_GFP" in e and "do not belong together" in e for e in v.errors)
    assert validate_run_spec(RunSpec.from_dict(spec_dict(env)), env.settings).errors == []  # a matching pair is fine


# ---------------------------------------------------------------- run_laava


def test_run_laava_success(env):
    res = env.svc.run_laava(spec_dict(env))
    assert res["status"] == "succeeded" and res["run_id"] == "run-001" and res["sample_id"] == "ss.spikein"
    assert res["warnings"] == []
    assert res["summary"]["total_reads"] == GOLDEN_TOTAL_READS
    assert res["summary"]["unmapped_reads"] == GOLDEN_UNMAPPED
    assert res["summary"]["by_reference_label"]["vector"] == GOLDEN_TOTAL_READS - GOLDEN_UNMAPPED
    assert Path(res["outputs"]["report_html"]).exists()

    run_dir = env.settings.runs_dir / "run-001"
    params = json.loads((run_dir / "params.json").read_text())
    assert params["sample_unique_id"] == "ss.spikein"
    assert all(Path(params[k]).is_absolute() for k in ("seq_reads_file", "vector_fa", "vector_bed", "host_fa"))
    assert params["repcap_name"] == "pRep2Cap9"

    call = env.runner.nextflow_calls()[0]
    a = call["args"]
    assert a[:4] == ["nextflow", "run", "-profile", "local"]
    assert a[a.index("--output") + 1] == str(run_dir / "output")
    assert a[a.index("-w") + 1] == str(run_dir / "work")
    assert a[a.index("--memory") + 1] == "6 GB"  # one argv element, no shell quoting needed
    assert a[a.index("--container_version") + 1] == ":latest"
    assert call["cwd"] == run_dir and call["env"] == {"NXF_SYNTAX_PARSER": "v1"}

    doc = env.store.get_run("run-001")
    assert doc["status"] == "succeeded" and doc["finished_utc"]
    reads_record = next(r for r in doc["inputs"] if r["role"] == "reads")
    assert reads_record["sha256"] == hashlib.sha256(b"reads").hexdigest()
    prov = doc["provenance"]
    assert prov["laava"]["git_commit"] == "abc123" and prov["laava"]["git_dirty"] is False
    assert prov["laava"]["image"]["id"] == "sha256:deadbeef"
    assert prov["nextflow_version"] == "25.10.0" and prov["tools_image"]["repo_digests"]


def test_failed_run_is_recorded_with_log_tail(env):
    env.runner.nf_returncode = 1
    res = env.svc.run_laava(spec_dict(env))
    assert res["status"] == "failed" and "code 1" in res["error"] and "something broke" in res["log_tail"]
    doc = env.store.get_run(res["run_id"])
    assert doc["status"] == "failed" and "something broke" in doc["log_tail"]
    assert env.svc.screen_contaminants(res["run_id"])["status"] == "error"  # failed runs cannot be screened


def test_missing_outputs_count_as_failure(env):
    env.runner.laava_outputs = False
    res = env.svc.run_laava(spec_dict(env))
    assert res["status"] == "failed" and "missing" in res["error"] and "report_html" in res["error"]


def test_unexpected_exception_marks_run_failed_then_propagates(env):
    env.runner.raise_on_nextflow = RuntimeError("disk on fire")
    with pytest.raises(RuntimeError):
        env.svc.run_laava(spec_dict(env))
    assert env.store.get_run("run-001")["status"] == "failed"


def test_provenance_degrades_gracefully(env):
    env.runner.git_ok = False
    env.runner.docker_ok = False
    res = env.svc.run_laava(spec_dict(env))
    assert res["status"] == "succeeded"
    prov = env.store.get_run(res["run_id"])["provenance"]
    assert prov["laava"]["git_commit"] is None and prov["laava"]["image"] is None
    assert prov["tools_image"] is None


# ---------------------------------------------------------------- screen_contaminants


def test_screen_success_and_repeat(env):
    run_id = env.svc.run_laava(spec_dict(env))["run_id"]
    res = env.svc.screen_contaminants(run_id)
    assert res["status"] == "succeeded" and res["screen_id"] == "screen-001" and res["database"] == "demo"
    assert res["buckets"] == {"enterobacteria": 1}
    assert res["database_recorded"]["db_title"] == "AAV demo contaminants"
    assert "validation" not in res
    html = Path(res["report_html"]).read_text(encoding="utf-8")
    assert html.count('id="contaminant-screening" class') == 1
    assert html.index('id="contaminant-screening"') < html.index('<div id="methods"')  # real report structure

    a = env.runner.nextflow_calls()[-1]["args"]
    assert a[a.index("--sample_id") + 1] == "ss.spikein"
    assert a[a.index("--blast_db_dir") + 1] == str(env.tmp / "blastdb")
    assert a[a.index("--blast_db_name") + 1] == "demo_contaminants"
    assert "--truth" not in a
    assert env.store.get_run(run_id)["latest_screen"] == "screen-001"

    assert env.svc.screen_contaminants(run_id)["screen_id"] == "screen-002"  # re-screen, new folder
    shown = env.svc.get_run(run_id)
    assert [s["screen_id"] for s in shown["screens"]] == ["screen-001", "screen-002"]


def test_screen_with_truth_reports_validation(env):
    run_id = env.svc.run_laava(spec_dict(env))["run_id"]
    res = env.svc.screen_contaminants(run_id, truth=str(env.fx / "truth.tsv"))
    assert res["status"] == "succeeded" and res["validation"]["recovery_of_classifiable"] == 1.0
    a = env.runner.nextflow_calls()[-1]["args"]
    assert "--truth" in a and "--show_validation" in a


def test_screen_errors_are_data_with_next_steps(env, tmp_path):
    run_id = env.svc.run_laava(spec_dict(env))["run_id"]
    unknown_run = env.svc.screen_contaminants("nope")
    assert unknown_run["status"] == "error" and run_id in unknown_run["recent_runs"]
    unknown_db = env.svc.screen_contaminants(run_id, database="genbank")
    assert unknown_db["status"] == "error" and unknown_db["available_databases"] == ["demo", "nt"]
    unavailable = env.svc.screen_contaminants(run_id, database="nt")
    assert unavailable["status"] == "error" and unavailable["available_databases"] == ["demo"]
    outside = env.svc.screen_contaminants(run_id, truth="/etc/passwd")
    assert outside["status"] == "error" and "outside the folders" in outside["message"]
    assert env.store.list_screens(run_id) == []  # none of the refusals left a record behind


def test_failed_screen_is_recorded(env):
    run_id = env.svc.run_laava(spec_dict(env))["run_id"]
    env.runner.nf_returncode = 1
    res = env.svc.screen_contaminants(run_id)
    assert res["status"] == "failed" and "something broke" in res["log_tail"]
    assert env.store.list_screens(run_id)[0]["status"] == "failed"


# ---------------------------------------------------------------- read tools


def test_list_databases(env):
    dbs = {d["name"]: d for d in env.svc.list_databases()}
    assert dbs["demo"]["available"] and dbs["demo"]["default"]
    assert not dbs["nt"]["available"] and not dbs["nt"]["default"]


def test_get_run_compact_vs_detail_and_listing(env):
    run_id = env.svc.run_laava(spec_dict(env))["run_id"]
    compact = env.svc.get_run(run_id)
    assert compact["id"] == run_id and "_id" not in compact and "provenance" not in compact
    assert "provenance" in env.svc.get_run(run_id, detail=True)
    missing = env.svc.get_run("nope")
    assert missing["status"] == "error" and run_id in missing["recent_runs"]
    second = env.svc.run_laava(spec_dict(env))["run_id"]
    rows = env.svc.list_runs()
    assert [r["run_id"] for r in rows] == [second, run_id] or {r["run_id"] for r in rows} == {run_id, second}
    assert rows[0]["unmapped_reads"] == 1


# ---------------------------------------------------------------- stores and settings


def test_store_contract(store):
    store.create_run({"_id": "a", "created_utc": "2026-10-01T00:00:00Z", "status": "running"})
    store.create_run({"_id": "b", "created_utc": "2026-10-02T00:00:00Z", "status": "running"})
    store.update_run("a", {"status": "succeeded", "summary": {"n": 1}})
    got = store.get_run("a")
    got["status"] = "tampered"  # mutating a returned document must not change the stored one
    assert store.get_run("a")["status"] == "succeeded" and store.get_run("a")["summary"] == {"n": 1}
    assert store.get_run("zzz") is None
    assert [r["_id"] for r in store.list_runs()] == ["b", "a"]
    assert [r["_id"] for r in store.list_runs(limit=1)] == ["b"]
    store.create_screen({"_id": "a:s2", "run_id": "a", "created_utc": "2026-10-03T00:00:00Z", "status": "running"})
    store.create_screen({"_id": "a:s1", "run_id": "a", "created_utc": "2026-10-02T00:00:00Z", "status": "running"})
    store.create_screen({"_id": "b:s1", "run_id": "b", "created_utc": "2026-10-02T00:00:00Z", "status": "running"})
    store.update_screen("a:s1", {"status": "succeeded"})
    screens = store.list_screens("a")
    assert [s["_id"] for s in screens] == ["a:s1", "a:s2"] and screens[0]["status"] == "succeeded"


def test_settings_from_yaml_resolves_relative_paths(tmp_path):
    cfg = tmp_path / "config" / "settings.yaml"
    cfg.parent.mkdir()
    cfg.write_text(
        "data_dir: ../data\nlaava_dir: ../../laava\nscreen_pipeline: ../pipeline/contaminant_screen.nf\n"
        "taxdump_dir: ../data/taxonomy\ncategories: categories.yaml\ndefault_database: demo\n"
        "databases:\n  demo: {dir: ../data/blastdb, db_name: demo_contaminants, description: demo}\n"
        "nxf_syntax_parser: v1\ncpus: 8\n"
    )
    s = Settings.from_yaml(cfg)
    assert s.data_dir == (tmp_path / "data").resolve()
    assert s.categories == (cfg.parent / "categories.yaml").resolve()
    assert s.databases["demo"].dir == (tmp_path / "data" / "blastdb").resolve()
    assert s.cpus == 8 and s.nxf_syntax_parser == "v1"
    assert s.allowed_input_roots == [s.data_dir, s.laava_dir]  # sensible default
    with pytest.raises(ValueError):
        Settings(data_dir=tmp_path, laava_dir=tmp_path, screen_pipeline=tmp_path, taxdump_dir=tmp_path,
                 categories=tmp_path, databases={}, default_database="missing")


# ---------------------------------------------------------------- background jobs


def _wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_start_laava_returns_immediately_and_wait_for_job_polls(env):
    env.runner.hold = threading.Event()
    res = env.svc.start_laava(spec_dict(env))
    assert res["status"] == "queued" and "wait_for_job" in res["next"]
    run_id = res["run_id"]
    assert _wait_until(lambda: env.svc.job_status(run_id)["status"] == "running")
    waiting = env.svc.wait_for_job(run_id, timeout_s=0.2, poll_s=0.05)
    assert waiting["status"] == "running" and "again" in waiting["message"] and waiting["waited_s"] >= 0.1
    env.runner.hold.set()
    done = env.svc.wait_for_job(run_id, timeout_s=10, poll_s=0.05)
    assert done["status"] == "succeeded" and done["summary"]["unmapped_reads"] == GOLDEN_UNMAPPED
    assert "message" not in done


def test_start_with_invalid_spec_is_immediate_and_creates_nothing(env):
    res = env.svc.start_laava(spec_dict(env, vector_type="nope"))
    assert res["status"] == "invalid" and env.svc.list_runs() == []


def test_jobs_run_one_at_a_time(env):
    env.runner.hold = threading.Event()
    first = env.svc.start_laava(spec_dict(env))["run_id"]
    second = env.svc.start_laava(spec_dict(env))["run_id"]
    assert _wait_until(lambda: env.svc.job_status(first)["status"] == "running")
    assert env.svc.job_status(second)["status"] == "queued"
    env.runner.hold.set()
    assert env.svc.wait_for_job(first, timeout_s=10, poll_s=0.05)["status"] == "succeeded"
    assert env.svc.wait_for_job(second, timeout_s=10, poll_s=0.05)["status"] == "succeeded"
    assert env.runner.max_active == 1


def test_start_screen_and_wait(env):
    run_id = env.svc.run_laava(spec_dict(env))["run_id"]
    env.runner.hold = threading.Event()
    first = env.svc.start_screen(run_id)
    second = env.svc.start_screen(run_id)
    assert (first["status"], first["screen_id"]) == ("queued", "screen-001")
    assert second["screen_id"] == "screen-002"  # numbered atomically even when queued back to back
    env.runner.hold.set()
    done = env.svc.wait_for_job(run_id, "screen-001", timeout_s=10, poll_s=0.05)
    assert done["status"] == "succeeded" and done["buckets"] == {"enterobacteria": 1} and done["database"] == "demo"
    assert env.svc.wait_for_job(run_id, "screen-002", timeout_s=10, poll_s=0.05)["status"] == "succeeded"
    missing = env.svc.job_status(run_id, "screen-009")
    assert missing["status"] == "error" and missing["screens"] == ["screen-001", "screen-002"]


def test_background_crash_is_visible_not_a_hang(env):
    env.runner.raise_on_nextflow = RuntimeError("disk on fire")
    res = env.svc.start_laava(spec_dict(env))
    done = env.svc.wait_for_job(res["run_id"], timeout_s=10, poll_s=0.05)
    assert done["status"] == "failed" and "disk on fire" in done["error"]


def test_waiting_on_an_unknown_job_returns_at_once(env):
    t0 = time.monotonic()
    res = env.svc.wait_for_job("nope", timeout_s=30)
    assert res["status"] == "error" and time.monotonic() - t0 < 2


def test_recover_interrupted_marks_stale_records_failed(env):
    for rid, status, day in (("old1", "running", "01"), ("old2", "queued", "02"), ("ok", "succeeded", "03")):
        env.store.create_run({"_id": rid, "type": "laava", "status": status, "created_utc": f"2026-01-{day}T00:00:00Z"})
    env.store.create_screen({"_id": "ok:screen-001", "run_id": "ok", "screen_id": "screen-001",
                             "status": "running", "created_utc": "2026-01-03T00:00:01Z", "database": {"name": "demo"}})
    assert env.svc.recover_interrupted() == 3
    assert env.store.get_run("old1")["status"] == "failed" and "interrupted" in env.store.get_run("old1")["error"]
    assert env.store.get_run("old2")["status"] == "failed"
    assert env.store.get_run("ok")["status"] == "succeeded"
    assert env.store.list_screens("ok")[0]["status"] == "failed"


def test_file_store_survives_restart_and_rejects_unsafe_ids(tmp_path):
    a = FileRunStore(tmp_path)
    a.create_run({"_id": "r1", "created_utc": "2026-01-01T00:00:00Z", "status": "running"})
    a.update_run("r1", {"status": "succeeded"})
    assert FileRunStore(tmp_path).get_run("r1")["status"] == "succeeded"  # a new process sees it
    with pytest.raises(ValueError):
        a.get_run("../escape")
    with pytest.raises(ValueError):
        a.create_run({"_id": "r1", "created_utc": "2026-01-02T00:00:00Z", "status": "running"})
    assert not list(tmp_path.glob("*.tmp")) and not list((tmp_path / "runs").glob("*.tmp"))


def test_list_input_files_finds_real_paths_and_skips_work_folders(env):
    (env.fx / "sub").mkdir()
    (env.fx / "sub" / "b.fastq.gz").write_bytes(b"x")
    (env.fx / "work").mkdir()
    (env.fx / "work" / "hidden.fastq.gz").write_bytes(b"x")
    res = env.svc.list_input_files("*.fastq.gz")
    names = [Path(p).name for p in paths(res)]
    assert names == ["ss.spikein.fastq.gz", "b.fastq.gz"] or sorted(names) == ["b.fastq.gz", "ss.spikein.fastq.gz"]
    assert "hidden.fastq.gz" not in names and not res["truncated"]
    assert all(Path(p).is_relative_to(r) for p in paths(res) for r in env.settings.allowed_input_roots
               if str(p).startswith(str(r)))
    limited = env.svc.list_input_files("*.fastq.gz", limit=1)
    assert len(limited["files"]) == 1 and limited["truncated"]
    assert env.svc.list_input_files("*.nothing")["files"] == []


def test_list_input_files_by_kind_pattern_and_folder(env):
    (env.settings.laava_dir / "test" / "fasta").mkdir(parents=True)
    for name in ("packaging.fa", "hg38.chr19trunc-chrM.fa", "AAV2.flipflop.fasta", "notes.txt"):
        (env.settings.laava_dir / "test" / "fasta" / name).write_text(">x\nACGT\n")
    # kind="fasta" covers .fa and .fasta alike: the extension need not be guessed
    res = env.svc.list_input_files(kind="fasta", folder="laava/test")
    names = sorted(Path(p).name for p in paths(res))
    assert res["status"] == "ok" and names == ["AAV2.flipflop.fasta", "hg38.chr19trunc-chrM.fa", "packaging.fa"]
    assert res["folders_searched"] == [str((env.settings.laava_dir / "test").resolve())]
    # comma-separated patterns, case-insensitive, combined with a folder
    by_name = env.svc.list_input_files(pattern="*PACKAGING*,*.flipflop.*", folder="laava/test/fasta")
    assert sorted(Path(p).name for p in paths(by_name)) == ["AAV2.flipflop.fasta", "packaging.fa"]
    # reads / bed kinds, and a folder given as an absolute path
    assert [Path(p).name for p in paths(env.svc.list_input_files(kind="reads", folder=str(env.fx)))] == ["ss.spikein.fastq.gz"]
    assert [Path(p).name for p in paths(env.svc.list_input_files(kind="bed"))] == ["vector.bed"]
    # previous run outputs and work folders are not inputs
    (env.settings.runs_dir / "r1" / "output").mkdir(parents=True)
    (env.settings.runs_dir / "r1" / "output" / "unmapped.fa").write_text(">x\n")
    assert "unmapped.fa" not in [Path(p).name for p in paths(env.svc.list_input_files(kind="fasta"))]


def test_list_input_files_errors_are_data_with_hints(env):
    missing = env.svc.list_input_files(folder="no/such/place")
    assert missing["status"] == "error" and missing["folders_searched"]
    outside = env.svc.list_input_files(folder="/etc")
    assert outside["status"] == "error"  # not under an allowed folder


def test_common_words_for_a_file_type_are_understood_and_unknown_ones_are_ignored_not_fatal(env):
    """In a transcript the model asked for kind="fastq" five times and gave up when each was refused."""
    reads = paths(env.svc.list_input_files(kind="reads", folder=str(env.fx)))
    assert reads and reads[0].endswith("ss.spikein.fastq.gz")
    for word in ("fastq", "FASTQ", ".fastq", "fq", "fastq.gz", "*.fastq.gz", "sequencing reads", " read "):
        res = env.svc.list_input_files(kind=word, folder=str(env.fx))
        assert paths(res) == reads and "note" not in res, word
    assert [Path(p).name for p in paths(env.svc.list_input_files(kind="fa", folder=str(env.fx)))] == sorted(
        Path(p).name for p in paths(env.svc.list_input_files(kind="fasta", folder=str(env.fx))))
    assert [Path(p).name for p in paths(env.svc.list_input_files(kind="annotation"))] == ["vector.bed"]
    assert [Path(p).name for p in paths(env.svc.list_input_files(kind="ubam", folder=str(env.fx)))] == []

    # an unrecognised word is dropped, with a note, instead of failing the search
    res = env.svc.list_input_files(kind="spreadsheet", query="spike", folder=str(env.fx))
    assert res["status"] == "ok" and "'spreadsheet' is not recognised and was ignored" in res["note"]
    assert "bam, bed, fasta, reads" in res["note"] and paths(res)[0].endswith("ss.spikein.fastq.gz")
    # and the note is kept when the search is relaxed too
    both = env.svc.list_input_files(kind="spreadsheet", pattern="*zebrafish*", folder=str(env.fx))
    assert both["files"] == [] or "is not recognised" in both["note"]


def test_describe_input_reads_names_from_the_files(env):
    fasta = env.svc.describe_input(str(env.fx / "packaging.fa"))
    assert fasta["status"] == "ok" and fasta["type"] == "fasta" and fasta["n_records"] == 3
    assert [r["id"] for r in fasta["records"]] == ["pRep2Cap9", "pHelper", "Lambda"]
    assert fasta["records"][0]["length"] == 4 and not fasta["truncated"]

    bed = env.svc.describe_input(str(env.fx / "vector.bed"))
    assert bed["type"] == "bed" and bed["feature_names"] == {"ITR": 1} and bed["first_rows"][0]["name"] == "ITR"

    reads = env.svc.describe_input(str(env.fx / "ss.spikein.fastq.gz"))
    assert reads["type"] == "reads" and reads["format"] == "fastq"
    (env.fx / "notes.txt").write_text("hi")
    assert env.svc.describe_input(str(env.fx / "notes.txt"))["type"] == "other"
    assert env.svc.describe_input(str(env.fx / "missing.fa"))["status"] == "error"
    assert "outside the folders" in env.svc.describe_input("/etc/passwd")["message"]


def test_describe_input_truncates_long_fasta_listings_and_handles_headerless_beds(env):
    big = env.fx / "many.fa"
    big.write_text("".join(f">rec{i}\nACGTACGT\n" for i in range(120)))
    res = env.svc.describe_input(str(big))
    assert res["n_records"] == 120 and len(res["records"]) == 50 and res["truncated"]
    plain = env.fx / "plain.bed"
    plain.write_text("chr1\t0\t100\n# comment\nchr1\t200\t300\n")
    res = env.svc.describe_input(str(plain))
    assert res["n_features"] == 2 and res["feature_names"] == {} and "no name column" in res["note"]


def test_kind_and_pattern_narrow_each_other_instead_of_adding_up(env):
    (env.settings.laava_dir / "test" / "fasta").mkdir(parents=True)
    for name in ("packaging.fa", "hg38.chr19trunc-chrM.fa", "ss.construct.fasta"):
        (env.settings.laava_dir / "test" / "fasta" / name).write_text(">x\nACGT\n")
    only_packaging = env.svc.list_input_files(kind="fasta", pattern="*packaging*", folder="laava/test")
    assert [Path(p).name for p in paths(only_packaging)] == ["packaging.fa"]
    host_or_construct = env.svc.list_input_files(kind="fasta", pattern="*hg38*, *ss.construct*", folder="laava/test")
    assert sorted(Path(p).name for p in paths(host_or_construct)) == ["hg38.chr19trunc-chrM.fa", "ss.construct.fasta"]
    # the file name matches the pattern but is not of the requested kind: excluded
    assert env.svc.list_input_files(kind="reads", pattern="*construct*", folder="laava/test")["files"] == []  # no reads there
    assert [Path(p).name for p in paths(env.svc.list_input_files(kind="reads", pattern="*spike*"))] == ["ss.spikein.fastq.gz"]


# ---------------------------------------------------------------- sanity checks on LAAVA results


def _per_read(n_other_vector, n_unmapped=0, n_good=0):
    head = "read_id\thas_primary\thas_supp\tassigned_type\tassigned_subtype\teffective_count\treference_label\tread_target_overlap\n"
    rows = [f"o{i}\tTrue\tFalse\tother-vector\tNA\t1\tvector\t0.1\n" for i in range(n_other_vector)]
    rows += [f"u{i}\tFalse\tFalse\tNA\tNA\t1\t(unmapped)\t0\n" for i in range(n_unmapped)]
    rows += [f"g{i}\tTrue\tFalse\tssAAV\tfull\t1\tvector\t1.0\n" for i in range(n_good)]
    return head + "".join(rows)


def test_assess_summary_flags_inputs_that_probably_do_not_belong_together():
    from aav_agent.service import assess_summary

    assert assess_summary({"total_reads": 0}) == ["LAAVA found no reads in the input."]
    healthy = {"total_reads": 3627, "unmapped_reads": 130, "by_assigned_type": {"ssAAV": 2032, "other-vector": 1367}}
    assert assess_summary(healthy) == []  # the real spike-in sample: 38% other-vector is normal
    all_other = {"total_reads": 2000, "unmapped_reads": 0, "by_assigned_type": {"other-vector": 2000}}
    (w,) = assess_summary(all_other)
    assert "2,000 of 2,000 reads (100%)" in w and "belong together" in w
    assert assess_summary({"total_reads": 100, "unmapped_reads": 0, "by_assigned_type": {"other-vector": 89}}) == []
    assert len(assess_summary({"total_reads": 100, "unmapped_reads": 0, "by_assigned_type": {"other-vector": 90}})) == 1
    (w,) = assess_summary({"total_reads": 100, "unmapped_reads": 60, "by_assigned_type": {"NA": 60, "ssAAV": 40}})
    assert "60 of 100 reads (60%) did not map" in w


def test_a_run_whose_reads_are_all_other_vector_succeeds_but_warns(env):
    env.runner.per_read_text = _per_read(n_other_vector=200)
    res = env.svc.run_laava(spec_dict(env))
    assert res["status"] == "succeeded"
    assert len(res["warnings"]) == 1 and "other-vector" in res["warnings"][0]
    assert env.store.get_run(res["run_id"])["warnings"] == res["warnings"]  # kept in the record
    assert env.svc.job_status(res["run_id"])["warnings"] == res["warnings"]  # and visible to pollers


def test_start_up_and_result_warnings_are_combined(env):
    env.runner.per_read_text = _per_read(n_other_vector=0, n_unmapped=80, n_good=20)
    res = env.svc.run_laava(spec_dict(env, packaging_fa=None, host_fa=None, repcap_name="pRep2Cap9"))
    assert res["status"] == "succeeded" and len(res["warnings"]) == 2
    assert any("without a packaging_fa" in w for w in res["warnings"]) and any("did not map" in w for w in res["warnings"])


# ---------------------------------------------------------------- file descriptions


def test_listing_describes_each_file_by_content(env):
    (env.fx / "hg38.chr19trunc-chrM.fa").write_text(">chr19 truncated\nACGT\n>chrM\nACGT\n")
    (env.fx / "mm10.fa").write_text(">chr1\nACGT\n>chrX\nACGT\n")
    (env.fx / "humanized_vector.fa").write_text(">vec2\nACGT\n")
    listing = env.svc.list_input_files(kind="fasta", folder=str(env.fx))
    what = {Path(f["path"]).name: f["what"] for f in listing["files"]}
    assert "3 records (pRep2Cap9, pHelper, Lambda)" in what["packaging.fa"]
    assert "plasmid sequences" in what["packaging.fa"] and "name suggests" not in what["packaging.fa"]
    assert "1 record (vec)" in what["vector.fa"] and "single short sequence" in what["vector.fa"]
    host = what["hg38.chr19trunc-chrM.fa"]
    assert "2 records (chr19, chrM)" in host and "candidate host genome" in host and "suggests human" in host
    assert "suggests mouse" in what["mm10.fa"] and "candidate host genome" in what["mm10.fa"]
    assert "suggests" not in what["humanized_vector.fa"]  # "humanized" is not "human"
    beds = {Path(f["path"]).name: f["what"] for f in env.svc.list_input_files(kind="bed")["files"]}
    assert beds["vector.bed"] == "BED, 1 features on vec (ITR)"
    reads = env.svc.list_input_files(kind="reads", folder=str(env.fx))["files"][0]["what"]
    assert reads.startswith("reads (FASTQ)")


def test_descriptions_cope_with_big_unreadable_and_unknown_files(env):
    from aav_agent.inputs import fasta_profile, fasta_role_hint, summarize_file

    big = env.fx / "genome.fa"
    big.write_text("".join(f">chr{i}\n" + "ACGT" * 50 + "\n" for i in range(1, 8)))
    prof = fasta_profile(big, scan_limit=100)  # pretend it is too big to scan
    assert prof["scanned"] is False and prof["first_ids"] == ["chr1", "chr2", "chr3", "chr4"]
    assert "candidate host genome" in fasta_role_hint(prof)  # chromosome-like names are enough
    assert fasta_role_hint({"first_ids": ["x"], "n": 1, "total": 10**6, "scanned": True}) == ""  # nothing to say

    broken = env.fx / "broken.fa"
    broken.write_bytes(b"\xff\xfe>x\nACGT\n")
    assert summarize_file(broken).startswith("could not be read")
    (env.fx / "notes.txt").write_text("hi")
    assert summarize_file(env.fx / "notes.txt") == ""  # unknown types: no description
    listing = env.svc.list_input_files(pattern="broken.fa", folder=str(env.fx))
    assert paths(listing) == [str(broken)]  # one unreadable file does not break the listing


# ---------------------------------------------------------------- finding files from the user's own words


def _lab(env):
    """A small version of the real layout: LAAVA test files, plus the spike-in reads under the data folder."""
    base = env.settings.laava_dir / "test"
    (base / "fasta").mkdir(parents=True)
    (base / "samples").mkdir()
    (base / "fasta" / "hg38.chr19trunc-chrM.fa").write_text(">chr19\nACGT\n>chrM\nACGT\n")
    (base / "fasta" / "packaging.fa").write_text(">pRep2Cap9\nACGT\n>pHelper\nACGT\n")
    (base / "samples" / "ss.construct.fasta").write_text(">pAV_CMV_GFP\nACGT\n")
    (base / "samples" / "ss.annotation.bed").write_text("pAV_CMV_GFP\t0\t4\tITR\n")
    (base / "samples" / "ss.subsample005.bam").write_bytes(b"BAM")
    (env.settings.data_dir / "ss.spikein.fastq.gz").write_bytes(b"reads")  # the spike-in the user means
    (env.settings.data_dir / "laava_out").mkdir()
    (env.settings.data_dir / "laava_out" / "ss.spikein.sort_by_name.bam").write_bytes(b"BAM")  # an old output


def test_query_ranks_files_by_the_users_own_words(env):
    _lab(env)
    res = env.svc.list_input_files(query="ss spike-in fastq")  # no folder: look everywhere
    first = res["files"][0]
    assert Path(first["path"]).name == "ss.spikein.fastq.gz" and first["matches"] == ["ss", "spike", "fastq"]
    ranked = [Path(f["path"]).name for f in res["files"]]
    assert ranked.index("ss.spikein.sort_by_name.bam") > 0 and "ss.subsample005.bam" in ranked  # lesser matches follow
    assert res["patterns"]["query"] == ["ss", "spike", "fastq"]  # "in" is dropped as a stop word

    host = env.svc.list_input_files(query="human genome (hg38)", folder="laava/test")["files"][0]
    assert Path(host["path"]).name == "hg38.chr19trunc-chrM.fa" and {"human", "genome", "hg38"} <= set(host["matches"])
    typo = env.svc.list_input_files(query="human gneome", kind="fasta")["files"][0]  # one word is enough to rank it
    assert Path(typo["path"]).name == "hg38.chr19trunc-chrM.fa" and typo["matches"] == ["human"]
    pkg = env.svc.list_input_files(query="packaging", kind="fasta", folder="laava/test")["files"]
    assert [Path(f["path"]).name for f in pkg] == ["packaging.fa"]
    everywhere = env.svc.list_input_files(query="packaging", kind="fasta")["files"]  # also the copy under fixtures
    assert len(everywhere) == 2 and all(Path(f["path"]).name == "packaging.fa" for f in everywhere)
    bed = env.svc.list_input_files(query="annotation bed")["files"][0]
    assert Path(bed["path"]).name == "ss.annotation.bed"


def test_a_pattern_made_of_real_words_also_matches_the_description(env):
    _lab(env)
    # the failing case from the transcripts: the model searches for *human*, which is not in the file name
    res = env.svc.list_input_files(pattern="*human*", kind="fasta", folder="laava/test")
    assert [Path(f["path"]).name for f in res["files"]] == ["hg38.chr19trunc-chrM.fa"] and "note" not in res
    # the guess that failed before: *host* is not in any file name, but the genome is described as a host candidate
    host = env.svc.list_input_files(pattern="*host*", kind="fasta", folder="laava/test")
    assert [Path(f["path"]).name for f in host["files"]] == ["hg38.chr19trunc-chrM.fa"] and "note" not in host
    # a short pattern such as *ss* is a name filter only, so it cannot match random description text
    names = [Path(f["path"]).name for f in env.svc.list_input_files(pattern="*ss*", kind="fasta", folder="laava/test")["files"]]
    assert names == ["ss.construct.fasta"]


def test_a_search_that_finds_nothing_falls_back_to_the_kind_and_folder_with_a_note(env):
    _lab(env)
    res = env.svc.list_input_files(pattern="*zebrafish*", kind="fasta", folder="laava/test")
    assert sorted(Path(f["path"]).name for f in res["files"]) == ["hg38.chr19trunc-chrM.fa", "packaging.fa", "ss.construct.fasta"]
    assert "Nothing matched pattern '*zebrafish*'" in res["note"] and "choose by the description" in res["note"].lower()
    q = env.svc.list_input_files(query="zebrafish", kind="fasta", folder="laava/test")
    assert len(q["files"]) == 3 and "Nothing matched query 'zebrafish'" in q["note"]
    # with no kind or folder to fall back to, an empty search stays empty and says nothing misleading
    nothing = env.svc.list_input_files(query="zebrafish")
    assert nothing["files"] == [] and "note" not in nothing
    # an unfiltered listing has no note either
    assert "note" not in env.svc.list_input_files(kind="fasta", folder="laava/test")


# ---------------------------------------------------------------- labels and mitr


def test_mitr_label_is_refused_for_ss_but_allowed_for_sc(env):
    ss = validate_run_spec(RunSpec.from_dict(spec_dict(env, mitr_label="ITR")), env.settings)
    assert any("mitr_label='ITR' is only for self-complementary (sc) vectors" in e for e in ss.errors)
    assert not any("is not a feature name" in e for e in ss.errors)  # one clear message, not two
    sc = validate_run_spec(RunSpec.from_dict(spec_dict(env, vector_type="sc", mitr_label="ITR")), env.settings)
    assert sc.errors == []


def test_packaging_records_keep_their_own_names_and_that_is_not_a_warning():
    """The agent never sets repcap/helper/lambda names, so packaging records are labelled with the names the
    researchers gave them in the FASTA. That is expected, not a sign of a mistake."""
    from aav_agent.service import assess_summary

    labels = {"vector": 3414, "pRep2Cap9": 27, "pHelper": 8, "Lambda": 2, "E1A_E1B": 4, "Fiber": 1, "host": 13,
              "chimeric-vector": 30, "(unmapped)": 130}
    summary = {"total_reads": 3629, "unmapped_reads": 130, "by_reference_label": labels,
               "by_assigned_type": {"ssAAV": 2032, "other-vector": 1367}}
    assert assess_summary(summary) == []
