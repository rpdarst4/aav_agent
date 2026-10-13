"""Tests for the tool layer. No Nextflow, Docker, Mongo or LLM needed: commands go through a
FakeRunner that creates the files the real pipelines would, and MongoRunStore is exercised
against a small fake collection."""

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from aav_agent.config import DatabaseConfig, PathNotAllowed, Settings
from aav_agent.lims.store import InMemoryRunStore, MongoRunStore
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
            if self.raise_on_nextflow:
                raise self.raise_on_nextflow
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("line 1\nline 2\nsomething broke\n" if self.nf_returncode else "ok\n")
            if self.nf_returncode == 0:
                script = next(a for a in args if a.endswith(".nf"))
                (self._fake_laava if script.endswith("main.nf") else self._fake_screen)(args)
            return CommandResult(self.nf_returncode)
        raise AssertionError(f"unexpected command: {args}")

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
        shutil.copy(GOLDEN_PER_READ, out / "report" / f"{sid}.per_read.tsv.gz")
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


@pytest.fixture(params=["memory", "mongo"])
def store(request):
    return InMemoryRunStore() if request.param == "memory" else MongoRunStore(FakeMongoDB())


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


def test_name_mismatch_is_a_warning_not_an_error(env):
    spec = RunSpec.from_dict(spec_dict(env, repcap_name="NotThere"))
    v = validate_run_spec(spec, env.settings)
    assert v.errors == []
    assert any("repcap_name='NotThere'" in w for w in v.warnings)


# ---------------------------------------------------------------- run_laava


def test_run_laava_success(env):
    res = env.svc.run_laava(spec_dict(env))
    assert res["status"] == "succeeded" and res["run_id"] == "run-001" and res["sample_id"] == "ss.spikein"
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
