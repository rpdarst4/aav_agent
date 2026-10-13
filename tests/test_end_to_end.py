"""End-to-end test with the real LAAVA and screening pipelines. Opt-in and slow.

Runs only when AAV_INTEGRATION=1 and Nextflow and a Docker daemon are available:

    AAV_INTEGRATION=1 AAV_SETTINGS=config/settings.yaml poetry run pytest -m integration -s

It reuses your real settings file (databases, taxdump, images) but writes runs to a temporary
data folder. The unit tests in tests/test_service.py use a fake runner, so they cannot notice
if LAAVA's output layout or behaviour changes. This test can.

Paths below follow the layout in params-local-ss.json; adjust them to your repo.
"""

import dataclasses
import gzip
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from aav_agent.config import Settings
from aav_agent.lims.store import InMemoryRunStore
from aav_agent.runner import SubprocessRunner
from aav_agent.service import AavService

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    if os.environ.get("AAV_INTEGRATION") != "1":
        pytest.skip("set AAV_INTEGRATION=1 to run the slow end-to-end test")
    if not shutil.which("nextflow") or not _docker_ok():
        pytest.skip("needs Nextflow and a running Docker daemon")
    base = Settings.from_yaml(os.environ.get("AAV_SETTINGS", REPO / "config" / "settings.yaml"))
    data = tmp_path_factory.mktemp("data")
    settings = dataclasses.replace(
        base, data_dir=data, allowed_input_roots=[data, base.laava_dir, REPO / "fixtures"]
    )
    return AavService(settings, InMemoryRunStore(), SubprocessRunner())


@pytest.fixture(scope="module")
def spec(service):
    laava = service.settings.laava_dir
    return {
        "reads": str(REPO / "fixtures/spike_in/ss.spikein.fastq.gz"),
        "vector_type": "ss",
        "vector_bed": str(laava / "test/samples/ss.annotation.bed"),
        "vector_fa": str(laava / "test/samples/ss.construct.fasta"),
        "packaging_fa": str(laava / "test/fasta/packaging.fa"),
        "host_fa": str(laava / "test/fasta/hg38.chr19trunc-chrM.fa"),
        "itr_label_1": "ITR", "itr_label_2": "ITR",
        "repcap_name": "pRep2Cap9", "helper_name": "pHelper", "lambda_name": "Lambda",
        "flipflop_name": "AAV2", "flipflop_fa": str(laava / "test/fasta/AAV2.flipflop.fasta"),
    }


@pytest.fixture(scope="module")
def run(service, spec):
    return service.run_laava(spec)


def _fastq_records(path: Path) -> int:
    with gzip.open(path, "rt") as fh:
        return sum(1 for _ in fh) // 4


def test_laava_run_succeeds_with_expected_layout(run):
    assert run["status"] == "succeeded", run.get("error") or run
    # These file names are exactly what the fake runner in the unit tests assumes.
    for key in ("name_sorted_bam", "per_read_tsv", "report_html"):
        assert Path(run["outputs"][key]).is_file(), key


def test_every_input_read_is_accounted_for(run, spec):
    assert run["summary"]["total_reads"] == _fastq_records(Path(spec["reads"]))
    assert run["summary"]["unmapped_reads"] > 0  # the spiked-in contaminants


def test_provenance_is_recorded(service, run):
    prov = service.store.get_run(run["run_id"])["provenance"]
    assert prov["laava"]["git_commit"], "LAAVA checkout is not a git repository?"
    assert prov["laava"]["image"] and prov["laava"]["image"]["id"]
    assert prov["nextflow_version"]


def test_screen_recovers_the_spiked_reads(service, run):
    truth = REPO / "fixtures/spike_in/ss.spikein.truth.tsv"
    res = service.screen_contaminants(run["run_id"], truth=str(truth))
    assert res["status"] == "succeeded", res.get("error") or res
    assert sum(res["buckets"].values()) == res["unmapped_reads_screened"] == run["summary"]["unmapped_reads"]
    assert res["validation"]["recovery_of_classifiable"] >= 0.95
    html = Path(res["report_html"]).read_text(encoding="utf-8")
    assert html.count('id="contaminant-screening"') == 1
    assert res["database_recorded"]["db_title"] in html  # the report names the database blastn actually used
