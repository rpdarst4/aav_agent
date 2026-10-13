"""Tests for database provenance in classify_reads / add_contaminants_to_report.
Adjust the imports to your package path (e.g. from aav_agent.tools import ...)."""

import json
import logging

import yaml

from aav_agent.tools import add_contaminants_to_report as ac
from aav_agent.tools import classify_reads as cr

SUMMARY = {
    "total_reads": 1,
    "params": {},
    "buckets": {"plant": {"reads": 1, "read_fraction": 1.0, "bases": 10, "base_fraction": 1.0}},
}
META = {"db_name": "demo_contaminants", "db_title": "AAV demo contaminants", "db_date": "Oct 7, 2026"}


def test_label_comes_from_recorded_database():
    label = ac.database_label({**SUMMARY, "database": META})
    assert label == "AAV demo contaminants, Oct 7, 2026"
    section = ac.build_section({**SUMMARY, "database": META}, [])
    assert "AAV demo contaminants, Oct 7, 2026" in section


def test_conflicting_label_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING):
        label = ac.database_label({**SUMMARY, "database": META}, "NCBI core_nt")
    assert label == "NCBI core_nt"  # the explicit override is honoured...
    assert "differs from the database recorded" in caplog.text  # ...but not silently


def test_matching_label_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        ac.database_label({**SUMMARY, "database": META}, "AAV demo contaminants")
    assert caplog.text == ""


def test_no_provenance_and_no_label_omits_database_text():
    assert ac.database_label(SUMMARY) == ""


def test_show_validation_without_results_warns(caplog):
    with caplog.at_level(logging.WARNING):
        ac.build_section(SUMMARY, [], show_validation=True)
    assert "re-run classify_reads.py with --truth" in caplog.text


def test_classify_records_meta_sidecar(tmp_path):
    (tmp_path / "taxdump").mkdir()
    (tmp_path / "taxdump/nodes.dmp").write_text("1\t|\t1\t|\tno rank\t|\n9606\t|\t1\t|\tspecies\t|\n")
    (tmp_path / "taxdump/names.dmp").write_text("9606\t|\tHomo sapiens\t|\t\t|\tscientific name\t|\n")
    (tmp_path / "cats.yaml").write_text(yaml.safe_dump({"categories": [{"name": "host", "taxids": [9606]}]}))
    (tmp_path / "q.fa").write_text(">a\n" + "A" * 300 + "\n")
    (tmp_path / "hits.tsv").write_text("a\ts\t99\t300\t1\t300\t1\t300\t0\t500\t9606\n")
    (tmp_path / "hits.tsv.meta.json").write_text(json.dumps(META))
    args = ["--blast", str(tmp_path / "hits.tsv"), "--queries", str(tmp_path / "q.fa"),
            "--taxdump", str(tmp_path / "taxdump"), "--categories", str(tmp_path / "cats.yaml"),
            "--out-tsv", str(tmp_path / "o.tsv"), "--out-summary", str(tmp_path / "o.json")]
    assert cr.main(args) == 0
    assert json.loads((tmp_path / "o.json").read_text())["database"]["db_title"] == "AAV demo contaminants"
    (tmp_path / "hits.tsv.meta.json").unlink()  # missing sidecar: still works, just no database record
    assert cr.main(args) == 0
    assert "database" not in json.loads((tmp_path / "o.json").read_text())
