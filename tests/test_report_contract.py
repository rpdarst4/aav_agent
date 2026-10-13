"""The splice must work on the real structure of a LAAVA report (a trimmed real report is in
tests/fixtures). If a new LAAVA version changes the report's markup, regenerate the fixture with
scripts/make_report_skeleton.py and this test will say whether the splice anchors still exist."""

from pathlib import Path

from aav_agent.tools import add_contaminants_to_report as ac

SKELETON = (Path(__file__).parent / "fixtures" / "laava_report_skeleton.html").read_text(encoding="utf-8")
SUMMARY = {
    "total_reads": 20,
    "buckets": {
        "enterobacteria": {"reads": 7, "read_fraction": 0.35, "bases": 20000, "base_fraction": 0.4},
        "no hit": {"reads": 13, "read_fraction": 0.65, "bases": 30000, "base_fraction": 0.6},
    },
    "params": {"min_pident": 80.0, "min_aln_len": 100, "max_evalue": 1e-5, "lca_window": 0.02},
    "database": {"db_title": "AAV demo contaminants", "db_date": "Oct 7, 2026"},
}
ROWS = [{"lca_name": "Escherichia coli", "bucket": "enterobacteria"}] * 7 + [{"lca_name": "", "bucket": "no hit"}] * 13


def test_anchors_exist_in_real_report_structure():
    assert '<div id="methods" class="section level1">' in SKELETON
    assert '<li><a href="#methods"' in SKELETON


def test_splice_into_real_structure():
    out = ac.splice(SKELETON, ac.build_section(SUMMARY, ROWS, laava_total=3517))
    assert out.index('id="contaminant-screening"') < out.index('<div id="methods"')
    assert out.index("toc-contaminant-screening") < out.index('id="toc-methods"')
    assert (out.count("<div") - out.count("</div>")) == (SKELETON.count("<div") - SKELETON.count("</div>"))
    assert "AAV demo contaminants, Oct 7, 2026" in out
    again = ac.splice(out, ac.build_section(SUMMARY, ROWS, laava_total=3517))
    assert again.count('id="contaminant-screening" class') == 1 and len(again) == len(out)
