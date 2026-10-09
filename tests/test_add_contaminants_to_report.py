import base64
import re

import pytest

from src.aav_agent.tools import add_contaminants_to_report as ac

REPORT = """<html><body>
<div id="TOC"><ul>
<li><a href="#flip" id="toc-flip">Flip/flop</a></li>
<li><a href="#methods" id="toc-methods">Methods</a></li>
</ul></div>
<div id="flip" class="section level1"><h1>Flip/flop</h1></div>
<div id="methods" class="section level1"><h1>Methods</h1></div>
</body></html>
"""

SUMMARY = {
    "total_reads": 4,
    "buckets": {
        "enterobacteria": {"reads": 3, "read_fraction": 0.75, "bases": 3000, "base_fraction": 0.8},
        "no hit": {"reads": 1, "read_fraction": 0.25, "bases": 800, "base_fraction": 0.2},
    },
    "params": {"min_pident": 80.0, "min_aln_len": 100, "max_evalue": 1e-5, "lca_window": 0.02},
}
ROWS = [
    {"lca_name": "Escherichia coli", "bucket": "enterobacteria"},
    {"lca_name": "Escherichia coli", "bucket": "enterobacteria"},
    {"lca_name": "Escherichia coli", "bucket": "enterobacteria"},
    {"lca_name": "", "bucket": "no hit"},
]


def test_section_goes_before_methods_and_toc_updated():
    out = ac.splice(REPORT, ac.build_section(SUMMARY, ROWS, laava_total=100))
    assert out.index('id="contaminant-screening"') < out.index('<div id="methods"')
    assert out.index("toc-contaminant-screening") < out.index('id="toc-methods"')
    assert out.count("<div") == out.count("</div>")
    assert "4 reads" in out and "4.00% of 100 reads" in out


def test_embedded_png_is_valid():
    section = ac.build_section(SUMMARY, ROWS)
    b64 = re.search(r"base64,([A-Za-z0-9+/=]+)", section).group(1)
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_rerun_replaces_instead_of_duplicating():
    once = ac.splice(REPORT, ac.build_section(SUMMARY, ROWS))
    twice = ac.splice(once, ac.build_section(SUMMARY, ROWS))
    assert twice.count('id="contaminant-screening"') == 1
    assert twice.count("toc-contaminant-screening") == 1
    assert len(twice) == len(once)


def test_no_unmapped_reads_gives_message_not_chart():
    empty = {"total_reads": 0, "buckets": {}, "params": {}}
    section = ac.build_section(empty, [])
    assert "nothing to screen" in section and "<img" not in section


def test_html_in_names_is_escaped():
    rows = [{"lca_name": "<script>x</script>", "bucket": "plant"}]
    summary = {"total_reads": 1, "params": {},
               "buckets": {"plant": {"reads": 1, "read_fraction": 1.0, "bases": 10, "base_fraction": 1.0}}}
    assert "<script>" not in ac.build_section(summary, rows)


def test_fallback_when_methods_missing_and_error_when_not_html():
    out = ac.splice("<html><body><p>x</p></body></html>", "<div>SECTION</div>")
    assert out.index("SECTION") < out.index("</body>")
    with pytest.raises(ValueError):
        ac.splice("not html at all", "<div>SECTION</div>")
