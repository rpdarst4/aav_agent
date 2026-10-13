"""Tests for classify_reads. The taxonomy is a small synthetic taxdump built in a
temp dir: real taxids, simplified lineages (not a faithful NCBI excerpt)."""

import csv
import json

import pytest
import yaml

from aav_agent.tools import classify_reads as cr

# (taxid, parent, rank, name)
NODES = [
    (1, 1, "no rank", "root"),
    (131567, 1, "no rank", "cellular organisms"),
    (2, 131567, "superkingdom", "Bacteria"),
    (1224, 2, "phylum", "Pseudomonadota"),
    (1236, 1224, "class", "Gammaproteobacteria"),
    (543, 1236, "family", "Enterobacteriaceae"),
    (561, 543, "genus", "Escherichia"),
    (562, 561, "species", "Escherichia coli"),
    (28901, 543, "species", "Salmonella enterica"),
    (286, 1236, "genus", "Pseudomonas"),
    (303, 286, "species", "Pseudomonas putida"),
    (1239, 2, "phylum", "Bacillota"),
    (1386, 1239, "genus", "Bacillus"),
    (1423, 1386, "species", "Bacillus subtilis"),
    (2759, 131567, "superkingdom", "Eukaryota"),
    (33208, 2759, "kingdom", "Metazoa"),
    (9606, 33208, "species", "Homo sapiens"),
    (50557, 33208, "class", "Insecta"),
    (7108, 50557, "species", "Spodoptera frugiperda"),
    (33090, 2759, "kingdom", "Viridiplantae"),
    (3702, 33090, "species", "Arabidopsis thaliana"),
    (10239, 1, "superkingdom", "Viruses"),
    (10442, 10239, "family", "Baculoviridae"),
    (46015, 10442, "species", "Autographa californica nucleopolyhedrovirus"),
]

CATEGORIES = {
    "categories": [
        {"name": "host (human)", "taxids": [9606]},
        {"name": "enterobacteria", "taxids": [543]},
        {"name": "soil bacteria", "taxids": [1386, 286]},
        {"name": "other bacteria", "taxids": [2]},
        {"name": "baculovirus", "taxids": [10442]},
        {"name": "insect", "taxids": [50557]},
        {"name": "plant", "taxids": [33090]},
        {"name": "other eukaryote", "taxids": [2759]},
    ],
    "default": "unclassified",
}


@pytest.fixture
def env(tmp_path):
    d = tmp_path / "taxdump"
    d.mkdir()
    (d / "nodes.dmp").write_text(
        "".join(f"{t}\t|\t{p}\t|\t{r}\t|\t\t|\n" for t, p, r, _ in NODES)
    )
    (d / "names.dmp").write_text(
        "".join(f"{t}\t|\t{n}\t|\t\t|\tscientific name\t|\n" for t, _, _, n in NODES)
        + "562\t|\tE. coli\t|\t\t|\tsynonym\t|\n"
    )
    (d / "merged.dmp").write_text("99999\t|\t562\t|\n")
    tax = cr.Taxonomy(d)
    return tax, cr.Categorizer(tax, CATEGORIES), tmp_path


def hsp(q, s, taxids, qs=1, qe=1000, bits=1000.0, pident=99.0, length=None, evalue=0.0):
    return cr.Hsp(q, s, pident, length or (qe - qs + 1), qs, qe, evalue, bits, tuple(taxids))


def classify(env, hsps, qlen=1000, **kw):
    tax, cats, _ = env
    return cr.classify_read("r1", qlen, hsps, tax, cats, cr.Params(**kw))


# ---- taxonomy ----

def test_lca_and_names(env):
    tax, _, _ = env
    assert tax.lca([562, 28901]) == 543
    assert tax.lca([562, 3702]) == 131567
    assert tax.lca([562]) == 562
    assert tax.name(562) == "Escherichia coli"  # scientific name, not the synonym
    assert tax.rank_of(543) == "family"
    assert tax.resolve(99999) == 562  # merged
    assert tax.resolve(123456789) is None


# ---- per-read classification ----

def test_many_strains_same_species(env):
    hsps = [hsp("r1", f"strain{i}", [562], bits=3300 + i) for i in range(10)]
    row = classify(env, hsps)
    assert row["bucket"] == "enterobacteria"
    assert row["lca_taxid"] == 562 and row["n_subjects_kept"] == 10


def test_tie_between_species_goes_to_lca(env):
    row = classify(env, [hsp("r1", "a", [562]), hsp("r1", "b", [28901])])
    assert row["lca_taxid"] == 543 and row["bucket"] == "enterobacteria"


def test_clear_best_hit_beats_weak_alternative(env):
    row = classify(env, [hsp("r1", "a", [562], bits=1000), hsp("r1", "b", [28901], bits=500)])
    assert row["lca_taxid"] == 562


def test_tie_across_kingdoms_is_ambiguous(env):
    row = classify(env, [hsp("r1", "a", [562]), hsp("r1", "b", [3702])])
    assert row["bucket"] == cr.AMBIGUOUS and row["lca_taxid"] == 131567


def test_lca_among_bacteria_falls_to_other_bacteria(env):
    row = classify(env, [hsp("r1", "a", [562]), hsp("r1", "b", [1423])])
    assert row["bucket"] == "other bacteria"


def test_no_hit_and_weak_hits(env):
    assert classify(env, [])["bucket"] == cr.NO_HIT
    weak = classify(env, [hsp("r1", "a", [562], pident=70.0)])
    assert weak["bucket"] == cr.NO_HIT and weak["note"] == "only weak hits"
    short = classify(env, [hsp("r1", "a", [562], qs=1, qe=50)])
    assert short["bucket"] == cr.NO_HIT


def test_unknown_and_multiple_taxids(env):
    row = classify(env, [hsp("r1", "a", [9606, 9598])])  # 9598 not in taxonomy
    assert row["bucket"] == "host (human)" and row["lca_taxid"] == 9606
    assert classify(env, [hsp("r1", "a", [424242])])["note"] == "no hit with a known taxid"


def test_low_coverage(env):
    row = classify(env, [hsp("r1", "a", [562], qs=1, qe=200)], qlen=2000)
    assert row["bucket"] == cr.LOW_COVERAGE and row["query_coverage"] == 0.1


def test_inverted_repeat_read_counts_as_one_organism(env):
    # like the E. coli read: two HSPs to the same subject on opposite strands
    row = classify(
        env,
        [hsp("r1", "chr", [562], qs=1, qe=1836, bits=3300),
         hsp("r1", "chr", [562], qs=3187, qe=5015, bits=3300)],
        qlen=5015,
    )
    assert row["bucket"] == "enterobacteria"
    assert row["query_coverage"] == pytest.approx(0.73, abs=0.01)


def test_mixed_read(env):
    row = classify(
        env,
        [hsp("r1", "ec", [562], qs=1, qe=1000, bits=2000),
         hsp("r1", "bs", [1423], qs=1001, qe=2000, bits=1500)],
        qlen=2000,
    )
    assert row["bucket"] == cr.MIXED and "enterobacteria" in row["note"]


def test_overlapping_second_category_is_not_mixed(env):
    # the weaker bacillus hit covers the same part of the read
    row = classify(
        env,
        [hsp("r1", "ec", [562], qs=1, qe=2000, bits=3000),
         hsp("r1", "bs", [1423], qs=1, qe=2000, bits=1000)],
        qlen=2000,
    )
    assert row["bucket"] == "enterobacteria"


def test_interval_helpers():
    assert cr.merge_intervals([(5, 10), (1, 3), (4, 6)]) == [(1, 10)]
    assert cr.subtract_intervals([(1, 100)], [(10, 20), (50, 60)]) == [(1, 9), (21, 49), (61, 100)]
    assert cr.interval_length([(1, 10), (20, 29)]) == 20


# ---- file level ----

def write_blast(path, hsps):
    with open(path, "w") as fh:
        fh.write("# comment line is ignored\n")
        for h in hsps:
            fh.write("\t".join(map(str, [
                h.qseqid, h.sseqid, h.pident, h.length, h.qstart, h.qend, 1, h.length,
                h.evalue, h.bitscore, ";".join(map(str, h.taxids)),
            ])) + "\n")


def test_classify_all_summary_and_validation(env):
    tax, cats, tmp = env
    (tmp / "q.fa").write_text(
        ">spike/562/00000/ccs\n" + "A" * 1000 + "\n"
        ">spike/7108/00000/ccs\n" + "A" * 1000 + "\n"
        ">real/1 extra words\n" + "A" * 500 + "\n"
    )
    write_blast(tmp / "hits.tsv", [
        hsp("spike/562/00000/ccs", "s1", [562]),
        hsp("spike/7108/00000/ccs", "s2", [7108]),
    ])
    rows = cr.classify_all(tmp / "hits.tsv", tmp / "q.fa", tax, cats, cr.Params())
    assert [r["bucket"] for r in rows] == ["enterobacteria", "insect", cr.NO_HIT]
    summary = cr.summarize(rows)
    assert summary["total_reads"] == 3
    assert summary["buckets"]["insect"]["reads"] == 1
    assert summary["buckets"][cr.NO_HIT]["bases"] == 500

    truth = [
        {"read_id": "spike/562/00000/ccs", "taxid": "562"},
        {"read_id": "spike/7108/00000/ccs", "taxid": "7108"},
        {"read_id": "spike/562/00001/ccs", "taxid": "562"},  # never reached BLAST: leaked
    ]
    v = cr.validate(rows, truth, tax, cats)
    assert v["spiked_reads"] == 3 and v["leaked_reads"] == 1
    assert v["correct"] == 2 and v["recovery_of_classifiable"] == 1.0
    assert {"expected": "enterobacteria", "observed": cr.LEAKED, "reads": 1} in v["confusion"]


def test_parse_blast_rejects_wrong_columns(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("q\ts\t99\t100\n")
    with pytest.raises(ValueError, match="expected 11 columns"):
        list(cr.parse_blast(bad))


def test_cli_end_to_end(env, capsys):
    tax, cats, tmp = env
    (tmp / "cats.yaml").write_text(yaml.safe_dump(CATEGORIES))
    (tmp / "q.fa").write_text(">a\n" + "A" * 1000 + "\n>b\n" + "A" * 1000 + "\n")
    write_blast(tmp / "hits.tsv", [hsp("a", "s", [9606])])
    (tmp / "truth.tsv").write_text("read_id\ttaxid\na\t9606\nb\t562\n")
    rc = cr.main([
        "--blast", str(tmp / "hits.tsv"), "--queries", str(tmp / "q.fa"),
        "--taxdump", str(tmp / "taxdump"), "--categories", str(tmp / "cats.yaml"),
        "--out-tsv", str(tmp / "out.tsv"), "--out-summary", str(tmp / "out.json"),
        "--truth", str(tmp / "truth.tsv"),
    ])
    assert rc == 0
    rows = list(csv.DictReader(open(tmp / "out.tsv"), delimiter="\t"))
    assert [r["bucket"] for r in rows] == ["host (human)", cr.NO_HIT]
    assert "\r" not in (tmp / "out.tsv").read_text()
    summary = json.loads((tmp / "out.json").read_text())
    assert summary["validation"]["leaked_reads"] == 0
    assert cr.main(["--print-outfmt"]) == 0
    assert capsys.readouterr().out.strip().endswith("staxids")
