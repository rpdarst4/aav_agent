import csv
import random

import pytest

from src.aav_agent.tools import spike_in


def _random_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


@pytest.fixture
def inputs(tmp_path):
    rng = random.Random(1)
    # Two tiny "genomes" of different size
    (tmp_path / "bac.fa").write_text(
        f">chr1 test\n{_random_seq(rng, 5000)}\n>plasmid1\n{_random_seq(rng, 1500)}\n"
    )
    (tmp_path / "insect.fa").write_text(f">ctg1\n{_random_seq(rng, 8000)}\n")
    spec = tmp_path / "spike.yaml"
    spec.write_text(
        "seed: 7\n"
        "organisms:\n"
        "  - {name: Bacillus subtilis, taxid: 1423, category: soil bacteria, fasta: bac.fa, count: 12}\n"
        "  - {name: Spodoptera frugiperda, taxid: 7108, category: insect, fasta: insect.fa, count: 8}\n"
    )
    # Ten "real" reads in FASTQ, 1-2 kb
    lines = []
    for i in range(10):
        seq = _random_seq(rng, 1000 + 100 * i)
        lines.append(f"@m54119U/{i}/ccs\n{seq}\n+\n{'I' * len(seq)}\n")
    (tmp_path / "real.fastq").write_text("".join(lines))
    return tmp_path


def _run(inputs, name="out", **kw):
    out_fq = inputs / f"{name}.fastq"
    out_tsv = inputs / f"{name}.tsv"
    counts = spike_in.spike(
        inputs / "real.fastq", inputs / "spike.yaml", out_fq, out_tsv, **kw
    )
    return counts, out_fq, out_tsv


def test_counts_and_truth(inputs):
    counts, out_fq, out_tsv = _run(inputs)
    assert counts == {"real": 10, "spiked": 20, "total": 30}
    reads = list(spike_in.get_reads(out_fq))
    assert len(reads) == 30
    assert len({r.name for r in reads}) == 30  # unique ids
    truth = list(csv.DictReader(open(out_tsv), delimiter="\t"))
    assert len(truth) == 20
    assert {t["taxid"] for t in truth} == {"1423", "7108"}
    # every truth id is present in the FASTQ and no real read was dropped
    names = {r.name for r in reads}
    assert {t["read_id"] for t in truth} <= names
    assert sum(n.startswith("m54119U/") for n in names) == 10


def test_deterministic_with_seed(inputs):
    _, fq1, _ = _run(inputs, "a", seed=3)
    _, fq2, _ = _run(inputs, "b", seed=3)
    _, fq3, _ = _run(inputs, "c", seed=4)
    assert fq1.read_text() == fq2.read_text()
    assert fq1.read_text() != fq3.read_text()


def test_zero_error_reads_match_source(inputs):
    _, out_fq, out_tsv = _run(inputs, sub=0, ins=0, dele=0)
    bac = spike_in.read_fasta(inputs / "bac.fa")
    insect = spike_in.read_fasta(inputs / "insect.fa")
    genomes = {"1423": bac, "7108": {"ctg1": insect["ctg1"]}}
    seqs = {r.name: r.seq for r in spike_in.get_reads(out_fq)}
    for t in csv.DictReader(open(out_tsv), delimiter="\t"):
        frag = genomes[t["taxid"]][t["source_contig"]][int(t["start"]) : int(t["end"])]
        if t["strand"] == "-":
            frag = spike_in.revcomp(frag)
        assert seqs[t["read_id"]] == frag
        assert int(t["length"]) == len(frag)
        assert (t["n_sub"], t["n_ins"], t["n_del"]) == ("0", "0", "0")


def test_errors_are_injected_and_counted(inputs):
    _, _, out_tsv = _run(inputs, sub=0.02, ins=0.01, dele=0.01)
    truth = list(csv.DictReader(open(out_tsv), delimiter="\t"))
    assert sum(int(t["n_sub"]) + int(t["n_ins"]) + int(t["n_del"]) for t in truth) > 0
    for t in truth:
        expected = int(t["end"]) - int(t["start"]) + int(t["n_ins"]) - int(t["n_del"])
        assert int(t["length"]) == expected


def test_lengths_capped_by_contig(inputs):
    # real reads are <= 1.9 kb, so spiked reads should never exceed that
    _, _, out_tsv = _run(inputs)
    for t in csv.DictReader(open(out_tsv), delimiter="\t"):
        assert int(t["end"]) - int(t["start"]) <= 1900


def test_duplicate_taxid_rejected(inputs):
    (inputs / "dup.yaml").write_text(
        "organisms:\n"
        "  - {name: A, taxid: 1, category: x, fasta: bac.fa, count: 1}\n"
        "  - {name: B, taxid: 1, category: x, fasta: bac.fa, count: 1}\n"
    )
    with pytest.raises(ValueError, match="duplicate taxid"):
        spike_in.spike(
            inputs / "real.fastq", inputs / "dup.yaml", inputs / "o.fq", inputs / "o.tsv"
        )
