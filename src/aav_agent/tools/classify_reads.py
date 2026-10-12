"""Classify unmapped reads into contaminant categories from blastn hits.

Input is blastn tabular output (outfmt 6) with exactly the columns in
BLAST_FIELDS, plus the query FASTA of unmapped reads (so reads with no hit are
counted). Run blastn with ``-outfmt "$(python classify_reads.py --print-outfmt)"``
or use scripts/run_blastn.sh.

Per read:
  1. Drop weak HSPs (identity, alignment length, e-value thresholds).
  2. Score each subject sequence by the sum of its HSP bit scores.
  3. Keep subjects within ``lca_window`` of the best subject (default 2%), so
     dozens of near-identical strain genomes do not make the call arbitrary.
  4. Take the lowest common ancestor (LCA) of the kept subjects' taxids.
  5. Map the LCA to a category using an ordered YAML list of taxonomy nodes.
  6. Flag reads that are "mixed" (a second category covers a large part of the
     read not explained by the first) or have "low coverage".

The taxonomy is read directly from NCBI taxdump files (nodes.dmp, names.dmp,
optionally merged.dmp), so TaxonKit is not required. A full taxdump takes a few
seconds and a few hundred MB of RAM to load.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

BLAST_FIELDS = [
    "qseqid", "sseqid", "pident", "length", "qstart", "qend",
    "sstart", "send", "evalue", "bitscore", "staxids",
]
BLAST_OUTFMT = "6 " + " ".join(BLAST_FIELDS)

# Bucket names used when a read is not assigned to a category.
NO_HIT = "no hit"
AMBIGUOUS = "ambiguous"
LOW_COVERAGE = "low coverage"
MIXED = "mixed"
LEAKED = "not unmapped (matched a LAAVA reference)"


# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------


class Taxonomy:
    """Minimal NCBI taxonomy: parent, rank, scientific name, LCA."""

    def __init__(self, taxdump_dir: str | Path):
        d = Path(taxdump_dir)
        self.parent: dict[int, int] = {}
        self.rank: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.merged: dict[int, int] = {}
        with open(d / "nodes.dmp") as fh:
            for line in fh:
                f = [x.strip() for x in line.split("|")]
                self.parent[int(f[0])] = int(f[1])
                self.rank[int(f[0])] = f[2]
        with open(d / "names.dmp") as fh:
            for line in fh:
                f = [x.strip() for x in line.split("|")]
                if f[3] == "scientific name":
                    self.names[int(f[0])] = f[1]
        merged = d / "merged.dmp"
        if merged.exists():
            with open(merged) as fh:
                for line in fh:
                    f = [x.strip() for x in line.split("|")]
                    self.merged[int(f[0])] = int(f[1])
        self.lineage = lru_cache(maxsize=None)(self._lineage)

    def resolve(self, taxid: int) -> int | None:
        """Follow merged.dmp; None if the taxid is unknown."""
        taxid = self.merged.get(taxid, taxid)
        return taxid if taxid in self.parent else None

    def _lineage(self, taxid: int) -> tuple[int, ...]:
        """Root-to-leaf tuple of taxids (root first)."""
        path = []
        seen = set()
        while taxid not in seen:
            seen.add(taxid)
            path.append(taxid)
            nxt = self.parent.get(taxid, taxid)
            if nxt == taxid:
                break
            taxid = nxt
        return tuple(reversed(path))

    def lca(self, taxids: Iterable[int]) -> int | None:
        lineages = [self.lineage(t) for t in taxids]
        if not lineages:
            return None
        common = None
        for lin in lineages:
            if common is None:
                common = lin
                continue
            n = 0
            for a, b in zip(common, lin):
                if a != b:
                    break
                n += 1
            common = common[:n]
        return common[-1] if common else None

    def name(self, taxid: int | None) -> str:
        return self.names.get(taxid, "") if taxid is not None else ""

    def rank_of(self, taxid: int | None) -> str:
        return self.rank.get(taxid, "") if taxid is not None else ""


class Categorizer:
    """Ordered category list; the first category whose node is in the lineage wins."""

    def __init__(self, tax: Taxonomy, spec: dict):
        self.tax = tax
        self.cats = [(c["name"], set(int(t) for t in c["taxids"])) for c in spec["categories"]]
        self.default = spec.get("default", "unclassified")
        self.names = [n for n, _ in self.cats]

    def category_of(self, taxid: int | None) -> str | None:
        if taxid is None:
            return None
        lin = set(self.tax.lineage(taxid))
        for name, nodes in self.cats:
            if lin & nodes:
                return name
        return None


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass
class Hsp:
    qseqid: str
    sseqid: str
    pident: float
    length: int
    qstart: int
    qend: int
    evalue: float
    bitscore: float
    taxids: tuple[int, ...]


def _parse_taxids(field: str) -> tuple[int, ...]:
    out = []
    for part in field.replace(",", ";").split(";"):
        part = part.strip()
        if part.isdigit() and int(part) > 0:
            out.append(int(part))
    return tuple(out)


def parse_blast(path: str | Path) -> Iterator[Hsp]:
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < len(BLAST_FIELDS):
                raise ValueError(
                    f"{path}: expected {len(BLAST_FIELDS)} columns "
                    f"({' '.join(BLAST_FIELDS)}), got {len(f)}: {line[:80]!r}"
                )
            qstart, qend = int(f[4]), int(f[5])
            yield Hsp(
                qseqid=f[0], sseqid=f[1], pident=float(f[2]), length=int(f[3]),
                qstart=min(qstart, qend), qend=max(qstart, qend),
                evalue=float(f[8]), bitscore=float(f[9]), taxids=_parse_taxids(f[10]),
            )


def read_fasta_lengths(path: str | Path) -> dict[str, int]:
    """Ordered {id: length}; id is the first word of the header."""
    lengths: dict[str, int] = {}
    current = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                current = line[1:].split()[0]
                lengths[current] = 0
            elif current is not None:
                lengths[current] += len(line)
    return lengths


# --------------------------------------------------------------------------
# Interval helpers (1-based inclusive query coordinates)
# --------------------------------------------------------------------------


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(e - s + 1 for s, e in intervals)


def subtract_intervals(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Parts of a not covered by b (both merged)."""
    out = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be < cur or bs > e:
                continue
            if bs > cur:
                out.append((cur, bs - 1))
            cur = max(cur, be + 1)
            if cur > e:
                break
        if cur <= e:
            out.append((cur, e))
    return out


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@dataclass
class Params:
    min_pident: float = 80.0
    min_aln_len: int = 100
    max_evalue: float = 1e-5
    lca_window: float = 0.02
    min_query_coverage: float = 0.5
    mixed_fraction: float = 0.3


def classify_read(
    read_id: str,
    qlen: int,
    hsps: list[Hsp],
    tax: Taxonomy,
    cats: Categorizer,
    p: Params,
) -> dict:
    row = {
        "read_id": read_id, "length": qlen, "n_hsps": len(hsps), "n_subjects_kept": 0,
        "query_coverage": 0.0, "lca_taxid": "", "lca_name": "", "lca_rank": "",
        "category": "", "bucket": NO_HIT, "note": "",
    }
    if not hsps:
        return row

    strong = [h for h in hsps if h.pident >= p.min_pident and h.length >= p.min_aln_len
              and h.evalue <= p.max_evalue]
    if not strong:
        row["note"] = "only weak hits"
        return row

    # Resolve taxids; drop HSPs whose taxids are all unknown to the taxonomy.
    resolved: list[tuple[Hsp, tuple[int, ...]]] = []
    for h in strong:
        ids = tuple(t for t in (tax.resolve(x) for x in h.taxids) if t is not None)
        if ids:
            resolved.append((h, ids))
    if not resolved:
        row["note"] = "no hit with a known taxid"
        return row

    by_subject: dict[str, list[tuple[Hsp, tuple[int, ...]]]] = defaultdict(list)
    for h, ids in resolved:
        by_subject[h.sseqid].append((h, ids))
    score = {s: sum(h.bitscore for h, _ in hs) for s, hs in by_subject.items()}
    best = max(score.values())
    kept = {s for s, v in score.items() if v >= best * (1 - p.lca_window)}

    kept_taxids = {t for s in kept for _, ids in by_subject[s] for t in ids}
    lca = tax.lca(kept_taxids)
    row.update(
        n_subjects_kept=len(kept), lca_taxid=lca, lca_name=tax.name(lca), lca_rank=tax.rank_of(lca)
    )

    kept_hsps = [h for s in kept for h, _ in by_subject[s]]
    kept_cov = merge_intervals((h.qstart, h.qend) for h in kept_hsps)
    coverage = interval_length(kept_cov) / qlen if qlen else 0.0
    row["query_coverage"] = round(coverage, 3)

    category = cats.category_of(lca)
    row["category"] = category or cats.default
    if category is None:
        row["bucket"] = AMBIGUOUS
        row["note"] = "LCA above all configured categories"
        return row

    # Mixed: another category explains a large part of the read the first one does not.
    other_by_cat: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s, hs in by_subject.items():
        if s in kept:
            continue
        for h, ids in hs:
            c = cats.category_of(tax.lca(ids))
            if c is not None and c != category:
                other_by_cat[c].append((h.qstart, h.qend))
    for c, ivs in other_by_cat.items():
        extra = subtract_intervals(merge_intervals(ivs), kept_cov)
        if qlen and interval_length(extra) >= p.mixed_fraction * qlen:
            row["bucket"] = MIXED
            row["note"] = f"{category} + {c}"
            return row

    if coverage < p.min_query_coverage:
        row["bucket"] = LOW_COVERAGE
        row["note"] = f"{category} over {coverage:.0%} of read"
        return row

    row["bucket"] = category
    return row


def classify_all(
    blast_path: str | Path,
    query_fasta: str | Path,
    tax: Taxonomy,
    cats: Categorizer,
    p: Params,
) -> list[dict]:
    lengths = read_fasta_lengths(query_fasta)
    by_query: dict[str, list[Hsp]] = defaultdict(list)
    for h in parse_blast(blast_path):
        by_query[h.qseqid].append(h)
    unknown = set(by_query) - set(lengths)
    if unknown:
        logger.warning("%d BLAST queries are not in %s (e.g. %s); classifying anyway",
                       len(unknown), query_fasta, sorted(unknown)[:2])
        for q in unknown:
            lengths[q] = max(h.qend for h in by_query[q])
    return [classify_read(rid, ql, by_query.get(rid, []), tax, cats, p) for rid, ql in lengths.items()]


def summarize(rows: list[dict]) -> dict:
    reads = Counter(r["bucket"] for r in rows)
    bases: Counter = Counter()
    for r in rows:
        bases[r["bucket"]] += int(r["length"])
    total = len(rows)
    total_bases = sum(bases.values())
    return {
        "total_reads": total,
        "buckets": {
            b: {
                "reads": n,
                "read_fraction": round(n / total, 4) if total else 0.0,
                "bases": bases[b],
                "base_fraction": round(bases[b] / total_bases, 4) if total_bases else 0.0,
            }
            for b, n in reads.most_common()
        },
    }


# --------------------------------------------------------------------------
# Validation against the spike-in truth table
# --------------------------------------------------------------------------


def validate(rows: list[dict], truth: list[dict], tax: Taxonomy, cats: Categorizer) -> dict:
    """Compare observed buckets with what the truth table says each spike read is.

    Spiked reads absent from the classified set did not end up unmapped in LAAVA
    (e.g. E. coli fragments covering plasmid backbone), so they are reported
    separately as LEAKED, not as misclassifications.
    """
    observed = {r["read_id"]: r["bucket"] for r in rows}
    confusion: Counter = Counter()
    for t in truth:
        taxid = tax.resolve(int(t["taxid"]))
        expected = cats.category_of(taxid) or cats.default
        got = observed.get(t["read_id"], LEAKED)
        confusion[(expected, got)] += 1
    n = sum(confusion.values())
    classifiable = sum(v for (_, g), v in confusion.items() if g != LEAKED)
    correct = sum(v for (e, g), v in confusion.items() if e == g)
    return {
        "spiked_reads": n,
        "leaked_reads": n - classifiable,
        "correct": correct,
        "recovery_of_classifiable": round(correct / classifiable, 4) if classifiable else None,
        "confusion": [
            {"expected": e, "observed": g, "reads": v} for (e, g), v in sorted(confusion.items())
        ],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def write_rows(path: str | Path, rows: list[dict]) -> None:
    fields = ["read_id", "length", "n_hsps", "n_subjects_kept", "query_coverage", "lca_taxid",
              "lca_name", "lca_rank", "category", "bucket", "note"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print-outfmt", action="store_true", help="Print the required -outfmt string and exit")
    ap.add_argument("--blast", type=Path, help="blastn outfmt 6 results")
    ap.add_argument("--queries", type=Path, help="FASTA of the reads that were BLASTed")
    ap.add_argument("--taxdump", type=Path, help="Directory with nodes.dmp, names.dmp (merged.dmp optional)")
    ap.add_argument("--categories", type=Path, help="Categories YAML")
    ap.add_argument("--out-tsv", type=Path, help="Per-read classification TSV")
    ap.add_argument("--out-summary", type=Path, help="Summary JSON")
    ap.add_argument("--truth", type=Path, help="Spike-in truth TSV; adds a validation section to the summary")
    ap.add_argument("--blast-meta", type=Path, default=None,
                    help="Provenance JSON written by run_blastn.sh (default: <blast>.meta.json if present)")
    d = Params()
    ap.add_argument("--min-pident", type=float, default=d.min_pident)
    ap.add_argument("--min-aln-len", type=int, default=d.min_aln_len)
    ap.add_argument("--max-evalue", type=float, default=d.max_evalue)
    ap.add_argument("--lca-window", type=float, default=d.lca_window)
    ap.add_argument("--min-query-coverage", type=float, default=d.min_query_coverage)
    ap.add_argument("--mixed-fraction", type=float, default=d.mixed_fraction)
    a = ap.parse_args(argv)

    if a.print_outfmt:
        print(BLAST_OUTFMT)
        return 0
    for req in ("blast", "queries", "taxdump", "categories", "out_tsv", "out_summary"):
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = Params(a.min_pident, a.min_aln_len, a.max_evalue, a.lca_window,
                    a.min_query_coverage, a.mixed_fraction)
    tax = Taxonomy(a.taxdump)
    cats = Categorizer(tax, yaml.safe_load(a.categories.read_text()))
    rows = classify_all(a.blast, a.queries, tax, cats, params)
    write_rows(a.out_tsv, rows)

    summary = summarize(rows)
    summary["params"] = vars(params)
    meta_path = a.blast_meta or Path(str(a.blast) + ".meta.json")
    if meta_path.exists():
        summary["database"] = json.loads(meta_path.read_text())
    else:
        logger.warning("No BLAST provenance file at %s; the summary will not record which database was used",
                       meta_path)
    if a.truth:
        with open(a.truth, newline="") as fh:
            summary["validation"] = validate(rows, list(csv.DictReader(fh, delimiter="\t")), tax, cats)
    a.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Classified %d reads: %s", len(rows),
                ", ".join(f"{b}={v['reads']}" for b, v in summary["buckets"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
