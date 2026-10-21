"""Looking inside input files, so names can be taken from the files instead of guessed."""

from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

# File-type groups the file finder understands, so callers need not guess extensions
# (packaging.fa and hg38.chr19trunc-chrM.fa are FASTA files even though they do not end in .fasta).
KIND_PATTERNS = {
    "reads": ["*.fastq", "*.fastq.gz", "*.fq", "*.fq.gz", "*.bam"],
    "fasta": ["*.fa", "*.fasta", "*.fna", "*.fa.gz", "*.fasta.gz", "*.fna.gz"],
    "bed": ["*.bed"],
    "bam": ["*.bam"],
}
FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")
READ_SUFFIXES = (".bam", ".fastq", ".fastq.gz", ".fq", ".fq.gz")


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def is_fasta(name: str) -> bool:
    return name.lower().endswith(FASTA_SUFFIXES)


def fasta_headers(path: Path) -> list[str]:
    """Every FASTA header line, without the leading '>'."""
    with _open_text(path) as fh:
        return [line[1:].strip() for line in fh if line.startswith(">")]


def fasta_ids(path: Path) -> list[str]:
    """First word of every FASTA header."""
    return [(h.split() or [""])[0] for h in fasta_headers(path)]


def name_matches_header(name: str, header: str) -> bool:
    """Does `name` identify this FASTA record? It must equal the record id, be a leading part of the id
    (pHelper for pHelper_v2), or equal another word of the header. A name that merely appears inside a
    longer word does not count: "Helper" is not "pHelper"."""
    words = header.split()
    if not words:
        return False
    return name == words[0] or words[0].startswith(name) or name in words[1:]


def fasta_records(path: Path, max_records: int = 50, max_scan_bytes: int = 1 << 30) -> dict:
    """Record names and lengths. Very large files are not scanned."""
    size = path.stat().st_size
    if size > max_scan_bytes:
        return {"scanned": False, "note": f"file is {size:,} bytes; too large to scan for record names"}
    records: list[dict] = []
    n = 0
    current: dict | None = None
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
                words = line[1:].split()
                current = {"id": words[0] if words else "", "length": 0}
                if n <= max_records:
                    records.append(current)
                else:
                    current = None
            elif current is not None:
                current["length"] += len(line.strip())
    return {"scanned": True, "n_records": n, "records": records, "truncated": n > max_records}


def _bed_rows(path: Path):
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            yield line.split("\t") if "\t" in line else line.split()


def bed_feature_names(path: Path) -> set[str]:
    """Feature names (4th column), or an empty set if the file has no name column."""
    return {cols[3] for cols in _bed_rows(path) if len(cols) >= 4}


def bed_chroms(path: Path) -> set[str]:
    """Sequence names a BED file refers to (1st column)."""
    return {cols[0] for cols in _bed_rows(path)}


def bed_features(path: Path, max_rows: int = 10) -> dict:
    names: Counter = Counter()
    first = []
    n = 0
    for cols in _bed_rows(path):
        n += 1
        if len(cols) >= 4:
            names[cols[3]] += 1
        if len(first) < max_rows:
            first.append({"chrom": cols[0], "start": cols[1] if len(cols) > 1 else None,
                          "end": cols[2] if len(cols) > 2 else None,
                          "name": cols[3] if len(cols) > 3 else None})
    out = {"n_features": n, "feature_names": dict(names), "first_rows": first}
    if not names:
        out["note"] = "no name column (4th column) found"
    return out
