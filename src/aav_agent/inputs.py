"""Looking inside input files, so names can be taken from the files instead of guessed."""

from __future__ import annotations

import gzip
import re
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


# ---------------------------------------------------------------------------------------------------
# One-line summaries of files, so a caller can choose between files by content instead of guessing names.
# ---------------------------------------------------------------------------------------------------

_CHROM_LIKE = re.compile(r"^(chr)?(\d{1,2}|[XYM]|MT|Un.*)(_[A-Za-z0-9_]+)?$|^N[CWT]_\d+(\.\d+)?$")

# Genome-assembly names that appear in file names. A hint only, and always reported as coming from the name.
SPECIES_TOKENS = {
    "human": ("hg19", "hg38", "grch37", "grch38", "chm13", "t2t", "human", "hsapiens"),
    "mouse": ("mm9", "mm10", "mm39", "grcm38", "grcm39", "mouse", "mmusculus"),
    "rat": ("rn6", "rn7", "rnor6"),
    "Chinese hamster (CHO)": ("cho", "cricgri", "cricetulus"),
    "Spodoptera frugiperda (Sf9)": ("sf9", "spodoptera"),
    "Drosophila": ("dm3", "dm6", "drosophila"),
}
_SCAN_LIMIT_BYTES = 64 * 1024 * 1024


def species_from_name(filename: str) -> str | None:
    tokens = [t for t in re.split(r"[^a-z0-9]+", filename.lower()) if t]
    for species, names in SPECIES_TOKENS.items():
        for tok in tokens:
            # assembly names (hg38, GRCm39, chm13) may carry a suffix (chm13v2); plain words must match exactly,
            # so "humanized_vector" is not "human"
            if any(tok == n or (len(n) >= 4 and any(c.isdigit() for c in n) and tok.startswith(n)) for n in names):
                return species
    return None


def _format_length(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} Mb"
    if n >= 10_000:
        return f"{n / 1_000:.1f} kb"
    return f"{n:,} bp"


def fasta_profile(path: Path, first_n: int = 4, scan_limit: int = _SCAN_LIMIT_BYTES) -> dict:
    """Record count, first ids and total length; large files are only sampled at the start."""
    size = path.stat().st_size
    ids: list[str] = []
    n = total = 0
    scanned = size <= scan_limit
    bytes_read = 0
    with _open_text(path) as fh:
        for line in fh:
            bytes_read += len(line)
            if line.startswith(">"):
                n += 1
                if len(ids) < first_n:
                    words = line[1:].split()
                    ids.append(words[0] if words else "")
            else:
                total += len(line.strip())
            if not scanned and bytes_read > 1_000_000 and len(ids) >= first_n:
                break  # a big file: the first records say enough
    return {"n": n, "first_ids": ids, "total": total, "scanned": scanned, "size": size}


def fasta_role_hint(profile: dict) -> str:
    ids = profile["first_ids"]
    chrom_like = sum(1 for i in ids if _CHROM_LIKE.match(i))
    if chrom_like and (profile["total"] >= 1_000_000 or chrom_like == len(ids) or not profile["scanned"]):
        return "a genome (chromosome records), so a candidate host genome"
    if profile["scanned"] and profile["n"] == 1 and profile["total"] <= 30_000:
        return "a single short sequence (a vector construct, or a reference such as flip/flop)"
    if profile["scanned"] and 2 <= profile["n"] <= 30 and profile["total"] <= 1_000_000:
        return "a set of plasmid sequences (packaging: RepCap, helper, lambda)"
    return ""


def summarize_file(path: Path) -> str:
    """A short plain-text description of what a file contains."""
    name = path.name.lower()
    try:
        if is_fasta(name):
            prof = fasta_profile(path)
            ids = ", ".join(prof["first_ids"])
            if prof["scanned"]:
                more = ", ..." if prof["n"] > len(prof["first_ids"]) else ""
                text = (f"FASTA, {prof['n']} record{'s' if prof['n'] != 1 else ''} ({ids}{more}), "
                        f"{_format_length(prof['total'])}")
            else:
                text = f"FASTA, large file ({prof['size'] / 1e6:,.0f} MB), first records: {ids}"
            role = fasta_role_hint(prof)
            species = species_from_name(path.name)
            if role:
                text += f"; looks like {role}"
            if species:
                text += f"; the file name suggests {species}"
            return text
        if name.endswith(".bed"):
            bf = bed_features(path, max_rows=0)
            names = ", ".join(list(bf["feature_names"])[:6])
            chroms = ", ".join(sorted(bed_chroms(path))[:3])
            return f"BED, {bf['n_features']} features on {chroms}" + (f" ({names})" if names else "")
        if name.endswith(READ_SUFFIXES):
            return f"reads ({'BAM' if name.endswith('.bam') else 'FASTQ'}), {path.stat().st_size / 1e6:,.1f} MB"
    except (OSError, UnicodeDecodeError, ValueError) as e:
        return f"could not be read ({type(e).__name__})"
    return ""
