"""Spike synthetic contaminant reads into a real PacBio HiFi read set.

Reads a real input (unaligned BAM, or FASTQ / FASTQ.gz), adds simulated reads
drawn from contaminant FASTA files, shuffles everything together, and writes:

  * a FASTQ(.gz) suitable as LAAVA input, and
  * a truth TSV with one row per spiked read (organism, taxid, source contig,
    coordinates, strand, injected error counts).

Real reads are not listed in the truth table; anything absent from it is real.

Spike specification (YAML):

    seed: 42                  # optional; --seed overrides
    organisms:
      - name: Bacillus subtilis
        taxid: 1423
        category: soil bacteria
        fasta: tests/fixtures/contaminants/bacillus.fa
        count: 30

Only the standard library plus PyYAML is needed; pysam is imported lazily and
only when the input is a BAM.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import random
import sys

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
BASES = "ACGT"


@dataclass
class Organism:
    name: str
    taxid: int
    category: str
    fasta: Path
    count: int


@dataclass
class Read:
    name: str
    seq: str
    quals: list[int]


# --------------------------------------------------------------------------
# Input / output helpers
# --------------------------------------------------------------------------


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def _open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def read_fasta(path: Path) -> dict[str, str]:
    """Minimal FASTA parser; contig id is the first word of the header."""
    contigs: dict[str, list[str]] = {}
    current = None
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                contigs[current] = []
            elif current is not None:
                contigs[current].append(line.upper())
            else:
                raise ValueError(f"{path}: sequence data before first header")
    return {name: "".join(parts) for name, parts in contigs.items()}


def get_reads(path: Path) -> Iterator[Read]:
    """Yield reads from an unaligned BAM or a FASTQ(.gz)."""
    if str(path).endswith(".bam"):
        import pysam  # noqa: PLC0415 - optional dependency, BAM input only

        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
            for rec in bam:
                if rec.is_secondary or rec.is_supplementary:
                    continue
                quals = list(rec.query_qualities) if rec.query_qualities is not None else None
                seq = rec.query_sequence
                yield Read(rec.query_name, seq, quals or [40] * len(seq))
        return
    with _open_text(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                return
            seq = handle.readline().strip()
            handle.readline()  # '+'
            qual = handle.readline().strip()
            if not header.startswith("@"):
                raise ValueError(f"{path}: malformed FASTQ near {header!r}")
            name = header[1:].strip().split()[0]
            yield Read(name, seq, [ord(c) - 33 for c in qual])


def write_fastq(path: Path, reads: list[Read]) -> None:
    with _open_text(path, "wt") as out:
        for r in reads:
            qual = "".join(chr(min(q, 93) + 33) for q in r.quals)
            out.write(f"@{r.name}\n{r.seq}\n+\n{qual}\n")


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def inject_errors(
    seq: str, rng: random.Random, sub: float, ins: float, dele: float
) -> tuple[str, int, int, int]:
    """Apply independent per-base substitution / insertion / deletion errors.

    Deliberately simple: HiFi consensus errors are rare (~0.1-0.5%), so there is
    no homopolymer bias or quality-aware model.
    """
    out: list[str] = []
    n_sub = n_ins = n_del = 0
    for base in seq:
        r = rng.random()
        if r < dele:
            n_del += 1
            continue
        if r < dele + sub:
            out.append(rng.choice([b for b in BASES if b != base]))
            n_sub += 1
            continue
        out.append(base)
        if rng.random() < ins:
            out.append(rng.choice(BASES))
            n_ins += 1
    return "".join(out), n_sub, n_ins, n_del


def simulate_organism(
    org: Organism,
    contigs: dict[str, str],
    lengths: list[int],
    rng: random.Random,
    error_rates: tuple[float, float, float],
    qual: int,
    min_len: int,
) -> tuple[list[Read], list[dict]]:
    """Simulate org.count reads; length drawn from the real read-length distribution."""
    usable = {k: v for k, v in contigs.items() if len(v) >= min_len}
    if not usable:
        raise ValueError(f"{org.name}: no contig >= {min_len} bp in {org.fasta}")
    names = list(usable)
    weights = [len(usable[n]) for n in names]
    reads: list[Read] = []
    truth: list[dict] = []
    for i in range(org.count):
        contig = rng.choices(names, weights=weights, k=1)[0]
        cseq = usable[contig]
        length = min(rng.choice(lengths), len(cseq))
        length = max(length, min_len)
        start = rng.randint(0, len(cseq) - length)
        frag = cseq[start : start + length]
        strand = "+" if rng.random() < 0.5 else "-"
        if strand == "-":
            frag = revcomp(frag)
        seq, n_sub, n_ins, n_del = inject_errors(frag, rng, *error_rates)
        read_id = f"spike/{org.taxid}/{i:05d}/ccs"
        reads.append(Read(read_id, seq, [qual] * len(seq)))
        truth.append(
            {
                "read_id": read_id,
                "organism": org.name,
                "taxid": org.taxid,
                "category": org.category,
                "source_contig": contig,
                "start": start,
                "end": start + length,
                "strand": strand,
                "length": len(seq),
                "n_sub": n_sub,
                "n_ins": n_ins,
                "n_del": n_del,
            }
        )
    return reads, truth


def load_spec(path: Path) -> tuple[list[Organism], int | None]:
    spec = yaml.safe_load(path.read_text())
    base = path.parent
    organisms = []
    for entry in spec["organisms"]:
        fasta = Path(entry["fasta"])
        if not fasta.is_absolute() and not fasta.exists():
            fasta = base / fasta
        organisms.append(
            Organism(
                name=entry["name"],
                taxid=int(entry["taxid"]),
                category=entry["category"],
                fasta=fasta,
                count=int(entry["count"]),
            )
        )
    ids = [o.taxid for o in organisms]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate taxid in spike spec; read ids would collide")
    return organisms, spec.get("seed")


def spike(
    reads_path: Path,
    spec_path: Path,
    out_fastq: Path,
    out_truth: Path,
    seed: int | None = None,
    sub: float = 0.0005,
    ins: float = 0.0003,
    dele: float = 0.0004,
    qual: int = 40,
    min_len: int = 200,
) -> dict[str, int]:
    organisms, spec_seed = load_spec(spec_path)
    rng = random.Random(seed if seed is not None else (spec_seed if spec_seed is not None else 0))

    real = list(get_reads(reads_path))
    if not real:
        raise ValueError(f"no reads found in {reads_path}")
    real_names = {r.name for r in real}
    lengths = [len(r.seq) for r in real]

    spiked: list[Read] = []
    truth: list[dict] = []
    for org in organisms:
        r, t = simulate_organism(
            org, read_fasta(org.fasta), lengths, rng, (sub, ins, dele), qual, min_len
        )
        spiked.extend(r)
        truth.extend(t)

    collisions = real_names & {r.name for r in spiked}
    if collisions:
        raise ValueError(f"spike read ids collide with real reads, e.g. {sorted(collisions)[:3]}")

    combined = real + spiked
    rng.shuffle(combined)
    write_fastq(out_fastq, combined)

    fields = [
        "read_id", "organism", "taxid", "category", "source_contig",
        "start", "end", "strand", "length", "n_sub", "n_ins", "n_del",
    ]
    with open(out_truth, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(truth)

    counts = {"real": len(real), "spiked": len(spiked), "total": len(combined)}
    logger.info(
        "Wrote %d reads (%d real + %d spiked, %.2f%% spiked)",
        counts["total"], counts["real"], counts["spiked"],
        100 * counts["spiked"] / counts["total"],
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reads", type=Path, help="Real reads: unaligned BAM or FASTQ(.gz)")
    parser.add_argument("spec", type=Path, help="YAML spike specification")
    parser.add_argument("out_fastq", type=Path, help="Output FASTQ (.gz allowed)")
    parser.add_argument("out_truth", type=Path, help="Output truth TSV")
    parser.add_argument("--seed", type=int, default=None, help="Overrides seed in the spec")
    parser.add_argument("--sub-rate", type=float, default=0.0005)
    parser.add_argument("--ins-rate", type=float, default=0.0003)
    parser.add_argument("--del-rate", type=float, default=0.0004)
    parser.add_argument("--qual", type=int, default=40, help="Constant Phred quality")
    parser.add_argument("--min-len", type=int, default=200)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    spike(
        args.reads, args.spec, args.out_fastq, args.out_truth,
        seed=args.seed, sub=args.sub_rate, ins=args.ins_rate, dele=args.del_rate,
        qual=args.qual, min_len=args.min_len,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
