"""The run specification: the contract between the language model and the tools.

Plain dataclasses with explicit validation. Validation returns readable messages
(never raises for bad input) so they can be fed straight back to the model.
"""

from __future__ import annotations

import gzip
import re

from dataclasses import dataclass, fields
from pathlib import Path

from aav_agent.config import PathNotAllowed, Settings

VECTOR_TYPES = ("ss", "sc", "unspecified")
READ_EXTENSIONS = (".bam", ".fastq", ".fastq.gz", ".fq", ".fq.gz")
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
REQUIRED_FILES = ("reads", "vector_fa", "vector_bed")
OPTIONAL_FILES = ("packaging_fa", "host_fa", "flipflop_fa")
INT_FIELDS = ("target_gap_threshold", "max_allowed_outside_vector", "max_allowed_missing_flanking")


class SpecError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class RunSpec:
    """Inputs for one LAAVA run. Paths may be absolute or relative to the data folder."""

    reads: str  # BAM or FASTQ(.gz) of PacBio reads
    vector_fa: str  # AAV vector plasmid/construct FASTA
    vector_bed: str  # LAAVA annotation BED for the vector
    vector_type: str = "unspecified"  # "ss", "sc" or "unspecified"
    packaging_fa: str | None = None  # RepCap + helper plasmid(s) (+ lambda) FASTA
    host_fa: str | None = None  # host cell genome FASTA
    repcap_name: str = ""  # record name(s) in packaging_fa
    helper_name: str = ""
    lambda_name: str = ""
    itr_label_1: str = "ITR"
    itr_label_2: str = "ITR"
    mitr_label: str = ""
    flipflop_name: str = ""
    flipflop_fa: str | None = None
    sample_id: str | None = None  # default: the reads file name without extension
    target_gap_threshold: int = 200
    max_allowed_outside_vector: int = 100
    max_allowed_missing_flanking: int = 100
    min_supp_joint_coverage: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> RunSpec:
        names = {f.name for f in fields(cls)}
        errors = []
        unknown = sorted(set(d) - names)
        if unknown:
            errors.append(f"unknown field(s): {', '.join(unknown)}")
        missing = [f for f in REQUIRED_FILES if not d.get(f)]
        errors += [f"{f} is required" for f in missing]
        clean = {k: v for k, v in d.items() if k in names}
        for k in INT_FIELDS:
            if k in clean:
                try:
                    clean[k] = int(clean[k])
                except (TypeError, ValueError):
                    errors.append(f"{k} must be an integer, got {clean[k]!r}")
        if "min_supp_joint_coverage" in clean:
            try:
                clean["min_supp_joint_coverage"] = float(clean["min_supp_joint_coverage"])
            except (TypeError, ValueError):
                errors.append(f"min_supp_joint_coverage must be a number, got {clean['min_supp_joint_coverage']!r}")
        for k, v in list(clean.items()):
            if isinstance(v, str):
                clean[k] = v.strip()
        if errors:
            raise SpecError(errors)
        return cls(**clean)


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    paths: dict[str, Path]
    sample_id: str


def strip_read_extension(name: str) -> str:
    for ext in sorted(READ_EXTENSIONS, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def fasta_ids(path: Path) -> list[str]:
    """First word of every FASTA header."""
    opener = gzip.open if str(path).endswith(".gz") else open
    ids = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].split()[0] if line[1:].split() else "")
    return ids


def validate_run_spec(spec: RunSpec, settings: Settings) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    paths: dict[str, Path] = {}

    for name in REQUIRED_FILES + OPTIONAL_FILES:
        raw = getattr(spec, name)
        if not raw:
            if name in REQUIRED_FILES:
                errors.append(f"{name} is required")
            continue
        try:
            p = settings.resolve_input(raw)
        except PathNotAllowed as e:
            errors.append(f"{name}: {e}")
            continue
        if not p.is_file():
            errors.append(f"{name}: file not found: {p}")
            continue
        paths[name] = p

    if "reads" in paths and not paths["reads"].name.endswith(READ_EXTENSIONS):
        errors.append(f"reads: unsupported file type {paths['reads'].name!r} (use {', '.join(READ_EXTENSIONS)})")

    if spec.vector_type not in VECTOR_TYPES:
        errors.append(f"vector_type must be one of {', '.join(VECTOR_TYPES)}, got {spec.vector_type!r}")

    if spec.sample_id:
        sample_id = spec.sample_id
    elif "reads" in paths:
        sample_id = strip_read_extension(paths["reads"].name)
    else:
        sample_id = ""
    if sample_id and not SAMPLE_ID_RE.match(sample_id):
        errors.append(f"sample_id {sample_id!r} must be 1-64 characters: letters, digits, '.', '_' or '-', "
                      "starting with a letter or digit")

    for k in INT_FIELDS:
        if getattr(spec, k) < 0:
            errors.append(f"{k} must be >= 0")
    if not 0 <= spec.min_supp_joint_coverage <= 1:
        errors.append("min_supp_joint_coverage must be between 0 and 1")

    # Names refer to records in the packaging FASTA; a mismatch is a warning because LAAVA's own
    # matching rules are not checked here.
    named = {k: getattr(spec, k) for k in ("repcap_name", "helper_name", "lambda_name") if getattr(spec, k)}
    if named and "packaging_fa" not in paths:
        warnings.append("repcap/helper/lambda names were given without a packaging_fa file")
    elif named:
        ids = fasta_ids(paths["packaging_fa"])
        for k, v in named.items():
            if not any(v == i or v in i for i in ids):
                warnings.append(f"{k}={v!r} does not match any record in {paths['packaging_fa'].name} "
                                f"(records: {', '.join(ids[:8])})")
    if bool(spec.flipflop_name) != bool(spec.flipflop_fa):
        warnings.append("flipflop_name and flipflop_fa are normally given together")

    return ValidationResult(errors, warnings, paths, sample_id)


def build_laava_params(spec: RunSpec, v: ValidationResult) -> dict:
    """The params file Nextflow reads; every path is absolute."""
    params: dict = {
        "seq_reads_file": str(v.paths["reads"]),
        "sample_unique_id": v.sample_id,
        "vector_type": spec.vector_type,
        "vector_fa": str(v.paths["vector_fa"]),
        "vector_bed": str(v.paths["vector_bed"]),
        "itr_label_1": spec.itr_label_1,
        "itr_label_2": spec.itr_label_2,
        "mitr_label": spec.mitr_label,
        "repcap_name": spec.repcap_name,
        "helper_name": spec.helper_name,
        "lambda_name": spec.lambda_name,
        "flipflop_name": spec.flipflop_name,
        "target_gap_threshold": spec.target_gap_threshold,
        "max_allowed_outside_vector": spec.max_allowed_outside_vector,
        "max_allowed_missing_flanking": spec.max_allowed_missing_flanking,
        "min_supp_joint_coverage": spec.min_supp_joint_coverage,
    }
    for k in OPTIONAL_FILES:
        if k in v.paths:
            params[k] = str(v.paths[k])
    return params
