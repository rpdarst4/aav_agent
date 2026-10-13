"""Settings for the AAV agent tool layer (paths, databases, container images)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class PathNotAllowed(ValueError):
    """A path supplied to a tool lies outside the folders the tools may read."""


@dataclass
class DatabaseConfig:
    name: str  # name the agent uses, e.g. "demo" or "nt"
    dir: Path  # directory holding the BLAST database files
    db_name: str  # database name inside that directory
    description: str = ""

    def available(self) -> bool:
        return self.dir.is_dir() and any(self.dir.glob(f"{self.db_name}.*"))


@dataclass
class Settings:
    data_dir: Path  # runs live under data_dir/runs
    laava_dir: Path  # checkout of formbio/laava
    screen_pipeline: Path  # pipeline/contaminant_screen.nf
    taxdump_dir: Path
    categories: Path
    databases: dict[str, DatabaseConfig]
    default_database: str
    allowed_input_roots: list[Path] = field(default_factory=list)
    nextflow: str = "nextflow"
    nxf_syntax_parser: str | None = None  # e.g. "v1" if LAAVA's config needs it
    nxf_version: str | None = None  # pin a Nextflow version via NXF_VER
    laava_image_repo: str = "ghcr.io/formbio"
    laava_container_version: str = ":latest"
    tools_image: str = "aav-agent-tools:local"
    cpus: int = 4
    memory: str = "6 GB"
    max_hash_bytes: int = 4 * 1024**3  # larger inputs are not checksummed

    def __post_init__(self) -> None:
        for name in ("data_dir", "laava_dir", "screen_pipeline", "taxdump_dir", "categories"):
            setattr(self, name, Path(getattr(self, name)))
        if not self.allowed_input_roots:
            self.allowed_input_roots = [self.data_dir, self.laava_dir]
        self.allowed_input_roots = [Path(p) for p in self.allowed_input_roots]
        if self.default_database not in self.databases:
            raise ValueError(f"default_database {self.default_database!r} is not among the configured databases")

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def laava_image(self) -> str:
        return f"{self.laava_image_repo}/laava{self.laava_container_version}"

    def resolve_input(self, raw: str) -> Path:
        """Resolve a user/LLM-supplied path, refusing anything outside the allowed roots.

        Relative paths are taken relative to data_dir. Symlinks are followed before the
        check, so a link inside an allowed folder cannot point out of it.
        """
        p = Path(str(raw)).expanduser()
        if not p.is_absolute():
            p = self.data_dir / p
        p = p.resolve()
        for root in self.allowed_input_roots:
            if p.is_relative_to(root.resolve()):
                return p
        roots = ", ".join(str(r) for r in self.allowed_input_roots)
        raise PathNotAllowed(f"{raw!r} is outside the folders the tools may read ({roots})")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """Load settings; relative paths in the file are relative to the file itself."""
        path = Path(path)
        base = path.parent
        raw = yaml.safe_load(path.read_text())

        def P(x: str) -> Path:
            p = Path(x).expanduser()
            return p if p.is_absolute() else (base / p).resolve()

        dbs = {
            name: DatabaseConfig(name, P(d["dir"]), d.get("db_name", name), d.get("description", ""))
            for name, d in raw["databases"].items()
        }
        optional = {
            k: raw[k]
            for k in ("nextflow", "nxf_syntax_parser", "nxf_version", "laava_image_repo",
                      "laava_container_version", "tools_image", "cpus", "memory", "max_hash_bytes")
            if k in raw
        }
        return cls(
            data_dir=P(raw["data_dir"]), laava_dir=P(raw["laava_dir"]),
            screen_pipeline=P(raw["screen_pipeline"]), taxdump_dir=P(raw["taxdump_dir"]),
            categories=P(raw["categories"]), databases=dbs, default_database=raw["default_database"],
            allowed_input_roots=[P(x) for x in raw.get("allowed_input_roots", [])], **optional,
        )
