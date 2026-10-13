"""Running external commands, behind an interface so tests can substitute a fake."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult: ...


class SubprocessRunner:
    """Runs commands without a shell. With log_path, output is appended to that file."""

    def run(self, args, *, cwd=None, env=None, log_path=None, timeout=None) -> CommandResult:
        full_env = {**os.environ, **(env or {})}
        try:
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "ab") as fh:
                    p = subprocess.run(args, cwd=cwd, env=full_env, stdout=fh, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
                return CommandResult(p.returncode)
            p = subprocess.run(args, cwd=cwd, env=full_env, capture_output=True, text=True,
                               timeout=timeout, check=False)
            return CommandResult(p.returncode, p.stdout, p.stderr)
        except FileNotFoundError:
            return CommandResult(127, "", f"command not found: {args[0]}")
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", f"timed out after {timeout} s: {' '.join(args[:3])} ...")


def tail(path: Path, lines: int = 25) -> str:
    """Last lines of a log file, for error messages."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])
