"""Record exactly which code, images and tools produced a run."""

from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone

from aav_agent.config import Settings
from aav_agent.runner import CommandRunner


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _out(runner: CommandRunner, args: list[str]) -> tuple[int, str]:
    r = runner.run(args)
    return r.returncode, (r.stdout or "").strip()


def image_info(runner: CommandRunner, image: str) -> dict | None:
    """Local image id and registry digests, or None if the image is not present locally."""
    rc, image_id = _out(runner, ["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    if rc != 0 or not image_id:
        return None
    rc, digests = _out(runner, ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"])
    try:
        repo_digests = json.loads(digests) if rc == 0 else []
    except json.JSONDecodeError:
        repo_digests = []
    return {"image": image, "id": image_id, "repo_digests": repo_digests}


def collect_provenance(settings: Settings, runner: CommandRunner, include_laava: bool = True) -> dict:
    """Best effort: anything that cannot be determined is recorded as None, never raised."""
    prov: dict = {
        "collected_utc": utcnow(),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "tools_image": image_info(runner, settings.tools_image),
    }
    rc, version_text = _out(runner, [settings.nextflow, "-version"])
    m = re.search(r"version\s+([\w.\-]+)", version_text) if rc == 0 else None
    prov["nextflow_version"] = m.group(1) if m else None
    prov["nxf_syntax_parser"] = settings.nxf_syntax_parser
    try:
        from importlib.metadata import version

        prov["aav_agent_version"] = version("aav-agent")
    except Exception:  # not installed as a package (e.g. running from a checkout)
        prov["aav_agent_version"] = None

    if include_laava:
        git = ["git", "-C", str(settings.laava_dir)]
        rc, commit = _out(runner, git + ["rev-parse", "HEAD"])
        rc_b, branch = _out(runner, git + ["rev-parse", "--abbrev-ref", "HEAD"])
        rc_s, status = _out(runner, git + ["status", "--porcelain"])
        prov["laava"] = {
            "git_commit": commit if rc == 0 else None,
            "git_branch": branch if rc_b == 0 else None,
            "git_dirty": (status != "") if rc_s == 0 else None,
            "image": image_info(runner, settings.laava_image),
        }
    return prov
