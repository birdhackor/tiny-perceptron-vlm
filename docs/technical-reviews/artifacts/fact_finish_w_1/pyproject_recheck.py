"""Recheck W.1's installation claim against the current project metadata."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_state():
    return {
        "pyproject": sha256(ROOT / "pyproject.toml"),
        "lock": sha256(ROOT / "uv.lock"),
        "python_pin": sha256(ROOT / ".python-version"),
        "installed_distributions": sorted(
            (package.metadata["Name"], package.version) for package in importlib.metadata.distributions()
        ),
    }


def main():
    previous = tomllib.loads((BASE / "repo-pyproject.toml").read_text())
    current = tomllib.loads((BASE / "pyproject-current.toml").read_text())
    checks = {
        "project": previous["project"] == current["project"],
        "dependency_groups": previous["dependency-groups"] == current["dependency-groups"],
        "build_system": previous["build-system"] == current["build-system"],
        "uv_configuration": previous["tool"]["uv"] == current["tool"]["uv"],
        "hatch_configuration": previous["tool"]["hatch"] == current["tool"]["hatch"],
        "python_requires": current["project"]["requires-python"] == ">=3.11",
        "cpu_extra": current["project"]["optional-dependencies"]["cpu"] == ["torch>=2.11"],
        "notebook_group": current["dependency-groups"]["notebook"]
        == ["ipykernel>=6", "matplotlib>=3.9", "jupyterlab>=4", "nbclient>=0.10"],
    }
    before = protected_state()
    command = ["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run", "--offline"]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=35, check=False)
    after = protected_state()
    record = {
        "reviewer_task": "/root/fact_finish_w_1",
        "executed_at": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_finish_w_1/pyproject_recheck.py",
        "read_scope": "Personally reread full current W.1, full current pyproject, saved actual old/new diff and c5.",
        "metadata_checks": checks,
        "actual_change": "Only tool.ruff.extend-exclude adds docs/natural-assistant/evidence and explanatory comment.",
        "subprocess": {
            "command": command,
            "cwd": str(ROOT),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        "protected_state_unchanged": before == after,
        "source_hashes": {
            "W.1": sha256(BASE / "W.1-pyproject-recheck-source.txt"),
            "previous_pyproject": sha256(BASE / "repo-pyproject.toml"),
            "current_pyproject": sha256(BASE / "pyproject-current.toml"),
            "current_lock": before["lock"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": importlib.metadata.version("torch"),
            "device": "CPU; dry-run only; no installation or kernel execution",
        },
        "conclusion": "W.1/c5 remains valid: project requirements, CPU extra, Notebook group, build metadata and uv config are identical.",
    }
    (BASE / "pyproject-recheck-execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    assert all(checks.values())
    assert result.returncode == 0
    assert before == after


if __name__ == "__main__":
    main()
