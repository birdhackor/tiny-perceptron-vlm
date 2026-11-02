"""Bounded read-only review: no config write, clone, installation, or training."""

import hashlib
import json
import platform
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(args, expected):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode in expected, (args, result.returncode, result.stderr)
    return {
        "command": args,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


contracts = []
for url, expected_sha, ranges in [
    (
        "https://raw.githubusercontent.com/git-for-windows/git/v2.47.1.windows.1/Documentation/config/core.txt",
        "1b2808ff99ef5b106434098ffde1b730e3244e99d8e332e7e90b7e783640bfa0",
        [(694, 699)],
    ),
    (
        "https://raw.githubusercontent.com/git/git/v2.47.1/Documentation/git-config.txt",
        "f954cfa937f32969cdd6926be559c8dedf5adea4a721af78f1f6e173e7c0f3d4",
        [(42, 52), (147, 156), (292, 302), (358, 365)],
    ),
]:
    raw = urllib.request.urlopen(url, timeout=30).read()
    assert sha(raw) == expected_sha
    lines = raw.decode().splitlines()
    contracts.append({
        "url": url,
        "accessed_at": datetime.now(UTC).isoformat(),
        "original_sha256": sha(raw),
        "original_bytes": len(raw),
        "excerpts": [
            {"lines": f"{start}-{end}", "text": "\n".join(lines[start - 1:end])}
            for start, end in ranges
        ],
    })

snapshot = OUT / "git-original-contracts.json"
snapshot.write_text(json.dumps(contracts, ensure_ascii=False, indent=2) + "\n")

version = run(["git", "--version"], {0})
help_result = run(["git", "config", "set", "-h"], {0, 129})
assert "git config set" in help_result["stdout"] + help_result["stderr"]
assert "--[no-]global" in help_result["stdout"] + help_result["stderr"]
boolean = run(["git", "-c", "core.longpaths=true", "config", "--type=bool", "--get", "core.longpaths"], {0})
assert boolean["stdout"] == "true\n"

source_path = ROOT / "assets/training/README.md"
source_raw = source_path.read_bytes()
assert sha(source_raw) == "80607864bff80235b57645839359f71c8e9375741017e947ec205f09ee8ffb67"
paragraph = next(p for p in source_raw.decode().split("\n\n") if p.startswith("Windows PowerShell"))
steps = ["git config --global core.longpaths true", '$env:GIT_LFS_SKIP_SMUDGE="1"', "並clone"]
positions = [paragraph.index(step) for step in steps]
assert positions == sorted(positions)
assert "先執行" in paragraph and "再設定" in paragraph

print(json.dumps({
    "created_at": datetime.now(UTC).isoformat(),
    "command": f"{sys.executable} {Path(__file__).relative_to(ROOT)}",
    "environment": {"python": platform.python_version(), "git": version["stdout"].strip(), "platform": platform.platform(), "device": "CPU; text and Git configuration query only"},
    "source_sha256": sha(source_raw),
    "official_snapshot": {"path": snapshot.relative_to(ROOT).as_posix(), "sha256": sha(snapshot.read_bytes())},
    "executions": [version, help_result, boolean],
    "powershell_paragraph": paragraph,
    "step_positions_within_paragraph": dict(zip(steps, positions)),
    "result": "PASS: explicit PowerShell order is config, environment assignment, clone. Official Git for Windows contract enables long paths for builtin Git commands; official Git documentation defines global configuration as user-scoped. Local Git setter help exposes --[no-]global; command-scoped bool query returns true.",
    "probe_adjustment": "Initial probe searched the literal --global, but this Git help renders the option as --[no-]global. The assertion now matches the actual help spelling; no reviewed claim changed.",
    "limitations": "Linux read-only check and original documentation inspection; no Windows clone or filesystem trial. core.longpaths is a Git for Windows builtin-command option, not a guarantee for other Windows tools. No user global/local configuration was written; -c applies only to the query process.",
}, ensure_ascii=False, indent=2))
