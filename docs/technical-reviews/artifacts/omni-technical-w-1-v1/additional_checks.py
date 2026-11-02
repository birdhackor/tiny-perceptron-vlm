"""Isolated optional Git LFS setup and default uv-path checks for W.1."""

import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

root = Path(__file__).resolve().parents[4]
artifacts = Path(__file__).resolve().parent
scratch = Path("/workspace/work/omni-technical-w-1-v1/git-lfs-local-check")
scratch.mkdir(exist_ok=True)
env = {**os.environ, "GIT_CONFIG_GLOBAL": str(scratch / "isolated-global-config"), "GIT_CONFIG_NOSYSTEM": "1"}
runs = []
for command, cwd in [
    (["git", "lfs", "version"], root),
    (["git", "init"], scratch),
    (["git", "lfs", "install", "--local"], scratch),
    (["git", "config", "--local", "--get", "filter.lfs.process"], scratch),
    (["uv", "sync", "--frozen", "--extra", "cpu", "--group", "notebook", "--dry-run"], root),
]:
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=True)
    runs.append({"command": shlex.join(command), "cwd": str(cwd), "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
assert runs[0]["stdout"].startswith("git-lfs/")
assert runs[3]["stdout"].strip() == "git-lfs filter-process"
assert ".venv" in runs[-1]["stderr"] + runs[-1]["stdout"]
source = Path("/workspace/work/reference/omni-technical-w-1-v1/git-longpath.txt")
literal = int(re.search(r"MAX_PATH = (\d+)", source.read_text())[1])
assert literal == 260
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "environment": {"platform": "Linux", "git": "2.52.0", "git_lfs": runs[0]["stdout"].strip(), "uv": "0.12.19"},
    "isolation": {"GIT_CONFIG_GLOBAL": env["GIT_CONFIG_GLOBAL"], "GIT_CONFIG_NOSYSTEM": "1", "note": "No commit, no shared repository Git config changes, no training assets fetched."},
    "runs": runs,
    "numeric_source_literal": {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "locator": "compat/mingw.h:106", "MAX_PATH_documented_value": literal, "scope": "Original Windows-specific source literal, not a Windows runtime experiment; the same comment distinguishes directory limit 248."},
}
(artifacts / "additional-execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"passed": True, "git_lfs": runs[0]["stdout"].strip(), "local_filter": runs[3]["stdout"].strip(), "MAX_PATH_source_literal": literal}))
