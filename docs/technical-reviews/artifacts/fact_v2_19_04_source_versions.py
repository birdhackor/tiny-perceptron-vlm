"""Bind personally inspected code to every historical 19.4 formal run."""

import hashlib
import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
COMMAND = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_04_source_versions.py"
review = json.loads((ROOT / "docs/technical-reviews/19.4.json").read_text())
audit = json.loads((ROOT / "docs/technical-reviews/artifacts/fact_v2_19_04_audit.json").read_text())
checks = []
for stage in audit["stages"]:
    for source in review["sources"]:
        if source["kind"] == "repository_code":
            current = hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            historical = stage["original_result"]["code_sha256"].get(source["path"])
            checks.append({
                "stage": stage["stage"], "path": source["path"],
                "current_sha256": current, "formal_run_sha256": historical,
                "matches": current == historical,
            })
assert len(checks) == 28 and all(item["matches"] for item in checks)
result = {
    "command": COMMAND,
    "result": "All 28 formal-run/current source SHA-256 digests exactly match.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "checks": checks,
}
output = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_04_code_version_check.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(result["result"])
