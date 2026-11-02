"""Actual bounded W.7 callback: rerun the original algorithm with v11 inputs."""

import datetime
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_7/technical-original-v1.json"
EXPECTED = "bd5e74c9bf7cb3f815d8194e06fb97966299075df09ba870f0ad0544486ccf9a"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = datetime.datetime.now(datetime.UTC).isoformat()
    raw_names = ["audit_w7.py", "assemble_report.py", "route-inspection.json", "cpu-verification.json",
                 "necessary-current-slices.json", "original-source-excerpts.json"]
    before = {name: digest(HERE / name) for name in raw_names}
    assert digest(ORIGINAL) == EXPECTED
    spec = importlib.util.spec_from_file_location("w7_original_audit_algorithm", HERE / "audit_w7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v11.json"
    module.SNAPSHOT = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v11/sections/W.7.md"

    def save_current(name, value):
        filename = name.removesuffix(".json") + "-current-v1.json"
        with (HERE / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    module.save = save_current
    module.main()
    assert before == {name: digest(HERE / name) for name in raw_names}
    assert digest(ORIGINAL) == EXPECTED
    result = {"started_at": started, "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
              "reviewer_task": "/root/omni_technical_coordinator/technical_w_7",
              "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-w-7-v1/audit_w7_current_v1.py",
              "method": "Import the immutable original audit algorithm, explicitly substitute v11 FREEZE/SNAPSHOT and exclusive-create current outputs, then run its whole main: 65 routes, 58 exact AST definitions, W.7 inline pathlib, exceptions, and observed no-update CPU dry-run.",
              "freeze": {"path": str(module.FREEZE.relative_to(ROOT)), "sha256": digest(module.FREEZE)},
              "snapshot": {"path": str(module.SNAPSHOT.relative_to(ROOT)), "sha256": digest(module.SNAPSHOT)},
              "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": EXPECTED},
              "original_raw_sha256_unchanged": before,
              "current_outputs": {name: digest(HERE / name) for name in ["route-inspection-current-v1.json", "cpu-verification-current-v1.json"]}}
    save_current("bounded-run.json", result)


if __name__ == "__main__":
    main()
