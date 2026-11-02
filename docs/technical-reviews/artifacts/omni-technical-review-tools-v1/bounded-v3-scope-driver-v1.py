"""Metadata clarification of already executed v2 data scope; no probe rerun."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REVIEW = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_review_tools"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def main():
    previous = REVIEW / "bounded-tool-fix-verification-v2.json"
    assert sha(previous.read_bytes()) == "bc53df4c634c812b6daa5db2c3ffa6ce33c7c5859752d06b208ae0529a0de1c8"
    v2 = json.loads(previous.read_bytes())
    decision_path = ROOT / "docs/course-revision-20261029-omni/reviews/technical-author-tools-fix01-decisions-v3.json"
    decision = json.loads(decision_path.read_bytes())
    receipt = ROOT / "docs/course-revision-20261029-omni/reviews/coherence-stage/old-tools-bounded-followup-receipt-v1.json"
    assert sha(receipt.read_bytes()) == "de4996ccf3f6479e45c461064a15ea206bb27118a1d8d1a55704a37fbd9536e5"
    actual_checked = []
    for entry in v2["checked_files"]:
        actual = ref(ROOT / entry["path"])
        assert actual == entry
        actual_checked.append(actual)
    actual_data = {key: ref(ROOT / v2["verified_inputs"][key]["path"]) for key in ("bundle", "issue_index")}
    for key, reference in actual_data.items():
        assert reference == v2["verified_inputs"][key]
    execution_path = OUT / "sealed-local-fix-results-v1.json"
    execution = json.loads(execution_path.read_bytes())
    assert execution["bundle"] == actual_data["bundle"] and execution["issue_index"] == actual_data["issue_index"]
    assert execution["summary"]["native_original_passes"] == execution["summary"]["native_current_passes"] == execution["summary"]["native_sealed_report_pair_passes"] == 30
    by_issue = {
        "TOOLS-ORIGINAL-SCHEMA-01": {
            "original_json_pointer": "/issues/0",
            "affected_files": ["scripts/omni_revision/review_setup.py"],
            "affected_files_sha256": {entry["path"]: entry["sha256"] for entry in actual_checked if entry["path"] == "scripts/omni_revision/review_setup.py"},
            "evidence_scope": "Original schema/software guard correction, independently executed in v2's retained61 probes; code bytes unchanged.",
        },
        "TOOLS-W1-INTEGRATION-PENDING-02": {
            "original_json_pointer": "/issues/1",
            "affected_files": [actual_data["bundle"]["path"], actual_data["issue_index"]["path"]],
            "affected_files_sha256": {value["path"]: value["sha256"] for value in actual_data.values()},
            "evidence_scope": "Actual source15 binding data:26section original/current references,4whole bundle links, exact issue index and real sealed30 pairs were already inspected/executed in v2. This is checked data, not a new full implementation audit.",
        },
    }
    for item in decision["items"]:
        match = next(value for value in by_issue.values() if value["original_json_pointer"] == item["json_pointer"])
        assert item["affected_files"] == match["affected_files"]
        assert item.get("affected_pages", []) == []
    data = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": v2["reviewer_task"],
            "command": "uv run --extra cpu python " + Path(__file__).relative_to(ROOT).as_posix(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; metadata and SHA consistency only"},
            "kind": "metadata_clarification_of_already_checked_v2_data_scope_not_new_tests",
            "previous_v2": ref(previous), "actual_v2_continuity_receipt": ref(receipt), "author_decision_input": ref(decision_path),
            "prior_native_execution": ref(execution_path), "unchanged_v2_checked_files": actual_checked,
            "already_verified_data_promoted_to_checked_files": list(actual_data.values()), "allowed_scope_by_issue": by_issue,
            "result": "All exact bytes match v2's actual execution inputs and current files. Bundle5/index5 were genuinely checked data; v2 checked_files/affected scope omitted them. Author intended affected-file sets match the two precise per-issue sets. No61-code or30-pair execution repeated.",
            "scope_boundary": "Only declare actual prior data inspection accurately. Not manifest.tool_audit/full210+fresh original/historical full_review/globalgate/activation; no scope containment guard waived."}
    path = OUT / "bounded-v3-scope-results-v1.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": ref(path), "allowed_scope_by_issue": by_issue, "result": data["result"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
