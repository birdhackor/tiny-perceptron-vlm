"""Actual source14/26-section recursion and four whole metadata; isolated overlays only.

No sealed real collection is invented. A local-path overlay only redirects selected
JSON/snapshot reads to temporary bytes; unmodified helper SHA checks and validators
still execute. All unaffected reads use the real repository.
"""

import copy
import hashlib
import importlib.util
import json
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = "/root/omni_technical_coordinator/technical_review_tools"
MANIFEST = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v5.json"
EXPECTED_HELPER = "701a87eb97fe51d2449b88602c4814236bb73297dce1494964e6102ff6684ad0"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert sha((ROOT / "scripts/omni_revision/review_setup.py").read_bytes()) == EXPECTED_HELPER
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    base_tool = load("actual_graph_base", ROOT / "scripts/omni_revision/review_setup.py")
    base_bundle = base_tool.read_ref(manifest["numbered_section_reviews"])
    base_index = base_tool.read_ref(base_bundle["issue_inventory"])
    current_canonical = base_tool.read_ref(manifest["source_freeze"])
    current_sections = base_tool.read_ref(base_bundle["section_freeze"])
    w1 = next(row for row in base_bundle["sections"] if row["lesson_id"] == "W.1")
    cases = ("actual_intake_positive", "W1_current_behind_unchanged_bundle", "W1_original_behind_unchanged_bundle", "W1_bad_artifact_rebound_report",
             "W1_bad_source_identity_rebound_report", "W1_author_owner_rebound_report", "W1_drop_original_snapshots", "W1_bad_original_snapshot_ref",
             "drop_W1_entry", "omit_W1_issue_from_index", "wrong_W1_issue_pointer", "drop_firststeps_from_scope", "W1_section_snapshot_drift", "firststeps_canonical_snapshot_drift")
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="actual-W1-isolated-overlay-") as folder:
            temp = Path(folder)
            tool = load("actual_graph_" + case, ROOT / "scripts/omni_revision/review_setup.py")
            overlay = {}
            original_local = tool.local

            def local(name):
                return overlay.get(str(name), original_local(name))

            tool.local = local
            draft = copy.deepcopy(manifest)
            bundle = copy.deepcopy(base_bundle)
            index = copy.deepcopy(base_index)
            entry = next(row for row in bundle["sections"] if row["lesson_id"] == "W.1")

            def put(path, value):
                raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
                target = temp / str(len(overlay))
                target.write_bytes(raw)
                overlay[path] = target
                return {"path": path, "sha256": sha(raw)}

            if case in {"W1_current_behind_unchanged_bundle", "W1_original_behind_unchanged_bundle"}:
                key = "report" if case == "W1_current_behind_unchanged_bundle" else "original_report"
                report = base_tool.read_ref(entry[key])
                report["isolated_mutation"] = True
                put(entry[key]["path"], report)  # Retain original ref SHA to test transitive drift.
            elif case in {"W1_bad_artifact_rebound_report", "W1_bad_source_identity_rebound_report", "W1_author_owner_rebound_report"}:
                report = base_tool.read_ref(entry["report"])
                if case == "W1_bad_artifact_rebound_report":
                    report["artifacts"][0]["sha256"] = "0" * 64
                elif case == "W1_bad_source_identity_rebound_report":
                    report["source_sha256"] = "0" * 64
                else:
                    report["reviewer_task"] = "/root"
                entry["report"] = put(entry["report"]["path"], report)
                draft["numbered_section_reviews"] = put(draft["numbered_section_reviews"]["path"], bundle)
            elif case == "W1_drop_original_snapshots":
                entry["original_source_snapshots"] = []
                draft["numbered_section_reviews"] = put(draft["numbered_section_reviews"]["path"], bundle)
            elif case == "W1_bad_original_snapshot_ref":
                entry["original_source_snapshots"][0]["snapshot"]["sha256"] = "0" * 64
                draft["numbered_section_reviews"] = put(draft["numbered_section_reviews"]["path"], bundle)
            elif case == "drop_W1_entry":
                bundle["sections"] = [row for row in bundle["sections"] if row["lesson_id"] != "W.1"]
                draft["numbered_section_reviews"] = put(draft["numbered_section_reviews"]["path"], bundle)
            elif case in {"omit_W1_issue_from_index", "wrong_W1_issue_pointer"}:
                found = next(i for i, issue in enumerate(index["issues"]) if issue["lesson_id"] == "W.1")
                if case == "omit_W1_issue_from_index":
                    del index["issues"][found]
                else:
                    index["issues"][found]["json_pointer"] = "/issues/999"
                bundle["issue_inventory"] = put(bundle["issue_inventory"]["path"], index)
                draft["numbered_section_reviews"] = put(draft["numbered_section_reviews"]["path"], bundle)
            elif case == "drop_firststeps_from_scope":
                draft["numbered_section_requirements"]["page_ids"].remove("first-steps")
            elif case in {"W1_section_snapshot_drift", "firststeps_canonical_snapshot_drift"}:
                frozen = current_sections if case == "W1_section_snapshot_drift" else current_canonical
                pid = "W.1" if case == "W1_section_snapshot_drift" else "first-steps"
                page = next(row for row in frozen["pages"] if row["page_id"] == pid)
                put(page["snapshot"], (ROOT / page["snapshot"]).read_bytes() + b"\nISOLATED DRIFT\n")
            trace = {"native_current_ids": [], "original_ids": []}
            module = tool.module

            def traced_module(name, relative):
                value = module(name, relative)
                if relative == "scripts/check_technical_reviews.py":
                    validate = value._validate

                    def recorded_validate(*args, **kwargs):
                        trace["native_current_ids"].append(args[2])
                        return validate(*args, **kwargs)

                    value._validate = recorded_validate
                return value

            tool.module = traced_module
            check_original = tool.check_original

            def traced_original(*args, **kwargs):
                trace["original_ids"].append(args[2])
                return check_original(*args, **kwargs)

            tool.check_original = traced_original
            try:
                value = tool.check_numbered(draft, intake=True, manifest_path=MANIFEST)
                result = {"case": case, "accepted": True, "result": value}
            except (ValueError, KeyError, TypeError, OSError) as error:
                result = {"case": case, "accepted": False, "error": str(error)}
            result["matches_expectation"] = result["accepted"] == (case == "actual_intake_positive")
            result["trace"] = trace
            results.append(result)
    api = load("actual_graph_api", ROOT / "docs/review-tools/incremental_revision.py")
    validator = load("actual_graph_validator", ROOT / "scripts/check_technical_reviews.py")
    assignments = base_tool.read_ref(manifest["assignments"])["tasks"]
    roles = {row["task"]: row for row in assignments}
    readers = {row["task"] for row in assignments if row["role"] == "reader"}
    owners = {}
    for entry in base_bundle["sections"]:
        report = base_tool.read_ref(entry["report"])
        owners.setdefault(report["reviewer_task"], []).append(entry["lesson_id"])
    whole_paths = {"course": "technical_course/technical-current-v2.json", "training": "technical_training/technical-current-v2.json",
                   "first-steps": "technical_first_steps/technical-current-v2.json", "glossary": "technical_glossary/technical-current-v3.json"}
    whole_reports = {}
    for pid, relative in whole_paths.items():
        path = ROOT / "docs/course-revision-20261029-omni/reviews/technical" / relative
        report = json.loads(path.read_bytes())
        whole_reports[pid] = (path, report)
        owners.setdefault(report["reviewer_task"], []).append(pid)
    whole = []
    for pid, (path, report) in whole_reports.items():
        page = next(row for row in current_canonical["pages"] if row["page_id"] == pid)
        errors = validator._validate(ROOT, ROOT / page["source"], pid, base_tool.frozen_body(page), report, owners, readers)
        assignment = roles.get(report["reviewer_task"], {})
        whole.append({"page_id": pid, "report": ref(path), "reviewer_task": report["reviewer_task"], "created_at": report["created_at"],
                      "source_sha256": report["source_sha256"], "source14_sha256": page["source_sha256"], "native_validation_errors": errors,
                      "bound_to_exact_bundle_v4": any(item.get("path") == manifest["numbered_section_reviews"]["path"] and item.get("sha256") == manifest["numbered_section_reviews"]["sha256"] for item in report["artifacts"]),
                      "actual_fresh_role_and_lesson": assignment.get("role") == "technical" and assignment.get("context") == "fresh" and assignment.get("lesson_id") == pid,
                      "author_excluded": report["reviewer_task"] not in manifest["author_tasks"], "original_report": report.get("original_report"),
                      "receipt_status": "Awaiting immutable actual collection index; no completed stage assumed."})
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
              "kind": "bounded_actual_W1_recursion_and_four_whole_metadata_not_sealed_graph",
              "command": "uv run --extra cpu python " + Path(__file__).relative_to(ROOT).as_posix(),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; read-only actual graph and isolated path overlays"},
              "helper": ref(ROOT / "scripts/omni_revision/review_setup.py"), "manifest": ref(ROOT / MANIFEST), "source_freeze": manifest["source_freeze"],
              "section_freeze": base_bundle["section_freeze"], "bundle": manifest["numbered_section_reviews"], "issue_index": base_bundle["issue_inventory"],
              "assignments": manifest["assignments"], "actual_W1": w1, "cases": results, "four_whole": whole,
              "summary": {"case_count": len(results), "unexpected": [row for row in results if not row["matches_expectation"]],
                          "positive_original_count": len(results[0]["trace"]["original_ids"]), "positive_native_current_count": len(results[0]["trace"]["native_current_ids"]),
                          "four_whole_metadata_valid": all(not row["native_validation_errors"] and row["bound_to_exact_bundle_v4"] and row["actual_fresh_role_and_lesson"] and row["author_excluded"] for row in whole)},
              "scope": "Actual intake traces 26 current native validations and 26 original checks, including W1; 13 negatives exercise transitive refs/source/artifact/role/issue binding. Four actual whole current reports independently pass metadata and bind exact bundle. No real reports mutated; no fake collection seal or final receipt/fullgraph approval."}
    target = OUT / "local-fix-graph-mutations-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": ref(target), "summary": output["summary"], "four_whole": whole}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
