"""Bounded callback for original two issues; no expanded full-audit claim."""

import ast
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
EXPECTED_HELPER = "701a87eb97fe51d2449b88602c4814236bb73297dce1494964e6102ff6684ad0"
TASK = "/root/omni_technical_coordinator/technical_review_tools"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    helper = ROOT / "scripts/omni_revision/review_setup.py"
    assert sha(helper.read_bytes()) == EXPECTED_HELPER
    retained_path = OUT / "helper-recheck-driver-v1.py"
    retained = load("retained_same_owner_tests", retained_path)
    retained.EXPECTED_HELPER = EXPECTED_HELPER
    parsed = ast.parse(retained_path.read_text())
    main = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # Execute exactly the retained independent fictional tests, stopping before old actual source08 data.
    end = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "tool" for target in node.targets)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
               and node.value.func.id == "load" and node.value.args
               and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == "callback_actual_helper")
    executable = ast.Module(body=main.body[:end], type_ignores=[])
    exec(compile(executable, str(retained_path) + "#fictional-prefix", "exec"), retained.__dict__)
    prior_cases = [*retained.repeated, *retained.originals, *retained.reconstruction]
    driver = retained.old_driver
    results = []
    cases = ("authenticated_original_code", "missing_snapshot", "wrong_snapshot_bytes", "wrong_snapshot_reference_hash", "wrong_source_name",
             "duplicate_snapshot", "unregistered_original_hash", "snapshot_missing_file", "outside_snapshot", "malformed_snapshot_fields",
             "snapshots_not_list", "borrow_another_source_digest", "verified_claim_still_needs_execution", "stale_current_still_strict")
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="independent-original-snapshot-") as folder:
            root = Path(folder)
            tool, manifest, api, validator, bundle, entry, original = retained.original_context(root, driver, case)
            report = copy.deepcopy(original)
            old_bytes = b"# FICTIONAL original repository implementation\n"
            source = "scripts/dependency.py"
            driver.save(root, source, b"# FICTIONAL changed current implementation\n")
            snapshot = driver.save(root, "preserved/dependency-original.py", old_bytes)
            report["sources"] = [{"id": "repository", "kind": "repository_code", "title": "FICTIONAL original code dependency",
                                  "path": source, "sha256": sha(old_bytes), "version": "FICTIONAL original bytes", "verified": True,
                                  "inspection_note": "FICTIONAL fixture only. True original bytes declared before the simulated current change."}]
            snapshots = [{"source": source, "snapshot": snapshot}]
            if case == "missing_snapshot":
                snapshots = []
            elif case == "wrong_snapshot_bytes":
                snapshots[0]["snapshot"] = driver.save(root, snapshot["path"], b"# FICTIONAL other bytes\n")
            elif case == "wrong_snapshot_reference_hash":
                snapshots[0]["snapshot"] = {**snapshot, "sha256": "0" * 64}
            elif case == "wrong_source_name":
                snapshots[0]["source"] = "scripts/unregistered.py"
            elif case == "duplicate_snapshot":
                snapshots.append(copy.deepcopy(snapshots[0]))
            elif case == "unregistered_original_hash":
                report["sources"][0]["sha256"] = "1" * 64
            elif case == "snapshot_missing_file":
                (root / snapshot["path"]).unlink()
            elif case == "outside_snapshot":
                snapshots[0]["snapshot"]["path"] = "../outside.py"
            elif case == "malformed_snapshot_fields":
                snapshots[0]["extra"] = "unregistered"
            elif case == "snapshots_not_list":
                snapshots = {"source": source, "snapshot": snapshot}
            elif case == "borrow_another_source_digest":
                second_source = "scripts/second.py"
                second_bytes = b"# FICTIONAL second original code\n"
                driver.save(root, second_source, b"# FICTIONAL changed second implementation\n")
                report["sources"].append({**report["sources"][0], "id": "second", "path": second_source, "sha256": sha(second_bytes)})
                snapshots[0]["source"] = second_source
            elif case == "verified_claim_still_needs_execution":
                report["claims"] = [{"id": "bad", "kind": "software", "status": "verified", "statement": "FICTIONAL software claim without evidence.",
                                     "location": "FICTIONAL R.1", "scope": "FICTIONAL fixture only."}]
            elif case == "stale_current_still_strict":
                report["verdict"] = "pass"
                report["issues"] = []
            before = copy.deepcopy(report)
            if case == "stale_current_still_strict":
                body = tool.frozen_body(next(row for row in tool.read_ref(entry["original_section_freeze"])["pages"] if row["page_id"] == "R.1"))
                errors = validator._validate(root, root / "course/README.md", "R.1", body, report,
                                             {report["reviewer_task"]: ["R.1"]}, set())
                result = {"case": case, "accepted": not errors, "expected_accept": False,
                          "matches_expectation": any("證據版本已改變" in error for error in errors), "errors": errors}
            else:
                result = retained.attempt(case, lambda: tool.check_original(report, entry["original_section_freeze"], "R.1", "course", report["reviewer_task"],
                                                                           validator, api, original_source_snapshots=snapshots), case == "authenticated_original_code")
            result["input_report_unmodified"] = before == report
            results.append(result)
    all_cases = [*prior_cases, *results]
    originals = [ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_review_tools" / name
                 for name in ("technical-original-v1.json", "helper-recheck-v1.json")]
    data = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK,
            "kind": "bounded_two_original_issue_code_mutations_not_full_tool_audit",
            "command": "uv run --extra cpu python " + Path(__file__).relative_to(ROOT).as_posix(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; isolated temporary fixtures"},
            "helper": ref(helper), "retained_independent_driver": ref(retained_path),
            "retained_fictional_prefix_cases": prior_cases, "new_original_snapshot_cases": results,
            "summary": {"case_count": len(all_cases), "unexpected": [item for item in all_cases if not item["matches_expectation"]],
                        "old_schema_counterexamples_rejected": all(not item["accepted"] for item in prior_cases if item["case"] in {"original_claim_without_evidence", "original_sources_malformed", "bool_schema_version"}),
                        "original_dicts_unchanged": all(item.get("input_report_unmodified", True) for item in all_cases)},
            "retained_reports": [ref(path) for path in originals], "helper_after_execution": ref(helper),
            "scope": "61 independent bounded cases. Original repository snapshots need both immutable original path/digest and actual matching saved bytes; current still strict. No real source/report/checker/seal modified and no expanded-scope audit claimed."}
    target = OUT / "local-fix-code-mutations-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": ref(target), "helper": data["helper"], "summary": data["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
