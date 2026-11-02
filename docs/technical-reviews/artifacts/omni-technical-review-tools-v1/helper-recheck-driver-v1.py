"""Same-owner bounded independent callback; no full tool/current or stage approval.

Real reports are read only. All mutations are in TemporaryDirectory or deep copies.
Author verification results are not used as assertions or substituted for execution.
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
EXPECTED_HELPER = "49cfe6cc69e12fbaccb6881c6ac525f3c0565100a5f1b942219ce8f0e8afe0f0"
INTAKE = "docs/course-revision-20261029-omni/reviews/technical-stage/numbered-intake-manifest-v3.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path.read_bytes())}


def attempt(label, action, expected_accept):
    try:
        value = action()
        return {"case": label, "accepted": True, "expected_accept": expected_accept,
                "matches_expectation": expected_accept, "result": value}
    except (ValueError, KeyError, TypeError, OSError) as error:
        return {"case": label, "accepted": False, "expected_accept": expected_accept,
                "matches_expectation": not expected_accept, "error": str(error)}


def original_context(root, old_driver, label):
    tool, manifest = old_driver.fixture(root, "valid")
    api = load("callback_fixture_api_" + label, root / "docs/review-tools/incremental_revision.py")
    validator = load("callback_fixture_validator_" + label, root / "scripts/check_technical_reviews.py")
    bundle = tool.read_ref(manifest["numbered_section_reviews"])
    entry = bundle["sections"][0]
    original = tool.read_ref(entry["original_report"])
    return tool, manifest, api, validator, bundle, entry, original


def main():
    helper_path = ROOT / "scripts/omni_revision/review_setup.py"
    assert sha(helper_path.read_bytes()) == EXPECTED_HELPER
    old_driver = load("retained_independent_fixture", OUT / "audit-driver-v1.py")
    original_results = json.loads((OUT / "mutation-results-v1.json").read_bytes())
    repeated = []
    for row in original_results["fixture_mutations"]:
        case = row["case"]
        with tempfile.TemporaryDirectory(prefix="independent-helper-repeat-") as folder:
            tool, manifest = old_driver.fixture(Path(folder), case)
            repeated.append(attempt(case, lambda: tool.check_numbered(manifest), case == "valid"))
    originals = []
    for case in ("bool_schema_version", "honest_unresolved_software", "unverified_external_source_for_unresolved", "unverified_source_used_by_verified_claim"):
        with tempfile.TemporaryDirectory(prefix="independent-helper-original-") as folder:
            root = Path(folder)
            tool, _, api, validator, _, entry, original = original_context(root, old_driver, case)
            report = copy.deepcopy(original)
            if case == "bool_schema_version":
                report["schema_version"] = True
            else:
                report["claims"] = [{"id": "blocked", "kind": "software", "status": "unresolved", "statement": "FICTIONAL software check could not run.",
                                     "location": "FICTIONAL R.1", "scope": "FICTIONAL fixture only.", "evidence": [], "artifact_ids": []}]
                for check in report["checks"].values():
                    check.update(status="revise", claim_ids=["blocked"])
                if case != "honest_unresolved_software":
                    report["sources"] = [{"id": "unavailable", "kind": "official_docs", "title": "FICTIONAL unavailable external documentation",
                                          "url": "https://example.invalid/fixture", "version": "FICTIONAL unavailable pinned version", "verified": False,
                                          "checked_original": False, "accessed_on": "2026-10-20", "authority_reason": "FICTIONAL negative fixture, no network access attempted.",
                                          "inspection_note": "FICTIONAL failed access, unresolved; no claim of actually checking this external source."}]
                    report["claims"][0]["evidence"] = [{"source_id": "unavailable", "locator": "FICTIONAL failed access", "supports": "Unresolved; no external support asserted."}]
                    if case == "unverified_source_used_by_verified_claim":
                        report["claims"][0]["status"] = "verified"
            before = copy.deepcopy(report)
            result = attempt(case, lambda: tool.check_original(report, entry["original_section_freeze"], "R.1", "course", report["reviewer_task"], validator, api),
                             case in {"honest_unresolved_software", "unverified_external_source_for_unresolved"})
            result["input_report_unmodified"] = report == before
            originals.append(result)
    reconstruction = []
    chapter_old = "# FICTIONAL chapter 99\n\n## 99.1 First\n\nAlpha.\n\n## 99.2 Second\n\nBeta.\n"
    chapter_new = chapter_old.replace("Beta.", "Changed current Beta.")
    chapter_cases = ("complete_chapter", "missing_section", "wrong_component_sha", "dependency_not_in_freeze", "missing_introduction", "wrong_reported_digest", "duplicate_section", "wrong_selector", "reordered_sections", "whole_md", "whole_md_wrong_sha", "whole_md_not_in_freeze", "python_no_exemption")
    for case in chapter_cases:
        with tempfile.TemporaryDirectory(prefix="independent-original-document-") as folder:
            root = Path(folder)
            tool, manifest, api, validator, _, entry, original = original_context(root, old_driver, case)
            canonical = tool.read_ref(manifest["source_freeze"])
            report = copy.deepcopy(original)
            dependency = "course/chapters/99.md"
            old_bytes = chapter_old.encode()
            if case.startswith("whole_md"):
                dependency = "course/linked-whole.md"
            if case == "python_no_exemption":
                dependency = "scripts/linked-dependency.py"
                old_bytes = b"# FICTIONAL older implementation\n"
            old_driver.save(root, dependency, chapter_new.encode() if case != "python_no_exemption" else b"# FICTIONAL changed implementation\n")
            pages = []
            if case.startswith("whole_md") or case == "python_no_exemption":
                snapshot = old_driver.save(root, "frozen/dependency-whole.txt", old_bytes)
                pages = [{"page_id": "linked", "source": dependency, "selector": "whole-file", "snapshot": snapshot["path"], "source_sha256": snapshot["sha256"], "figures_sha256": {}}]
            else:
                reading = load("callback_document_slices_" + case, root / "scripts/reading_time.py")
                parts = [("chapter-99", "before-first-section", reading.chapter_introduction(chapter_old))]
                parts.extend((sid, "section:" + sid, text) for sid, text in reading.lesson_slices(chapter_old).items())
                for sid, selector, text in parts:
                    snapshot = old_driver.save(root, "frozen/" + sid + ".md", text.encode())
                    pages.append({"page_id": sid, "source": dependency, "selector": selector, "snapshot": snapshot["path"], "source_sha256": snapshot["sha256"], "figures_sha256": {}})
            if case == "missing_section":
                pages = pages[:-1]
            elif case in {"wrong_component_sha", "whole_md_wrong_sha"}:
                pages[-1]["source_sha256"] = "0" * 64
            elif case in {"dependency_not_in_freeze", "whole_md_not_in_freeze"}:
                pages = []
            elif case == "missing_introduction":
                pages = pages[1:]
            elif case == "duplicate_section":
                pages.append(copy.deepcopy(pages[-1]))
            elif case == "wrong_selector":
                pages[-1]["selector"] = "section:99.1"
            elif case == "reordered_sections":
                pages[1], pages[2] = pages[2], pages[1]
            canonical["pages"].extend(pages)
            canonical_ref = old_driver.save(root, "frozen/canonical-with-dependency.json", canonical)
            auxiliary = tool.read_ref(entry["original_section_freeze"])
            auxiliary["canonical_source_freeze"] = canonical_ref
            auxiliary_ref = old_driver.save(root, "frozen/auxiliary-with-dependency.json", auxiliary)
            report["sources"] = [{"id": "historical_doc", "kind": "repository_code", "title": "FICTIONAL historical document dependency",
                                  "path": dependency, "sha256": "1" * 64 if case == "wrong_reported_digest" else sha(old_bytes),
                                  "version": "FICTIONAL original bytes retained in canonical freeze", "verified": True,
                                  "inspection_note": "FICTIONAL historical dependency, no claim of real course review."}]
            before = copy.deepcopy(report)
            result = attempt(case, lambda: tool.check_original(report, auxiliary_ref, "R.1", "course", report["reviewer_task"], validator, api),
                             case in {"complete_chapter", "whole_md"})
            result["input_report_unmodified"] = before == report
            result["old_digest"] = sha(old_bytes)
            result["live_digest"] = sha((root / dependency).read_bytes())
            reconstruction.append(result)
    # Real T.5 original: independently rehash and concatenate actual original-freeze components.
    tool = load("callback_actual_helper", helper_path)
    api = load("callback_actual_api", ROOT / "docs/review-tools/incremental_revision.py")
    validator = load("callback_actual_validator", ROOT / "scripts/check_technical_reviews.py")
    manifest = json.loads((ROOT / INTAKE).read_bytes())
    bundle = tool.read_ref(manifest["numbered_section_reviews"])
    t5_entry = next(row for row in bundle["sections"] if row["lesson_id"] == "T.5")
    t5 = tool.read_ref(t5_entry["original_report"])
    t5_original_freeze = tool.read_ref(t5_entry["original_section_freeze"])
    t5_canonical = tool.read_ref(t5_original_freeze["canonical_source_freeze"])
    components = [p for p in t5_canonical["pages"] if p["source"] == "course/chapters/09.md"]
    intro = next(p for p in components if p["selector"] == "before-first-section")
    section_rows = [p for p in components if p["selector"].startswith("section:")]
    raw_parts = []
    component_receipts = []
    for page in [intro, *section_rows]:
        raw = api.artifact(ROOT, {"path": page["snapshot"], "sha256": page["source_sha256"]})
        raw_parts.append(raw)
        component_receipts.append({"page_id": page["page_id"], "path": page["snapshot"], "sha256": sha(raw)})
    reconstructed = b"".join(raw_parts)
    old_source = next(s for s in t5["sources"] if s.get("path") == "course/chapters/09.md")
    actual = {"original_report": t5_entry["original_report"], "original_section_freeze": t5_entry["original_section_freeze"],
              "original_canonical_freeze": t5_original_freeze["canonical_source_freeze"], "components": component_receipts,
              "reconstructed_sha256": sha(reconstructed), "reported_dependency_sha256": old_source["sha256"],
              "current_dependency_sha256": sha((ROOT / "course/chapters/09.md").read_bytes())}
    actual["original_validation"] = attempt("actual_T5_original", lambda: tool.check_original(t5, t5_entry["original_section_freeze"], "T.5", "training", t5["reviewer_task"], validator, api), True)
    old_current_path = ROOT / t5_entry["previous_current_reports"][0]["path"]
    old_current = tool.read_ref(t5_entry["previous_current_reports"][0])
    current = tool.read_ref(t5_entry["report"])
    sections_freeze = tool.read_ref(bundle["section_freeze"])
    t5_page = next(p for p in sections_freeze["pages"] if p["page_id"] == "T.5")
    body = tool.frozen_body(t5_page)
    readers = {row["task"] for row in tool.read_ref(manifest["assignments"])["tasks"] if row["role"] == "reader"}
    _, owners = api.technical_reports(manifest, ROOT)
    for entry in bundle["sections"]:
        current_report = tool.read_ref(entry["report"])
        owners.setdefault(current_report["reviewer_task"], []).append(entry["lesson_id"])
    actual["old_current"] = {"report": ref(old_current_path), "errors": validator._validate(ROOT, ROOT / t5_page["source"], "T.5", body, old_current, owners, readers)}
    actual["current_v2"] = {"report": t5_entry["report"], "errors": validator._validate(ROOT, ROOT / t5_page["source"], "T.5", body, current, owners, readers)}
    intake_result = attempt("actual_source08_intake", lambda: tool.check_numbered(manifest, intake=True, manifest_path=INTAKE), True)
    sealed_result = attempt("actual_source08_is_not_sealed_approval", lambda: tool.check_numbered(manifest, manifest_path=INTAKE), False)
    current_freeze = tool.read_ref(manifest["source_freeze"])
    inventory = tool.module("callback_actual_inventory", "docs/review-tools/composite_review.py").current_inventory(ROOT)
    diff = tool.compare(current_freeze, inventory)
    original = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_review_tools/technical-original-v1.json"
    checks = [*repeated, *originals, *reconstruction, actual["original_validation"], intake_result, sealed_result]
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK, "kind": "bounded_independent_helper_recheck_not_full_tool_current",
              "command": "uv run --extra cpu python " + str(Path(__file__).relative_to(ROOT)),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; bounded offline metadata and temporary mutations"},
              "helper": ref(helper_path), "retained_original": ref(original),
              "repeated_original_30_cases": repeated, "original_status_boundary_cases": originals, "document_reconstruction_cases": reconstruction,
              "actual_T5": actual, "actual_intake": {"manifest": ref(ROOT / INTAKE), "source_freeze": manifest["source_freeze"], "bundle": manifest["numbered_section_reviews"],
              "section_freeze": bundle["section_freeze"], "assignments": manifest["assignments"], "issues": bundle["issue_inventory"],
              "intake": intake_result, "sealed": sealed_result, "actual_scope_count": len(bundle["sections"]), "intake_only_declared": manifest.get("technical_intake_only"),
              "intake_note": manifest.get("intake_note"), "source08_to_current_changed_page_ids": list(diff["changed"])},
              "summary": {"case_count": len(checks), "unexpected": [row for row in checks if not row["matches_expectation"]],
                          "actual_T5_reconstruction_equals_report": sha(reconstructed) == old_source["sha256"],
                          "stale_current_still_rejected": bool(actual["old_current"]["errors"]), "current_v2_metadata_errors": actual["current_v2"]["errors"]},
              "pending": ["TOOLS-W1-INTEGRATION-PENDING-02 remains open: final source11/main26, four whole collectors, actual original/current receipts and final invoked recursion not yet verified.",
                          "No overall current/pass, no final 127 collection, no final metadata certification or historical fix-chain closure in this bounded callback."],
              "helper_after_execution": ref(helper_path)}
    with (OUT / "helper-recheck-mutations-v1.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"helper": output["helper"], "summary": output["summary"], "accepted_repeated_cases": [row["case"] for row in repeated if row["accepted"]],
                      "actual_T5": {k: v for k, v in actual.items() if k != "components"}, "actual_intake": output["actual_intake"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
