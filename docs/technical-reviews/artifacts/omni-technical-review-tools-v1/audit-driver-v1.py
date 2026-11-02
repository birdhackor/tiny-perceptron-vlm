"""Independent, bounded tool audit. All mutation reports/seals are fictional temp fixtures.

No real course source, review, checker, active pointer, or implementation is rewritten.
"""

import copy
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK = "/root/omni_technical_coordinator/technical_review_tools"
MANIFEST = "docs/course-revision-20261029-omni/reviews/assembly-reader-01/manifest-reader.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return {"path": name, "sha256": digest(raw)}


def ref(root, name):
    return {"path": name, "sha256": digest((root / name).read_bytes())}


def report(sid, source, body, task, *, original=False):
    return {
        "schema_version": 1,
        "review_stage": "technical",
        "lesson_id": sid,
        "source": source,
        "reviewer_task": task,
        "reviewer_context": "fresh",
        "source_sha256": digest(body.encode()),
        "figure_sha256": {},
        "created_at": "2026-10-20T02:02:00+00:00" if original else "2026-10-20T02:04:00+00:00",
        "verdict": "pass",
        "claims": [],
        "sources": [],
        "artifacts": [],
        "issues": [],
        "applicability": {"substantive_claims": False, "reason": "FICTIONAL navigation-only test fixture."},
        "checks": {name: {"status": "not_applicable", "details": "FICTIONAL navigation fixture, no substantive claims.", "claim_ids": []}
                   for name in ("factual_accuracy", "numeric_verification", "figure_consistency", "source_verification", "limitations")},
    }


def fixture(root, case):
    for name in ("scripts/reading_time.py", "scripts/check_technical_reviews.py", "docs/review-tools/incremental_revision.py"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    tool = load("fictional_omni_" + case, ROOT / "scripts/omni_revision/review_setup.py")
    tool.ROOT = root
    source = "course/README.md"
    body = "# FICTIONAL navigation\n\n## R.1 Destination\n\nNew navigation.\n\n## R.2 More\n\nNext navigation.\n"
    before = body.replace("New navigation.", "Prior navigation.")
    save(root, source, body.encode())
    snapshot = save(root, "freeze/current-body.md", body.encode())
    prior_snapshot = save(root, "freeze/old-body.md", before.encode())
    page = {"page_id": "course", "selector": "whole-file", "source": source,
            "snapshot": snapshot["path"], "source_sha256": snapshot["sha256"], "figures_sha256": {}}
    canonical = save(root, "freeze/current.json", {"pages": [page], "figure_snapshots": {}})
    prior_page = {**page, "snapshot": prior_snapshot["path"], "source_sha256": prior_snapshot["sha256"]}
    prior = save(root, "freeze/prior.json", {"pages": [prior_page], "figure_snapshots": {}})
    accepted = save(root, "accepted/manifest.json", {"source_freeze": prior})
    # In-memory fixture certificate only; the real fixed certificate is separately tested below.
    tool.ACCEPTED_BASELINE = accepted
    reading = load("fixture_reading_" + case, root / "scripts/reading_time.py")
    sections = reading.lesson_slices(body)
    frozen = []
    for sid, text in sections.items():
        snap = save(root, "freeze/sections/" + sid + ".md", text.encode())
        frozen.append({"page_id": sid, "canonical_page_id": "course", "source": source,
                       "selector": "section:" + sid, "snapshot": snap["path"],
                       "source_sha256": snap["sha256"], "figures_sha256": {}})
    auxiliary = save(root, "freeze/sections.json", {
        "kind": "auxiliary_numbered_technical_freeze_not_canonical", "canonical_source_freeze": canonical,
        "canonical_page_ids": ["course"], "pages": frozen})
    entries, current_reports, originals, tasks = [], {}, {}, []
    for sid, text in sections.items():
        owner = "/root/fictional_section_" + sid.replace(".", "_")
        if case == "section_owner_reused" and sid == "R.2":
            owner = "/root/fictional_section_R_1"
        original = report(sid, source, text, owner, original=True)
        current = report(sid, source, text, owner)
        current["original_issue_resolutions"] = []
        if sid == "R.1":
            original["verdict"] = "revise"
            original["issues"] = [{"status": "unresolved", "necessity": "optional", "source_quote": "New navigation.",
                                    "details": "FICTIONAL optional finding."}]
            current["original_issue_resolutions"] = [{"json_pointer": "/issues/0", "resolution": "FICTIONAL same reviewer completed the check."}]
            if case == "original_claim_without_evidence":
                original["claims"] = [{"id": "c1", "kind": "software", "status": "verified", "statement": "FICTIONAL software claim.",
                                       "location": "FICTIONAL R.1", "scope": "FICTIONAL fixture only."}]
            if case == "original_sources_malformed":
                original["sources"] = [{"id": "s1", "kind": "repository_code"}]
            if case == "original_shell":
                original = {"reviewer_task": owner, "lesson_id": sid}
            if case == "original_after_current":
                original["created_at"] = "2026-10-20T02:06:00+00:00"
            if case == "missing_original_closure":
                current["original_issue_resolutions"] = []
            if case == "current_unresolved":
                current["verdict"] = "revise"
            if case == "current_creation_after_stage":
                current["created_at"] = "2026-10-20T02:40:00+00:00"
        oref = save(root, "reports/" + sid + "-original.json", original)
        cref = save(root, "reports/" + sid + "-current.json", current)
        originals[sid], current_reports[sid] = original, current
        entry = {"lesson_id": sid, "original_report": oref, "report": cref, "original_section_freeze": auxiliary,
                 "original_collection": {"wave": "fictional", "collection_role": "technical", "received_at": "2026-10-20T02:03:00+00:00"},
                 "current_collection": {"wave": "fictional", "collection_role": "technical", "received_at": "2026-10-20T02:05:00+00:00"}}
        if case == "callback_without_dispatch" and sid == "R.1":
            entry["current_collection"]["wave"] = "fictional-callback"
        entries.append(entry)
        if case != "section_owner_reused" or sid != "R.2":
            tasks.append({"task": owner, "role": "technical", "context": "fresh", "lesson_id": sid,
                          "wave": "fictional", "dispatch_window": {"not_before": "2026-10-20T02:00:00+00:00", "not_after": "2026-10-20T02:10:00+00:00"}})
    api = load("fictional_api_" + case, root / "docs/review-tools/incremental_revision.py")
    bundle = {"canonical_source_freeze": canonical, "section_freeze": auxiliary, "sections": entries}
    index = tool.section_issue_index(bundle, "/root", api)
    if case == "index_omits_original_issue":
        index["issues"] = []
    index_ref = save(root, "index/issues.json", index)
    bundle["issue_inventory"] = index_ref
    if case == "missing_section_R2":
        bundle["sections"] = bundle["sections"][:1]
    bundle_ref = save(root, "docs/technical-reviews/artifacts/fictional-bundle.json", bundle)
    owner = "/root/fictional_collector"
    original = report("course", source, body, owner, original=True)
    current = report("course", source, body, owner)
    if case != "collector_missing_binding":
        current["artifacts"] = [{"id": "bundle", "kind": "source_snapshot", **bundle_ref, "description": "FICTIONAL section bundle."}]
    oref = save(root, "reports/collector-original.json", original)
    cref = save(root, "reports/collector-current.json", current)
    proof = {"mode": "review", "wave": "fictional", "report": cref, "original_report": oref, "original_source_freeze": canonical,
             "original_collection": {"wave": "fictional", "collection_role": "technical", "received_at": "2026-10-20T02:03:00+00:00"},
             "current_collection": {"wave": "fictional", "collection_role": "technical", "received_at": "2026-10-20T02:05:00+00:00"}}
    tasks.append({"task": owner, "role": "technical", "context": "fresh", "lesson_id": "course", "wave": "fictional",
                  "dispatch_window": {"not_before": "2026-10-20T02:00:00+00:00", "not_after": "2026-10-20T02:10:00+00:00"}})
    if case == "section_reader_reuse":
        tasks[0]["role"] = "reader"
    assignments = save(root, "assignments.json", {"tasks": tasks})
    rows = []
    for entry in [*entries, proof]:
        for key, receipt_key in (("original_report", "original_collection"), ("report", "current_collection")):
            rows.append({**entry[key], "received_at": entry[receipt_key]["received_at"]})
    if case == "missing_collection_receipt":
        rows = rows[1:]
    if case == "duplicate_collection_receipt":
        rows.append(copy.deepcopy(rows[0]))
    if case == "receipt_time_mismatch":
        rows[0]["received_at"] = "2026-10-20T02:04:00+00:00"
    reader_seal = save(root, "seals/reader.json", {"role": "reader", "status": "completed", "started_at": "2026-10-20T01:00:00+00:00", "completed_at": "2026-10-20T01:20:00+00:00", "report_sha256s": []})
    tech_seal = save(root, "seals/technical.json", {"role": "technical", "status": "completed", "started_at": "2026-10-20T02:00:00+00:00", "completed_at": "2026-10-20T02:20:00+00:00", "report_sha256s": [row["sha256"] for row in rows], "collection_index": rows})
    decision = {"issue_id": "FICTIONAL-optional", "report": index_ref, "json_pointer": "/issues/0", "final_necessity": "optional", "action": "defer", "affected_pages": ["course"],
                **{name: "FICTIONAL reason." for name in ("source_support", "learning_benefit", "change_burden", "reason")},
                "section_verification": {"report": entries[0]["report"], "json_pointer": "/original_issue_resolutions/0"}}
    if case == "missing_author_reason":
        del decision["learning_benefit"]
    if case == "necessary_deferred":
        decision["final_necessity"] = "necessary"
    if case == "wrong_parent":
        decision["affected_pages"] = ["training"]
    if case == "wrong_section_verification":
        decision["section_verification"]["json_pointer"] = "/issues/0"
    if case == "fixed_without_collector":
        decision["action"] = "fixed"
    decisions = [] if case == "omit_optional_disposition" else [decision]
    if case == "duplicate_disposition":
        decisions.append(copy.deepcopy(decision))
    disposition_ref = save(root, "dispositions.json", {"at": "2026-10-20T02:06:00+00:00", "pending_count": 0, "items": decisions})
    manifest = {"author_tasks": ["/root"], "source_freeze": canonical, "numbered_section_requirements": {"comparison_source_freeze": prior, "page_ids": ["course"]},
                "numbered_section_reviews": bundle_ref, "assignments": assignments, "pages": [{"page_id": "course", "technical": proof}],
                "review_waves": [{"id": "fictional", "seals": {"reader": reader_seal, "technical": tech_seal}}],
                "issue_reports": [index_ref, oref, cref], "additional_issue_inventories": [{"report": index_ref, "json_pointer": "/issues"}],
                "dispositions": disposition_ref, "dispositions_collection": {"received_at": "2026-10-20T02:07:00+00:00", "wave": "fictional", "role": "technical"}}
    if case == "author_reuse":
        manifest["author_tasks"].append(tasks[0]["task"])
    if case == "missing_requirement":
        del manifest["numbered_section_requirements"]
    if case == "self_comparison":
        manifest["numbered_section_requirements"]["comparison_source_freeze"] = canonical
    if case == "omitted_page_scope":
        manifest["numbered_section_requirements"]["page_ids"] = []
    if case == "missing_sidecar":
        del manifest["numbered_section_reviews"]
    if case == "missing_issue_inventory_hook":
        manifest["additional_issue_inventories"] = []
    return tool, manifest


def main():
    cases = ["valid", "missing_requirement", "self_comparison", "omitted_page_scope", "missing_sidecar", "missing_section_R2", "section_owner_reused", "section_reader_reuse", "author_reuse", "original_shell", "original_claim_without_evidence", "original_sources_malformed", "original_after_current", "missing_original_closure", "current_unresolved", "current_creation_after_stage", "callback_without_dispatch", "missing_collection_receipt", "duplicate_collection_receipt", "receipt_time_mismatch", "collector_missing_binding", "index_omits_original_issue", "missing_author_reason", "necessary_deferred", "wrong_parent", "wrong_section_verification", "fixed_without_collector", "omit_optional_disposition", "duplicate_disposition", "missing_issue_inventory_hook"]
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="omni-independent-tools-") as folder:
            tool, manifest = fixture(Path(folder), case)
            try:
                observed = tool.check_numbered(manifest)
                results.append({"case": case, "accepted": True, "result": observed})
            except (ValueError, KeyError, TypeError, OSError) as error:
                results.append({"case": case, "accepted": False, "error": str(error)})
    real_tool = load("real_omni_scope_audit", ROOT / "scripts/omni_revision/review_setup.py")
    real = json.loads((ROOT / MANIFEST).read_bytes())
    accepted = real_tool.read_ref(real_tool.ACCEPTED_BASELINE)
    real_cases = []
    for name, manifest, path in [("exact_authenticated_baseline", accepted, real_tool.ACCEPTED_BASELINE["path"]), ("same_bytes_different_path_no_exemption", accepted, MANIFEST), ("candidate_missing_numbered_sidecar", real, MANIFEST)]:
        try:
            value = real_tool.check_numbered(manifest, intake=True, manifest_path=path)
            real_cases.append({"case": name, "accepted": True, "result": value})
        except (ValueError, KeyError, TypeError, OSError) as error:
            real_cases.append({"case": name, "accepted": False, "error": str(error)})
    # Show exactly what a normal source_snapshot hash binds: the bundle, not files named inside it.
    with tempfile.TemporaryDirectory(prefix="omni-independent-w1-") as folder:
        root = Path(folder)
        source = "course/first-steps.md"
        body = "# FICTIONAL navigation\n"
        save(root, source, body.encode())
        w1_ref = save(root, "reviews/W1-original.json", {"lesson_id": "W.1", "created_at": "2026-10-20T02:02:00+00:00"})
        bundle_ref = save(root, "docs/technical-reviews/artifacts/W1-bundle.json", {"original_report": w1_ref})
        collector = report("first-steps", source, body, "/root/fictional_w1_collector")
        collector["artifacts"] = [{"id": "bundle", "kind": "source_snapshot", **bundle_ref, "description": "FICTIONAL W1 evidence references."}]
        validator = load("real_validator_w1_chain", ROOT / "scripts/check_technical_reviews.py")
        before = validator._validate(root, root / source, "first-steps", body, collector, {collector["reviewer_task"]: ["first-steps"]}, set())
        save(root, w1_ref["path"], {"lesson_id": "W.1", "tampered": True})
        after = validator._validate(root, root / source, "first-steps", body, collector, {collector["reviewer_task"]: ["first-steps"]}, set())
        chain = {"before_errors": before, "after_referenced_W1_changed_errors": after, "bundle_bytes_unchanged": ref(root, bundle_ref["path"]) == bundle_ref, "referenced_W1_hash_changed": ref(root, w1_ref["path"]) != w1_ref,
                 "scope": "A bare collector bundle artifact is not a recursive W1 verifier; a separate invoked check is necessary."}
    output = {"created_at": datetime.now(UTC).isoformat(), "reviewer_task": TASK, "command": "uv run --extra cpu python " + str(Path(__file__).relative_to(ROOT)),
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; metadata-only"},
              "fixture_note": "All test reports/seals are fictional TemporaryDirectory data. In-memory ROOT/certificate substitution is restricted to fixtures. No real reports/seals were generated or edited.",
              "fixture_mutations": results, "real_scope_checks": real_cases, "W1_artifact_chain_mutation": chain}
    target = OUT / "mutation-results-v1.json"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(target.relative_to(ROOT)), "cases": len(results), "accepted_cases": [row["case"] for row in results if row["accepted"]], "real_scope_checks": real_cases, "W1_artifact_chain_mutation": chain}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
