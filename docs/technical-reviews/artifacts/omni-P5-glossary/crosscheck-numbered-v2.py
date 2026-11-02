"""Existing glossary owner's bounded metadata callback, not another fresh reading.

Read-only input checks. Keeps old reports, shared sidecars and checker unchanged;
does not run a model, final check-numbered, history gate, collection or seal.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
TASK = "/root/omni_p5_technical_glossary_whole"
ORIGINAL = {
    "path": "docs/course-revision-20261029-omni/reviews/P5-actual/technical-glossary/technical-original-v1.json",
    "sha256": "1e4367497e8dec241dfc0cf142cfe886a2ba2d1dac376ed623a7e6dcc7aece1a",
}
CURRENT1 = {
    "path": "docs/course-revision-20261029-omni/reviews/P5-actual/technical-glossary/technical-current-v1.json",
    "sha256": "8974f763d5fab6c4ad6c67beff96039cf941c688a88d38b801e3d16a47ab253a",
}
SIDE2 = {
    "path": "docs/technical-reviews/artifacts/omni-P5-numbered-v2/numbered-section-reviews-v1.json",
    "sha256": "a6b2189776362c7bc883c0ea26d01c2223b191b5989799dfa411fc48ba0dd87f",
}
FREEZE11_SHA = "d46a11ab7e27275f7e6908b8df8e40ed451a21ecc4fd6fedac02c4e01d0df7df"
SECTION6_SHA = "1ce6261143dc8321cf9ec7bc9e298c678aab07eed5daaaf66cb8ca04dd16a68a"
INTAKE15 = {
    "path": "docs/course-revision-20261029-omni/reviews/P5-actual/technical-original-collection-intake-v15.json",
    "sha256": "535b46de6b4da1c137163098a617db3330b0457c86cbaf8c35128c4673b15c80",
}


def reference(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_ref(value):
    actual = reference(value["path"])
    assert actual["sha256"] == value["sha256"], (value, actual)
    return json.loads((ROOT / value["path"]).read_bytes())


def main():
    started = datetime.now(timezone.utc).isoformat()
    own = read_ref(ORIGINAL)
    current1 = read_ref(CURRENT1)
    assert current1["claims"][:184] == own["claims"]
    assert current1["sources"][:103] == own["sources"]
    assert current1["artifacts"][:9] == own["artifacts"]
    assert "historical_issue_rechecks" not in own
    assert "historical_issue_rechecks" not in current1
    prior_check_ref = next(a for a in current1["artifacts"] if a["id"] == "a-current-sidecar-crosscheck")
    prior_check = read_ref(prior_check_ref)
    assert prior_check["status"] == "passed"
    side1 = read_ref(current1["numbered_section_reviews"])
    side2 = read_ref(SIDE2)
    assert side2["canonical_source_freeze"]["sha256"] == FREEZE11_SHA
    assert side2["section_freeze"]["sha256"] == SECTION6_SHA
    old_freeze = read_ref(side1["canonical_source_freeze"])
    freeze = read_ref(side2["canonical_source_freeze"])
    old_sections = read_ref(side1["section_freeze"])
    sections = read_ref(side2["section_freeze"])
    inv1 = read_ref(side1["issue_inventory"])
    inv2 = read_ref(side2["issue_inventory"])
    assert sections["canonical_source_freeze"] == side2["canonical_source_freeze"]
    assert inv2["canonical_source_freeze"] == side2["canonical_source_freeze"]
    assert inv2["section_freeze"] == side2["section_freeze"]
    changed_side_keys = sorted(k for k in side1.keys() | side2.keys() if side1.get(k) != side2.get(k))
    assert changed_side_keys == ["canonical_source_freeze", "created_at", "issue_inventory", "section_freeze"]
    assert side1["sections"] == side2["sections"] and len(side2["sections"]) == 26
    changed_inv_keys = sorted(k for k in inv1.keys() | inv2.keys() if inv1.get(k) != inv2.get(k))
    assert changed_inv_keys == ["canonical_source_freeze", "section_freeze"]
    old_pages = {p["page_id"]: p for p in old_freeze["pages"]}
    pages = {p["page_id"]: p for p in freeze["pages"]}
    changed_pages = sorted(k for k in old_pages.keys() | pages.keys() if old_pages.get(k) != pages.get(k))
    assert changed_pages == ["curriculum"]
    page = pages["glossary"]
    assert page == old_pages["glossary"]
    body = (ROOT / page["snapshot"]).read_bytes()
    assert body == (ROOT / "course/glossary.md").read_bytes()
    assert hashlib.sha256(body).hexdigest() == page["source_sha256"] == own["source_sha256"] == current1["source_sha256"]
    assert page["figures_sha256"] == own["figure_sha256"] == current1["figure_sha256"] == {}
    assert "notebook" not in page
    dependency_checks = []
    for source in own["sources"]:
        if source["kind"] != "repository_code":
            continue
        actual = reference(source["path"])
        assert actual["sha256"] == source["sha256"]
        dependency_checks.append({"source_id": source["id"], "original_sha256": source["sha256"], **actual})
    preserved_artifacts = []
    for item in current1["artifacts"]:
        actual = reference(item["path"])
        assert actual["sha256"] == item["sha256"]
        preserved_artifacts.append({"artifact_id": item["id"], **actual})
    changed_sources = [pages[p]["source"] for p in changed_pages]
    assert not set(changed_sources) & {s["path"] for s in dependency_checks}
    linked_local_paths = set()
    for dest in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body.decode()):
        url = urlsplit(dest)
        if url.scheme or url.netloc:
            continue
        resolved = (ROOT / "course" / unquote(url.path)).resolve() if url.path else ROOT / "course/glossary.md"
        if resolved.is_relative_to(ROOT):
            linked_local_paths.add(str(resolved.relative_to(ROOT)))
    assert not set(changed_sources) & linked_local_paths
    before_map = {p["page_id"]: p for p in old_sections["pages"]}
    now_map = {p["page_id"]: p for p in sections["pages"]}
    assert before_map.keys() == now_map.keys() and len(now_map) == 26
    for lesson, now in now_map.items():
        before = before_map[lesson]
        assert {k: v for k, v in now.items() if k != "snapshot"} == {k: v for k, v in before.items() if k != "snapshot"}
        raw = (ROOT / now["snapshot"]).read_bytes()
        assert raw == (ROOT / before["snapshot"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == now["source_sha256"]
    owners = defaultdict(list)
    reference_checks = []
    for entry in side2["sections"]:
        for field in ["report", "original_report", "original_section_freeze"]:
            value = read_ref(entry[field])
            reference_checks.append({"lesson_id": entry["lesson_id"], "field": field, **reference(entry[field]["path"])})
            if field == "report":
                owners[value["reviewer_task"]].append(entry["lesson_id"])
    assert all(len(v) == 1 for v in owners.values()) and TASK not in owners
    assert "technical-glossary/" not in json.dumps(side2)
    neutral = read_ref(current1["numbered_neutral_receipt_provenance"])
    assert neutral["current_wave"]["completed_technical_seal"] is None
    opened = read_ref(neutral["current_wave"]["technical_open"])
    assert opened["role"] == "technical" and opened["status"].startswith("open")
    assignments = read_ref(neutral["prior_assignment_source"])
    reader_tasks = {r["task"] for r in assignments.get("tasks", []) if r.get("role") == "reader"}
    spec = importlib.util.spec_from_file_location("existing_validator", ROOT / "scripts/check_technical_reviews.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    old_checks = {r["lesson_id"]: r for r in prior_check["sections"]}
    g_checks = []
    for entry in side2["sections"]:
        lesson = entry["lesson_id"]
        if not lesson.startswith("G."):
            continue
        report = read_ref(entry["report"])
        assert entry["report"] == entry["original_report"]
        now = now_map[lesson]
        raw = (ROOT / now["snapshot"]).read_bytes()
        original_freeze = read_ref(entry["original_section_freeze"])
        before = next(p for p in original_freeze["pages"] if p["page_id"] == lesson)
        before_raw = (ROOT / before["snapshot"]).read_bytes()
        assert raw == before_raw
        match = re.search(rb"^## " + re.escape(lesson.encode()) + rb" .*$", body, re.M)
        next_head = re.search(rb"^## ", body[match.end():], re.M)
        end = match.end() + next_head.start() if next_head else len(body)
        assert raw == body[match.start():end]
        assert now["source_sha256"] == report["source_sha256"] == old_checks[lesson]["source_sha256"]
        assert reference(entry["report"]["path"]) == old_checks[lesson]["report"]
        assert entry["current_collection"] == next(e for e in side1["sections"] if e["lesson_id"] == lesson)["current_collection"]
        current_errors = validator._validate(ROOT, ROOT / now["source"], lesson, raw.decode(), report, dict(owners), reader_tasks)
        original_errors = validator._validate(ROOT, ROOT / before["source"], lesson, before_raw.decode(), report, dict(owners), reader_tasks)
        assert not current_errors and not original_errors
        assert not report["issues"] and not [i for i in inv2["issues"] if i.get("lesson_id") == lesson]
        g_checks.append({"lesson_id": lesson, "source_sha256": now["source_sha256"], "section_bytes": len(raw),
                         "report": reference(entry["report"]["path"]), "section_claim_count": len(report["claims"]),
                         "current_original_same_report": True, "prior_semantic_check_preserved": True,
                         "existing_validate_current_errors": current_errors, "existing_validate_original_errors": original_errors,
                         "original_collection": entry["original_collection"], "current_collection": entry["current_collection"]})
    assert len(g_checks) == 4 and sum(g["section_claim_count"] for g in g_checks) == 456
    intake = read_ref(INTAKE15)
    receipt = intake["collection_index"][69]
    assert receipt["path"] == CURRENT1["path"] and receipt["sha256"] == CURRENT1["sha256"]
    assert datetime.fromisoformat(receipt["received_at"]) < datetime.fromisoformat(started)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "started_at": started, "reviewer_task": TASK,
        "kind": "same-original-owner-bounded-metadata-callback-not-fresh-reading-or-final-gate", "status": "passed",
        "command": "uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-glossary/crosscheck-numbered-v2.py",
        "environment": {"python": platform.python_version(), "device": "CPU; JSON/source/hash validation only"},
        "immutable_original": reference(ORIGINAL["path"]), "immutable_previous_current": reference(CURRENT1["path"]),
        "previous_current_actual_receipt": {"report": INTAKE15, "json_pointer": "/collection_index/69", "receipt": receipt},
        "previous_semantic_crosscheck": reference(prior_check_ref["path"]),
        "canonical_freeze": side2["canonical_source_freeze"], "section_freeze": side2["section_freeze"],
        "shared_sidecar": reference(SIDE2["path"]), "shared_issue_inventory": reference(side2["issue_inventory"]["path"]),
        "metadata_delta": {"sidecar_changed_keys": changed_side_keys, "issue_inventory_changed_keys": changed_inv_keys,
                           "all_26_section_entries_identical": True, "all_26_section_sources_and_snapshots_identical": True,
                           "canonical_changed_page_ids": changed_pages, "changed_page_in_original_dependencies_or_direct_glossary_links": False},
        "glossary": {"bytes": len(body), "source_sha256": page["source_sha256"], "figure_sha256": {},
                     "canonical_page_record_unchanged": True, "source_and_all_original_repository_dependencies_unchanged": True,
                     "original_independent_claims_retained": 184, "original_sources_retained": 103, "original_artifacts_retained": 9,
                     "prior_semantic_coverage_reused": "136 definition rows; 194 index terms; 455 internal link occurrences; no semantic rereading claimed by this callback"},
        "repository_dependency_checks": dependency_checks, "preserved_previous_current_artifact_checks": preserved_artifacts,
        "bundle_reference_checks": {"scope": "Hash/identity only for other22 sections; no semantic review", "reference_count": len(reference_checks),
                                    "all_references_match": True, "collector_back_reference_or_cycle": False},
        "sections": g_checks,
        "current_phase": {"wave": neutral["current_wave"]["id"], "technical_status": opened["status"],
                          "completed_technical_seal": None, "current2_collection_pending": True,
                          "historical_appendix_v2_pending_actual_current2_receipt": True,
                          "final_check_numbered_or_history_gate_run": False, "collection_or_seal_created": False},
        "checker_inputs": [reference("scripts/check_technical_reviews.py"), reference("docs/course-review-20261019/tools/check_technical_report.py")],
        "limits": ["No new fresh task, whole-page blind reading, primary fetch or G-section semantic review is claimed.",
                   "The prior independent full review and later G-sidecar semantic crosscheck are reused only after byte equality checks.",
                   "No model inference, training, formal final reopening, T/tau or policy tuning, Modal, workflow dispatch or publication.",
                   "No curriculum semantic review; its author correction does not change this page or necessary dependencies.",
                   "The old historical appendix remains untouched; no replacement appendix is written before root actually receives current2.",
                   "No shared bundle copy/mutation, checker/source change, final collection, completed phase or seal."]
    }
    path = PROOF / "crosscheck-numbered-v2.json"
    assert not path.exists()
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": "passed",
                      "dependencies": len(dependency_checks), "retained_artifacts": len(preserved_artifacts), "G_sections": 4}))


if __name__ == "__main__":
    main()
