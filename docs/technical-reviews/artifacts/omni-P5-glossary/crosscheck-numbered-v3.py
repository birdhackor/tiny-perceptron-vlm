"""Same glossary owner's bounded source12/sidecar callback for continuity03.

Reads inputs and writes only its own portable evidence/source snapshot. No model,
kernel, scalar experiment, final gate, shared-input mutation, collection or seal.
"""
from collections import defaultdict
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
TASK = "/root/omni_p5_technical_glossary_whole"
BASE = "docs/course-revision-20261029-omni/reviews/P5-actual/"
ORIGINAL = {"path": BASE + "technical-glossary/technical-original-v1.json", "sha256": "1e4367497e8dec241dfc0cf142cfe886a2ba2d1dac376ed623a7e6dcc7aece1a"}
CURRENT2 = {"path": BASE + "technical-glossary/technical-current-v2.json", "sha256": "c69d7c21076334e9cd67e0210f4d6269944dc1a3c25afcb745bb3b71206b89c8"}
SIDE3 = {"path": "docs/technical-reviews/artifacts/omni-P5-numbered-v3/numbered-section-reviews-v1.json", "sha256": "5225b24a74b01fe824481d792c61d299f6400c3141c1bfd185ed196b432ec158"}
REPAIRS = {"path": BASE + "coherence-three-author-repairs-source12-v1.json", "sha256": "76e29aef2e528bf06ae20eacfdf48f28327fa9432e6477d50996be44301a3ea7"}
DISPATCH = {"path": BASE + "source12-callback-dispatch-batch2-v1.json", "sha256": "c1e143904c55482223b82ee1d6787d5b9ad65c7489297e590eef7657b020722c"}
TECHNICAL = {"path": BASE + "technical-phase-proof-v1.json", "sha256": "b756abd42038c1baf010b0ee64de67cd016e903f0fc03371322ec5c64b35be64"}
CONTINUITY = {"path": BASE + "coherence-stage-open-v1.json", "sha256": "aa8ada440c4ab5000e5dcc3e01dd57216c2ec6e30f6e89262292dd4c9b877882"}


def reference(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_ref(value):
    assert reference(value["path"])["sha256"] == value["sha256"], value
    return json.loads((ROOT / value["path"]).read_bytes())


def section(raw, lesson):
    match = re.search(rb"^## " + re.escape(lesson.encode()) + rb" .*$", raw, re.M)
    assert match is not None
    following = re.search(rb"^## ", raw[match.end():], re.M)
    end = match.end() + following.start() if following else len(raw)
    return raw[match.start():end]


def main():
    started = datetime.now(timezone.utc).isoformat()
    own, previous, side, repairs, dispatch, technical, continuity = [read_ref(x) for x in [ORIGINAL, CURRENT2, SIDE3, REPAIRS, DISPATCH, TECHNICAL, CONTINUITY]]
    assert technical["role"] == "technical" and technical["status"] == "completed"
    assert continuity["role"] == "continuity" and continuity["status"].startswith("open")
    assert continuity["preceding_technical_seal"] == TECHNICAL
    assert dispatch["collection_role"] == "continuity" and TASK in dispatch["tasks"]
    at = datetime.fromisoformat
    assert at(technical["completed_at"]) < at(continuity["started_at"]) <= at(dispatch["dispatch_window"]["not_before"]) <= at(dispatch["dispatch_window"]["not_after"]) <= at(started)
    current2_receipts = [(i, r) for i, r in enumerate(technical["collection_index"]) if r.get("path") == CURRENT2["path"] and r.get("sha256") == CURRENT2["sha256"]]
    assert len(current2_receipts) == 1
    side2 = read_ref(previous["numbered_section_reviews"])
    frozen, before_freeze = read_ref(side["canonical_source_freeze"]), read_ref(side2["canonical_source_freeze"])
    numbered, before_numbered = read_ref(side["section_freeze"]), read_ref(side2["section_freeze"])
    assert side["canonical_source_freeze"]["sha256"] == "1d95d71c09a20a07334cd30707d61bcfe7a6334f8bc12241b04ee2dc66aed4b1"
    assert side["section_freeze"]["sha256"] == "f7ba7cc4df7edfb693f2024f819e8886a0fabba56bdfb0b6b94f18c364551423"
    assert repairs["source_freeze"] == side["canonical_source_freeze"]
    pages, before_pages = [{p["page_id"]: p for p in f["pages"]} for f in [frozen, before_freeze]]
    changed_pages = sorted(k for k in pages if pages[k] != before_pages[k])
    assert changed_pages == ["12.17", "19.23", "9.17"]
    page = pages["glossary"]
    assert page == before_pages["glossary"]
    body = (ROOT / page["snapshot"]).read_bytes()
    assert body == (ROOT / page["source"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == own["source_sha256"] == previous["source_sha256"]
    assert page["figures_sha256"] == own["figure_sha256"] == previous["figure_sha256"] == {}
    # Recover the exact original chapter bytes by reversing the independently
    # inspected one-line repair, then authenticate against the immutable report.
    ch09 = (ROOT / "course/chapters/09.md").read_bytes()
    new_line = b"chosen = min(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))"
    old_line = b"chosen = min(range(len(ids)), key=lambda i: (-float(p[i]), ids[i]))"
    assert ch09.count(new_line) == 1 and old_line not in ch09
    old_ch09 = ch09.replace(new_line, old_line)
    ch09_source = next(s for s in own["sources"] if s["id"] == "r-ch09")
    assert hashlib.sha256(old_ch09).hexdigest() == ch09_source["sha256"]
    assert section(ch09, "9.15") == section(old_ch09, "9.15")
    assert section(ch09, "9.16") == section(old_ch09, "9.16")
    change = next(x for x in repairs["changes"] if x["page_id"] == "9.17")
    assert (ROOT / change["after"]["path"]).read_bytes() == section(ch09, "9.17")
    assert (ROOT / change["before"]["path"]).read_bytes() == section(old_ch09, "9.17")
    assert hashlib.sha256(section(ch09, "9.17")).hexdigest() == change["after"]["sha256"]
    assert hashlib.sha256(section(old_ch09, "9.17")).hexdigest() == change["before"]["sha256"]
    direct_claims = [c for c in own["claims"] if any(e["source_id"] == "r-ch09" for e in c["evidence"])]
    assert [c["id"] for c in direct_claims] == ["g2-error-cost-conventions"]
    cost_quote = next(line for line in section(ch09, "9.16").decode().splitlines() if line.startswith("用人工訂的成本說明"))
    assert "開錯頁的成本是5單位" in cost_quote and "這個門檻只屬於這張簡化的成本表" in cost_quote
    glossary_cost_quote = next(line for line in body.decode().splitlines() if line.startswith("| 誤差、成本 |"))
    glossary_pointer_quote = next(line for line in body.decode().splitlines() if line.startswith("| pointer head"))
    old_path = PROOF / "original-sources/chapter-09-for-original-v1.md"
    assert not old_path.exists()
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_bytes(old_ch09)
    original_snapshot = {"source": "course/chapters/09.md", "snapshot": {k: v for k, v in reference(str(old_path.relative_to(ROOT))).items() if k != "bytes"}}
    dependencies, drift = [], []
    for source in own["sources"]:
        if source["kind"] != "repository_code":
            continue
        actual = reference(source["path"])
        changed = actual["sha256"] != source["sha256"]
        dependencies.append({"source_id": source["id"], "original_sha256": source["sha256"], "changed": changed, **actual})
        if changed:
            drift.append(source["id"])
    assert drift == ["r-ch09"]
    prior_artifacts = []
    for item in previous["artifacts"]:
        actual = reference(item["path"])
        assert actual["sha256"] == item["sha256"]
        prior_artifacts.append({"artifact_id": item["id"], **actual})
    old_map = {e["lesson_id"]: e for e in side2["sections"]}
    changed_entries = [e["lesson_id"] for e in side["sections"] if e != old_map[e["lesson_id"]]]
    assert len(side["sections"]) == 26 and changed_entries == ["W.7"]
    current_slices, old_slices = [{p["page_id"]: p for p in f["pages"]} for f in [numbered, before_numbered]]
    for lesson, now in current_slices.items():
        before = old_slices[lesson]
        assert {k: v for k, v in now.items() if k != "snapshot"} == {k: v for k, v in before.items() if k != "snapshot"}
        assert (ROOT / now["snapshot"]).read_bytes() == (ROOT / before["snapshot"]).read_bytes()
    inventory, old_inventory = read_ref(side["issue_inventory"]), read_ref(side2["issue_inventory"])
    assert inventory["issues"] == old_inventory["issues"]
    assert inventory["canonical_source_freeze"] == side["canonical_source_freeze"] and inventory["section_freeze"] == side["section_freeze"]
    owners = defaultdict(list)
    refs = []
    for entry in side["sections"]:
        for field in ["report", "original_report", "original_section_freeze"]:
            report = read_ref(entry[field])
            refs.append({"lesson_id": entry["lesson_id"], "field": field, **reference(entry[field]["path"])})
            if field == "report":
                owners[report["reviewer_task"]].append(entry["lesson_id"])
    assert all(len(v) == 1 for v in owners.values()) and TASK not in owners
    assert "technical-glossary/" not in json.dumps(side)
    w7 = next(e for e in side["sections"] if e["lesson_id"] == "W.7")
    w7_report, w7_original = read_ref(w7["report"]), read_ref(w7["original_report"])
    assert w7["original_report"] == old_map["W.7"]["original_report"]
    assert w7_report["reviewer_task"] == w7_original["reviewer_task"]
    w7_intake_path = BASE + "coherence-current-collection-intake-v3.json"
    w7_intake = json.loads((ROOT / w7_intake_path).read_bytes())
    w7_receipts = [(i, r) for i, r in enumerate(w7_intake["collection_index"]) if r.get("path") == w7["report"]["path"] and r.get("sha256") == w7["report"]["sha256"]]
    assert len(w7_receipts) == 1
    wi, wr = w7_receipts[0]
    assert wr["received_at"] == w7["current_collection"]["received_at"] and wr["collection_role"] == "continuity"
    assert at(continuity["started_at"]) <= at(w7["current_dispatch_window"]["not_before"]) <= at(w7["current_dispatch_window"]["not_after"]) <= at(w7_report["created_at"]) <= at(wr["received_at"])
    spec = importlib.util.spec_from_file_location("existing_validator", ROOT / "scripts/check_technical_reviews.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    g_results = []
    for entry in side["sections"]:
        lesson = entry["lesson_id"]
        if not lesson.startswith("G."):
            continue
        assert entry == old_map[lesson] and entry["report"] == entry["original_report"]
        report = read_ref(entry["report"])
        now = current_slices[lesson]
        raw = (ROOT / now["snapshot"]).read_bytes()
        original_slice_freeze = read_ref(entry["original_section_freeze"])
        before = next(p for p in original_slice_freeze["pages"] if p["page_id"] == lesson)
        before_raw = (ROOT / before["snapshot"]).read_bytes()
        assert raw == before_raw == section(body, lesson)
        current_errors = validator._validate(ROOT, ROOT / now["source"], lesson, raw.decode(), report, dict(owners), set())
        original_errors = validator._validate(ROOT, ROOT / before["source"], lesson, before_raw.decode(), report, dict(owners), set())
        assert not current_errors and not original_errors
        g_results.append({"lesson_id": lesson, "report": reference(entry["report"]["path"]),
                          "source_sha256": now["source_sha256"], "section_claim_count": len(report["claims"]),
                          "original_and_current_errors": [original_errors, current_errors], "entry_and_source_unchanged": True})
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "started_at": started, "reviewer_task": TASK,
        "kind": "same-original-owner-bounded-source12-dependency-and-sidecar-callback-for-continuity03", "status": "passed",
        "command": "uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-glossary/crosscheck-numbered-v3.py",
        "environment": {"python": platform.python_version(), "device": "CPU; JSON/hash/source inspection only; no kernel/model execution"},
        "immutable_original": reference(ORIGINAL["path"]), "immutable_previous_current": reference(CURRENT2["path"]),
        "author_repair_receipt": reference(REPAIRS["path"]), "canonical_freeze": side["canonical_source_freeze"], "section_freeze": side["section_freeze"],
        "shared_sidecar": reference(SIDE3["path"]), "shared_issue_inventory": reference(side["issue_inventory"]["path"]),
        "whole_glossary": {"bytes": len(body), "source_sha256": page["source_sha256"], "figure_sha256": {}, "page_record_body_and_four_G_slices_unchanged": True},
        "source12_dependency_review": {
            "changed_original_repository_sources": drift, "source": "course/chapters/09.md", "old_sha256": ch09_source["sha256"], "current": reference("course/chapters/09.md"),
            "full_chapter_only_diff": "".join(difflib.unified_diff(old_ch09.decode().splitlines(keepends=True), ch09.decode().splitlines(keepends=True), fromfile="original/chapter09", tofile="source12/chapter09", n=2)),
            "direct_original_claim_ids": [c["id"] for c in direct_claims], "original_claim_statement": direct_claims[0]["statement"],
            "current_cost_source_quote": cost_quote, "current_glossary_cost_quote": glossary_cost_quote,
            "cost_claim_reason": "The registered r-ch09 support is the artificial cost5 convention in9.16. Both9.15 and9.16 remain byte-identical, so the statement that this manual decision cost differs from training loss still holds.9.17 changes only how an artificial candidate winner is selected, not a cost unit, loss definition, cost table or empirical measurement.",
            "linked_pointer_original_claim_ids": ["g4-pointer", "g4-pointer-permutation"], "current_glossary_pointer_quote": glossary_pointer_quote,
            "pointer_support_reason": "Personally inspected the old/new9.17 complete linked mechanism and current pointer glossary row. Shared projections, candidate visibility, symmetric none, causal contrast and permutation qualification are unchanged. Selecting raw scores then stable ID preserves the explicitly intended highest-score/stable-tie rule directly; finite-precision softmax probabilities need not preserve distinct-score resolution. It adds no learned quality or general permutation guarantee. Original pointer claims and their pinned Kev/Transformer support remain valid.",
            "verification_method": "Source/diff and exact unchanged-section reading; no NN, kernel, numerical fixture or model-quality experiment.",
            "original_source_snapshot": original_snapshot},
        "repository_dependency_checks": dependencies, "previous_artifact_checks": prior_artifacts,
        "sidecar_delta": {"changed_entry_ids": changed_entries, "unchanged_entry_count": 25, "all_26_section_bodies_unchanged": True,
                          "all_78_report_original_freeze_references_hashed": len(refs) == 78, "other22_scope": "Reference identity only; no semantic review", "collector_hash_cycle": False},
        "w7_current_reference_receipt_check": {"entry": w7, "intake": reference(w7_intake_path), "json_pointer": f"/collection_index/{wi}", "receipt": wr, "same_original_owner": True},
        "sections": g_results,
        "current_phase": {"wave": dispatch["wave"], "collection_role": "continuity", "continuity_open": CONTINUITY,
                          "closed_technical_seal_preserved": TECHNICAL, "callback_dispatch": DISPATCH, "dispatch_window": dispatch["dispatch_window"],
                          "previous_current_technical_receipt_pointer": f"/collection_index/{current2_receipts[0][0]}",
                          "current3_actual_collection_pending": True, "historical_appendix3_pending_actual_current3_receipt": True,
                          "final_numbered_history_gate_run": False, "collection_or_seal_created": False},
        "limits": ["Same owner bounded dependency reading and metadata callback; no fresh/full blind whole-page reading.",
                   "Only the necessary chapter09 delta receives semantic review;12.17/19.23 author diffs and other sidecar sections are not independently reviewed under this task.",
                   "Original184 glossary statement judgments and prior arithmetic/primary/semantic evidence retained; current r-ch09 evidence is substantively rechecked before rebinding.",
                   "No NN/kernel, scalar experiment, training, formal final reopening, T/tau or policy tuning, Modal, workflow or publication.",
                   "No old report, book source, shared bundle, checker or sealed technical03 record changed; group6 appendix3 waits for actual root current3 receipt."]
    }
    path = PROOF / "crosscheck-numbered-v3.json"
    assert not path.exists()
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(reference(str(path.relative_to(ROOT)))))


if __name__ == "__main__":
    main()
