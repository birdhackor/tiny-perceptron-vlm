"""Read-only bounded appendix metadata continuation after actual current2 intake.

Reuses the same owner's four prior historical judgments and experiments. No new
blind reading, primary fetch, scalar/NN experiment, final gate, collection or seal.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
TASK = "/root/omni_p5_technical_glossary_whole"
REVIEW_DIR = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-glossary/"
PREVIOUS = {"path": REVIEW_DIR + "historical-group-6-appendix-v1.json", "sha256": "0e25d5e3f1afb133f4c870e703f64bf5376492e40a198bdbe1c75c2559630d70"}
CURRENT2 = {"path": REVIEW_DIR + "technical-current-v2.json", "sha256": "c69d7c21076334e9cd67e0210f4d6269944dc1a3c25afcb745bb3b71206b89c8"}
INTAKE19 = {"path": "docs/course-revision-20261029-omni/reviews/P5-actual/technical-original-collection-intake-v19.json", "sha256": "ed3a6774c6493e95224c66b89caae9d5e787d80ffb7f89f4274d5938a73b086a"}


def reference(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def read_ref(value):
    assert reference(value["path"])["sha256"] == value["sha256"], value
    return json.loads((ROOT / value["path"]).read_bytes())


def pointer(doc, path):
    for token in path.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        doc = doc[int(token)] if isinstance(doc, list) else doc[token]
    return doc


def main():
    started = datetime.now(timezone.utc).isoformat()
    previous = read_ref(PREVIOUS)
    current2 = read_ref(CURRENT2)
    original = read_ref(previous["original_review"])
    current1 = read_ref(previous["full_review"])
    intake = read_ref(INTAKE19)
    hits = [(i, r) for i, r in enumerate(intake["collection_index"])
            if r.get("path") == CURRENT2["path"] and r.get("sha256") == CURRENT2["sha256"]]
    assert len(hits) == 1
    index, receipt = hits[0]
    assert index == 82 and receipt["reviewer_task"] == TASK
    assert receipt["collection_role"] == "technical" and receipt["wave"] == "omni-mainline-03-actual"
    at = datetime.fromisoformat
    assert at(original["created_at"]) < at(current1["created_at"]) <= at(previous["full_review_actual_receipt"]["record"]["received_at"]) < at(previous["created_at"])
    assert at(previous["created_at"]) < at(current2["created_at"]) <= at(receipt["received_at"]) < at(started)
    assert receipt["created_at"] == current2["created_at"]
    assert all(r["reviewer_task"] == TASK for r in [previous, current1, current2, original])
    assert not current2.get("historical_issue_rechecks") and not original.get("historical_issue_rechecks")
    assert current2["claims"][:184] == original["claims"]
    assert current2["sources"][:103] == original["sources"]
    assert current2["artifacts"][:9] == original["artifacts"]
    freeze = read_ref(current2["canonical_source_freeze"])
    before_freeze = read_ref(previous["canonical_source_freeze"])
    page = next(p for p in freeze["pages"] if p["page_id"] == "glossary")
    before_page = next(p for p in before_freeze["pages"] if p["page_id"] == "glossary")
    assert page == before_page
    raw = (ROOT / page["snapshot"]).read_bytes()
    assert raw == (ROOT / page["source"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == previous["source_sha256"] == current2["source_sha256"] == original["source_sha256"]
    assert page["figures_sha256"] == previous["figure_sha256"] == current2["figure_sha256"] == {}
    text = raw.decode()
    prior_artifact_checks = []
    for item in previous["artifacts"]:
        actual = reference(item["path"])
        assert actual["sha256"] == item["sha256"]
        prior_artifact_checks.append({"artifact_id": item["id"], **actual})
    repo_source_checks = []
    for source in previous["sources"]:
        if source["kind"] == "repository_code":
            actual = reference(source["path"])
            assert actual["sha256"] == source["sha256"]
            repo_source_checks.append({"source_id": source["id"], **actual})
    request = read_ref(previous["historical_group_request"])
    group = pointer(request, previous["historical_group_request"]["json_pointer"])
    assert previous["historical_group_index"] == 6 and len(group["items"]) == 4
    assert group["old_owner"] == previous["old_owner"] and group["unavailability"] == previous["unavailability"]
    unavailable = pointer(read_ref(group["unavailability"]["report"]), group["unavailability"]["json_pointer"])
    assert unavailable["unavailable"] is True and unavailable["reviewer_task"] == group["old_owner"]
    rows = []
    for old_row, item in zip(previous["historical_issue_rechecks"], group["items"], strict=True):
        for field in ["disposition_ref", "original_report", "json_pointer", "historical_fix_verification", "issue_id", "necessity"]:
            assert old_row[field] == item[field]
        assert old_row["necessity"] == old_row["original_disposition_necessity"] == "necessary"
        assert old_row["original_disposition_action"] == "fixed"
        assert old_row["status"] == "still_fixed" and old_row["verdict"] == "pass"
        assert old_row["affected_pages"] == ["glossary"]
        assert old_row["current_source_sha256"] == page["source_sha256"] and old_row["current_figure_sha256"] == {}
        assert old_row["current_source_quote"] in text
        assert old_row["pages"]["glossary"]["source_quote"] in text
        assert old_row["pages"]["glossary"]["source_sha256"] == page["source_sha256"]
        disposition = pointer(read_ref(item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
        issue = pointer(read_ref(item["original_report"]), item["json_pointer"])
        assert disposition["issue_id"] == old_row["issue_id"] and disposition["action"] == "fixed"
        assert disposition["final_necessity"] == "necessary" and issue["status"] == old_row["original_issue_status"]
        assert item["historical_fix_verification"] in disposition["verification"]
        rows.append({"issue_id": old_row["issue_id"], "disposition_ref": old_row["disposition_ref"],
                     "original_report": old_row["original_report"], "json_pointer": old_row["json_pointer"],
                     "historical_fix_verification": old_row["historical_fix_verification"],
                     "necessity": "necessary", "original_disposition_action": "fixed",
                     "original_issue_status": old_row["original_issue_status"], "status": "still_fixed", "verdict": "pass",
                     "current_quote_present": True, "original_duplicate_context": old_row["original_duplicate_context"]})
    assert len({r["issue_id"] for r in rows}) == 4
    prior_probe = read_ref(next(a for a in previous["artifacts"] if a["id"] == "h-execution"))
    assert prior_probe["status"] == "passed"
    current_check = read_ref(next(a for a in current2["artifacts"] if a["id"] == "a-current-sidecar-crosscheck"))
    assert current_check["status"] == "passed"
    output = {
        "started_at": started, "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "same-owner-historical-group6-bounded-metadata-continuation-not-new-reading-or-experiment",
        "reviewer_task": TASK, "status": "passed",
        "command": "uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-glossary/historical-group-6-metadata-rebind-v2.py",
        "environment": {"python": platform.python_version(), "device": "CPU; JSON/source/hash checks only"},
        "previous_historical_appendix": reference(PREVIOUS["path"]),
        "full_review": reference(CURRENT2["path"]), "original_review": reference(previous["original_review"]["path"]),
        "previous_full_review": reference(previous["full_review"]["path"]),
        "actual_full_review_intake": {"report": reference(INTAKE19["path"]), "json_pointer": f"/collection_index/{index}", "record": receipt},
        "canonical_source_freeze": current2["canonical_source_freeze"],
        "current_source_sha256": page["source_sha256"], "current_figure_sha256": {},
        "glossary_body_and_page_record_identical_to_appendix1": True,
        "current_quotes": {"unicode": previous["historical_issue_rechecks"][0]["current_source_quote"],
                           "ppo": previous["historical_issue_rechecks"][1]["current_source_quote"]},
        "historical_instances": rows, "duplicate_instances_preserved": previous["duplicate_instances_preserved"],
        "unavailability": previous["unavailability"], "unavailability_record_preserved": True,
        "previous_artifact_checks": prior_artifact_checks, "repository_source_checks": repo_source_checks,
        "prior_scalar_primary_evidence_reused_without_new_execution": True,
        "scope": "Four existing still-fixed judgments and exact evidence reused under same body/code bytes. Only current2/freeze11/actual v19 receipt metadata is rebound after its real receipt. No additional history group or semantic fresh reading.",
        "pending": ["Root actual appendix2 collection", "Current technical stage completion and any final global numbered/history gate"],
        "not_run": ["new scalar/NN experiment", "new primary fetch", "fresh blind whole-page reading", "training", "formal final reopening", "T/tau or policy tuning", "Modal", "workflow dispatch", "publication", "final gate", "collection or seal creation"]
    }
    path = PROOF / "historical-group-6-metadata-rebind-v2.json"
    assert not path.exists()
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(reference(str(path.relative_to(ROOT)))))


if __name__ == "__main__":
    main()
