"""Bounded historical callback checks; does not construct or certify a stage seal."""

import hashlib
import importlib.util
import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OWNER = "/root/omni_technical_coordinator/technical_9_15"
PACKET = ROOT / "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
FREEZE = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
FULL = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_9_15/technical-current-v2.json"
BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


api = module("historical_915_incremental_api", ROOT / "docs/review-tools/incremental_revision.py")
packet_document = json.loads(PACKET.read_bytes())
assert sha(PACKET) == "714f16f7d5c20924e20f91a8b48bca7a87bd77e2602d1a1ae501d06d93f819e5"
group = packet_document["groups"][4]
assert group["old_owner"] == "/root/rv1019_technical_9-15"
assert len(group["items"]) == 3
full = api.document(ROOT, reference(FULL))
assert sha(FULL) == "73ac1874d0974881344dd598feecf9c1eae0998ed76d50782d9baaef56814986"
freeze = json.loads(FREEZE.read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "9.15")
body = api.artifact(ROOT, {"path": page["snapshot"], "sha256": page["source_sha256"]}).decode()
current_text = (ROOT / page["source"]).read_text()
start = re.search(r"^## 9\.15 ", current_text, re.M).start()
following = re.search(r"^## ", current_text[start + 1:], re.M)
current_text = current_text[start:start + 1 + following.start()] if following else current_text[start:]
assert current_text == body and full["source_sha256"] == page["source_sha256"]
for source in full["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"]


def inspect_originals():
    old_receipt_ref = {"path": "docs/technical-reviews/artifacts/review-20261019-9-15/recheck-current-01.json", "sha256": "c65202e5580d04e7eb43fb77c2885b18a40cf656212704f596dfcc3851c3ab7d"}
    old_receipt = api.document(ROOT, old_receipt_ref)
    official = old_receipt["official_source"]
    assert official["accessed_utc"].startswith("2026-10-20")
    assert "20 Oct 2026" in official["http_date"]
    assert official["body_sha256"] == "5c325ec0c0b69e78129406ca7c5d3aef113bce365abd34985b8674cd278c81be"
    own_receipt = api.document(ROOT, reference(BASE / "source-inspection-v1.json"))
    current_source = next(x for x in own_receipt["originals"] if x["cache_path"].endswith("typesafe-confidence.md"))
    primary = Path(current_source["cache_path"])
    assert sha(primary) == current_source["sha256"] == "4c72bbd74c7e68bfb7d87d25bd62b099d4ef5563cfac3f5e9e1fe48b8cdac1ee"
    assert official["excerpts"]["choice_equation"] in primary.read_text()
    headers = Path(own_receipt["historical_date_check"]["current_response_headers_path"])
    assert sha(headers) == own_receipt["historical_date_check"]["current_response_headers_sha256"]
    assert "date: Thu, 29 Oct 2026 14:43:18 GMT" in headers.read_text()
    quote = "本節於2026-10-29核對的官方文件，對 n 個 Choice 候選，使用"
    assert quote in body and "截至2026-10-16" not in body and "截至2026-10-20" not in body
    identities = []
    for item in group["items"]:
        disposition = api.node(api.document(ROOT, item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
        report = api.document(ROOT, item["original_report"])
        issue = api.node(report, item["json_pointer"])
        verification = api.document(ROOT, item["historical_fix_verification"]["report"])
        assert item["historical_fix_verification"] in disposition["verification"]
        assert item["necessity"] == disposition["final_necessity"] == "optional"
        assert issue["necessary"] is False
        assert api.owner(report) == api.owner(verification) == group["old_owner"]
        assert verification["verdict"] == "pass"
        assert any(x["id"] == "a_current" and x["path"] == old_receipt_ref["path"] and x["sha256"] == old_receipt_ref["sha256"] for x in verification["artifacts"])
        identities.append({"issue_id": item["issue_id"], "original_report": item["original_report"], "json_pointer": item["json_pointer"], "original_issue_canonical_sha256": hashlib.sha256(json.dumps(issue, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "necessity": item["necessity"], "historical_fix_verification": item["historical_fix_verification"]})
    assert len({(x["original_report"]["path"], x["original_report"]["sha256"], x["json_pointer"]) for x in identities}) == 3
    return {"created_at": datetime.now(UTC).isoformat(), "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-9-15-v1/verify_historical_appendix.py inspect", "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "device": "cpu; metadata and text inspection only"}, "packet": reference(PACKET) | {"json_pointer": "/groups/4"}, "full_review": reference(FULL), "current_source_freeze": reference(FREEZE), "current_source_sha256": page["source_sha256"], "current_source_quote": quote, "source_and_referenced_code_unchanged": True, "identities": identities, "historical_official_receipt": old_receipt_ref | {"accessed_utc": official["accessed_utc"], "http_date": official["http_date"], "body_sha256": official["body_sha256"], "formula_excerpt": official["excerpts"]["choice_equation"], "scope": "Inspected preserved receipt and short excerpts; original full body was not saved, and was not read or re-retrieved by this reviewer."}, "current_official_receipt": reference(BASE / "source-inspection-v1.json") | {"body_sha256": sha(primary), "headers_sha256": sha(headers), "response_date": "2026-10-29T14:43:18Z", "formula_read_in_original": True}, "result": "Three distinct identities and their original optional necessities retained; current wording is supported by this reviewer's actual 2026-10-29 official primary reading. No historical 2026-10-16/20 assertion is inferred from today's page. No new issue found."}


def bounded_check(appendix_path):
    appendix = api.document(ROOT, reference(appendix_path))
    historical = module("historical_915_v2", ROOT / "docs/review-tools/historical_fix_rechecks_v2.py")
    technical = module("historical_915_technical", ROOT / "scripts/check_technical_reviews.py")
    setup = module("historical_915_original", ROOT / "scripts/omni_revision/review_setup.py")
    original = api.document(ROOT, appendix["original_review"])
    setup.check_original(original, appendix["original_source_freeze"], "9.15", "9.15", OWNER, technical, api)
    assert appendix["full_review"] == reference(FULL)
    assert api.owner(appendix) == api.owner(full) == OWNER
    assert appendix["original_source_freeze"] == original["source_freeze"]
    check = historical.Rechecks.__new__(historical.Rechecks)
    check.root, check.api = ROOT, api
    # Bounded context comes directly from the immutable request and authenticated
    # accepted disposition bytes. It is not a fabricated current stage/manifest.
    check.manifest = {"author_tasks": [packet_document["author_task"]]}
    check.entries, check.used, check.reports, check.checked_readers = {}, set(), [], set()
    check.prior(group["prior_active_pointer"])
    check.decisions = {historical._legacy.issue_key(x): x for x in check.old_dispositions["items"]}
    check.live, check.frozen = {"9.15": page}, {"9.15": page}
    check.tasks = {OWNER: "technical", "/root/omni_reader_decision": "reader"}
    check.validate_page("9.15", OWNER, full, appendix)
    rows = appendix["historical_issue_rechecks"]
    assert len(rows) == 3
    for expected, row in zip(group["items"], rows, strict=True):
        for field in ["disposition_ref", "original_report", "json_pointer", "historical_fix_verification", "issue_id", "necessity"]:
            assert row[field] == expected[field]
        original_issue = api.node(api.document(ROOT, row["original_report"]), row["json_pointer"])
        assert row["original_issue"] == original_issue
        assert row["pages"]["9.15"]["source_quote"] in body
        assert row["pages"]["9.15"]["source_sha256"] == page["source_sha256"]
        check.closed_item(group, row, full, appendix, OWNER, "9.15", api.at(appendix["created_at"]))
    assert len(check.entries) == 3
    for row in rows:
        old = api.node(check.old_dispositions, row["disposition_ref"]["json_pointer"])
        assert check.page(old, row["historical_fix_verification"], "9.15")
    check.finish()
    return {"created_at": datetime.now(UTC).isoformat(), "status": "passed_bounded_metadata_checks", "full_review": reference(FULL), "appendix": reference(appendix_path), "source_freeze": reference(FREEZE), "original_source_freeze": appendix["original_source_freeze"], "identities_checked": [row["issue_id"] for row in rows], "api_methods": ["review_setup.check_original", "Rechecks.prior", "Rechecks.validate_page", "Rechecks.closed_item x3", "Rechecks.page x3", "Rechecks.finish"], "original_issue_dicts_preserved": True, "original_optional_necessities_preserved": True, "authenticated_prior_pointer_and_gate": True, "unavailability_receipts_checked_by_api": True, "current_full_and_appendix_validate": True, "scope": "Bounded API identity/source/reference check only. The preserved accepted decisions supply the bounded decision context; this does not certify the assembled current disposition manifest. No Rechecks.__init__/group or stage/seal/collection validation was performed, and no full gate is claimed."}


if __name__ == "__main__":
    if sys.argv[1] == "inspect":
        result = inspect_originals()
    elif sys.argv[1] == "check":
        result = bounded_check(ROOT / sys.argv[2])
    else:
        raise ValueError("Use inspect or check APPENDIX")
    print(json.dumps(result, ensure_ascii=False, indent=2))
