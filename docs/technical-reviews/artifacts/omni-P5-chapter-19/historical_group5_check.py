"""After-original group-5 historical identity/source/evidence checks; no neural work."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROOF = Path(__file__).resolve().parent
BASE = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual"
OUTPUT = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
          "reviewer_task": "/root/omni_p5_technical_chapter_19", "device": "CPU; metadata/hash/string checks only",
          "inputs": [], "scope": "Only prior-fix request group 5, after immutable own full original was actually received. Not an independent rediscovery, seal, collection, or complete assembled historical gate."}


def read(path, digest=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    raw = p.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if digest is not None:
        assert actual == digest, (path, digest, actual)
    OUTPUT["inputs"].append({"path": p.relative_to(ROOT).as_posix(), "sha256": actual, "bytes": len(raw)})
    return raw


def doc(reference):
    return json.loads(read(reference["path"], reference["sha256"]))


def node(value, pointer):
    for item in pointer.split("/")[1:]:
        item = item.replace("~1", "/").replace("~0", "~")
        value = value[int(item)] if isinstance(value, list) else value[item]
    return value


request_path = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
requests = json.loads(read(request_path))
group = requests["groups"][5]
assert group["page_id"] == "chapter-19" and group["kind"] == "page" and len(group["items"]) == 2
OUTPUT["request_group_index"] = 5
OUTPUT["request_group"] = group
full_ref = {"path": "docs/course-revision-20261029-omni/reviews/P5-actual/technical-chapter-19/technical-original-v1.json",
            "sha256": "46ffa79304701610f026614d51294399d983b76748e958e4131bfab14c5bee8e"}
full = doc(full_ref)
assert full["reviewer_task"] == OUTPUT["reviewer_task"] and full["verdict"] == "pass"
assert not full.get("historical_issue_rechecks")
OUTPUT["full_review"] = full_ref
OUTPUT["full_review_created_at"] = full["created_at"]
prior_pointer = doc(group["prior_active_pointer"])
prior_manifest = doc({"path": prior_pointer["manifest"], "sha256": prior_pointer["manifest_sha256"]})
prior_dispositions = doc(prior_manifest["dispositions"])
prior_freeze = doc(prior_manifest["source_freeze"])
prior_page = next(row for row in prior_freeze["pages"] if row["page_id"] == "chapter-19")
read(prior_page["snapshot"], prior_page["source_sha256"])
OUTPUT["prior_activated_at"] = prior_pointer["activated_at"]
OUTPUT["prior_frozen_page"] = prior_page

unavailable = node(doc(group["unavailability"]["report"]), group["unavailability"]["json_pointer"])
assert unavailable["reviewer_task"] == group["old_owner"] and unavailable["unavailable"] is True
assert prior_pointer["activated_at"] <= unavailable["recorded_at"] <= OUTPUT["created_at"]
events = unavailable["events"]
followups = [e for e in events if e["tool"] == "collaboration.followup_task" and e["request"]["target"] == group["old_owner"]]
assert len(followups) == 1
assert followups[0]["response"] == "agent_name must use only lowercase letters, digits, and underscores"
assert any(e["tool"] == "collaboration.list_agents" and group["old_owner"].startswith(e["request"]["path_prefix"])
           and e["response"] == {"agents": []} for e in events)
OUTPUT["unavailability_record"] = unavailable
OUTPUT["unavailability_interpretation"] = "Actual preserved invalid-name followup failure plus empty lawful task-prefix listing. Invalid-name is not relabeled path-not-found. No new or fabricated followup event."

freeze_ref = {"path": "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json",
              "sha256": "ddcc500db74f5df9ab3bb6fddcdc511ca177c98ce64e1a94160d1a7ccc1c35be"}
freeze = doc(freeze_ref)
page = next(row for row in freeze["pages"] if row["page_id"] == "chapter-19")
body = read(page["snapshot"], page["source_sha256"]).decode()
quote = "讀固定語法的事件或加法題，輸出答案預測（事件的狀態追蹤已通過預先訂的門檻，加法還沒達標）"
assert quote in body and full["source_sha256"] == page["source_sha256"]
OUTPUT["current_source_freeze"] = freeze_ref
OUTPUT["current_source_sha256"] = page["source_sha256"]
OUTPUT["current_figure_sha256"] = page["figures_sha256"]
OUTPUT["current_source_quote"] = quote

source_map = {s["id"]: s for s in full["sources"]}
for name in ["s_bcinference", "s_bcmodel", "s_freeze", "s_final"]:
    source = source_map[name]
    read(source["path"], source["sha256"])
inference = (ROOT / source_map["s_bcinference"]["path"]).read_text()
model = (ROOT / source_map["s_bcmodel"]["path"]).read_text()
assert "predicted = int(probabilities.argmax())" in inference
assert "self.answer = nn.Linear(config.width, 19)" in model
recount_ref = {"path": "docs/technical-reviews/artifacts/omni-P5-chapter-19/compact-recount-v1.json",
               "sha256": "fdf8719f88b9f2f9e0fbe983767661af"}
recount_path = PROOF / "compact-recount-v1.json"
recount_ref["sha256"] = hashlib.sha256(recount_path.read_bytes()).hexdigest()
recount = doc(recount_ref)
assert recount["results"]["boxes"]["correct"] == 2928 and recount["results"]["boxes"]["count"] == 2956
assert recount["results"]["sum"]["correct"] == 2 and recount["results"]["sum"]["count"] == 42
assert recount["results"]["boxes"]["pass"] and not recount["results"]["sum"]["pass"]
OUTPUT["current_saved_score_recount"] = {"report": recount_ref, "results": recount["results"],
                                         "scope": "Own already saved original pure-logit/compact-row recount, reused after SHA verification. No new 1/3/5-loop neural replay; original old reviewer claims remain attributed to that reviewer."}

OUTPUT["historical_instances"] = []
keys = set()
for item in group["items"]:
    ref = item["disposition_ref"]
    assert {k: ref[k] for k in ["path", "sha256"]} == prior_manifest["dispositions"]
    disposition = node(prior_dispositions, ref["json_pointer"])
    issue_report = doc(item["original_report"])
    issue = node(issue_report, item["json_pointer"])
    verification = item["historical_fix_verification"]
    fixed_report = doc(verification["report"])
    assert disposition["report"] == item["original_report"] and disposition["json_pointer"] == item["json_pointer"]
    assert disposition["issue_id"] == item["issue_id"] and disposition["action"] == "fixed"
    assert disposition["final_necessity"] == item["necessity"] == "necessary"
    assert disposition["affected_pages"] == item["affected_pages"] == ["chapter-19"]
    assert verification in disposition["verification"] and verification["json_pointer"] == ""
    assert issue_report["reviewer_task"] == fixed_report["reviewer_task"] == group["old_owner"]
    assert fixed_report["verdict"] == "pass" and fixed_report["source_sha256"] == prior_page["source_sha256"]
    assert fixed_report["figure_sha256"] == prior_page["figures_sha256"]
    assert node(fixed_report, "/issues/0")["status"] == "resolved"
    key = (item["original_report"]["path"], item["original_report"]["sha256"], item["json_pointer"])
    assert key not in keys
    keys.add(key)
    OUTPUT["historical_instances"].append({"request_item": item, "original_issue": issue,
                                             "original_disposition": disposition,
                                             "historical_verification_issue": node(fixed_report, "/issues/0"),
                                             "status": "still_fixed"})
assert len(keys) == 2
OUTPUT["duplicate_relationship"] = "Two distinct required inventory instances of the same c6 defect: original contradicted issue and current-01 resolved carry-forward. Both identities/pointers/necessary classifications retained; not merged or counted as new defects."
OUTPUT["verdict"] = "pass"
OUTPUT["finished_at"] = datetime.now(timezone.utc).isoformat()
with (PROOF / "historical-group-5-evidence-v1.json").open("x", encoding="utf-8") as f:
    json.dump(OUTPUT, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"created_at": OUTPUT["created_at"], "finished_at": OUTPUT["finished_at"],
                  "verdict": OUTPUT["verdict"], "historical_instances": len(keys),
                  "current_source_quote": quote, "full_review_unchanged": full_ref}, ensure_ascii=False, indent=2))
