"""Bounded personal recheck of the two supplied historical 12.13 fixes."""

import contextlib
from datetime import UTC, datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import re
import sys

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("historical-rv1019-execution-v1.json")
started = datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("historical_current_api", ROOT / "docs/review-tools/incremental_revision.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
request_path = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
group = json.loads((ROOT / request_path).read_bytes())["groups"][2]
assert group["page_id"] == "12.13"
assert group["old_owner"] == "/root/rv1019_technical_12-13"
assert len(group["items"]) == 2
own_ref = {
    "path": "docs/course-revision-20261029-omni/reviews/technical/technical_12_13/technical-original-v1.json",
    "sha256": "21075b0c15289a82eec82c5847d5052011204d3878582a0b5bb177370b9079bf",
}
own = api.document(ROOT, own_ref)
assert own["reviewer_task"] == "/root/omni_technical_coordinator/technical_12_13"
assert api.at(own["created_at"]) < api.at(started)
pointer = api.document(ROOT, group["prior_active_pointer"])
prior_manifest_ref = {"path": pointer["manifest"], "sha256": pointer["manifest_sha256"]}
prior_manifest = api.document(ROOT, prior_manifest_ref)
prior_freeze = api.document(ROOT, prior_manifest["source_freeze"])
prior_page = next(p for p in prior_freeze["pages"] if p["page_id"] == "12.13")
prior_assignments = api.document(ROOT, prior_manifest["assignments"])
old_assignment = next(a for a in prior_assignments["tasks"] if a["task"] == group["old_owner"])
assert old_assignment["role"] == "technical" and old_assignment["lesson_id"] == "12.13"
prior_dispositions = api.document(ROOT, prior_manifest["dispositions"])
assert prior_dispositions["pending_count"] == 0
api.artifact(ROOT, {"path": prior_page["snapshot"], "sha256": prior_page["source_sha256"]})
prior_gate = api.document(ROOT, pointer["actual_completed_gate"])
assert prior_gate["stage"] == "all" and prior_gate["status"] == "passed_metadata_checks"
assert prior_gate["failures"] == []
gate_page = next(p for p in prior_gate["records"] if p["page_id"] == "12.13")
assert gate_page["source_sha256"] == prior_page["source_sha256"]

identity_rows = []
for item in group["items"]:
    assert {k: item["disposition_ref"][k] for k in ("path", "sha256")} == prior_manifest["dispositions"]
    decision = api.node(prior_dispositions, item["disposition_ref"]["json_pointer"])
    assert decision["issue_id"] == item["issue_id"]
    assert decision["report"] == item["original_report"]
    assert decision["json_pointer"] == item["json_pointer"]
    assert decision["action"] == "fixed" and decision["final_necessity"] == item["necessity"] == "necessary"
    assert decision["affected_pages"] == item["affected_pages"] == ["12.13"]
    assert item["historical_fix_verification"] in decision["verification"]
    original = api.document(ROOT, item["original_report"])
    issue = api.node(original, item["json_pointer"])
    old_current = api.document(ROOT, item["historical_fix_verification"]["report"])
    previous = api.node(old_current, item["historical_fix_verification"]["json_pointer"])
    assert api.owner(original) == api.owner(old_current) == group["old_owner"]
    assert previous["source_sha256"] == prior_page["source_sha256"]
    assert previous["figure_sha256"] == prior_page["figures_sha256"]
    identity_rows.append({"issue_id": item["issue_id"], "disposition_pointer": item["disposition_ref"]["json_pointer"],
                          "original_issue_pointer": item["json_pointer"], "original_claim_id": issue["claim_id"],
                          "prior_verified_source_sha256": previous["source_sha256"], "identity_checks": "passed"})

unavailable_document = api.document(ROOT, group["unavailability"]["report"])
unavailable = api.node(unavailable_document, group["unavailability"]["json_pointer"])
assert unavailable["reviewer_task"] == group["old_owner"] and unavailable["unavailable"] is True
assert api.at(pointer["activated_at"]) <= api.at(unavailable["recorded_at"]) <= api.at(started)
events = unavailable["events"]
followups = [e for e in events if e["tool"] == "collaboration.followup_task"
             and e["request"]["target"] == group["old_owner"]]
assert len(followups) == 1
assert followups[0]["response"] == "agent_name must use only lowercase letters, digits, and underscores"
assert any(e["tool"] == "collaboration.list_agents" and
           group["old_owner"].startswith(e["request"]["path_prefix"]) and
           e["response"].get("agents") == [] for e in events)

freeze_path = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix11.json"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "12.13")
raw = api.artifact(ROOT, {"path": page["snapshot"], "sha256": page["source_sha256"]})
body = raw.decode()
assert re.search(r"^## 12\.13 .*?(?=^## |\Z)", (ROOT / page["source"]).read_text(), re.M | re.S)[0] == body
for path, digest in page["figures_sha256"].items():
    assert sha(ROOT / path) == digest
    assert hashlib.sha256(api.artifact(ROOT, prior_freeze["figure_snapshots"][path])).hexdigest() == digest
code = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(code) == 1
torch.set_num_threads(1)
namespace = {"__name__": "__historical_12_13_recheck__"}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code[0], page["snapshot"], "exec"), namespace)
assert stdout.getvalue() == "時間框與頻帶 21 16\n先後序列相同 False\n平均摘要近似相同 True\n"
features = namespace["sequence"]
assert torch.allclose(features.mean(0), features.sum(0) / len(features))

# Transparent one-band/two-time-point counterexample; not acoustic measurements.
power = torch.tensor([[1.0, 9.0], [3.0, 3.0]], dtype=torch.float64)
log_mean = power.clamp(min=1e-8).log().mean(1)
totals = power.sum(1)
assert torch.allclose(log_mean, torch.full((2,), math.log(3), dtype=torch.float64), atol=1e-15, rtol=0)
assert totals.tolist() == [10.0, 6.0]

# Deterministic loss values only; neither training nor accuracy measurement.
targets = torch.tensor([0, 1], dtype=torch.long)
shared_logits = torch.zeros(2, 2, dtype=torch.float64)
ordered_logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], dtype=torch.float64)
shared_loss = F.cross_entropy(shared_logits, targets)
ordered_loss = F.cross_entropy(ordered_logits, targets)
swapped_target_loss = F.cross_entropy(ordered_logits, targets.flip(0))
assert abs(float(shared_loss) - math.log(2)) <= 1e-14
assert float(ordered_loss) < float(shared_loss) < float(swapped_target_loss)

result = {
    "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-12-13-v1/historical-rv1019-recheck-v1.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "torch_git_version": torch.version.git_version,
                    "device": "cpu", "threads": str(torch.get_num_threads()), "platform": platform.platform()},
    "request": {"path": request_path, "sha256": sha(ROOT / request_path), "json_pointer": "/groups/2"},
    "own_full_original_unchanged": own_ref, "current_source_sha256": page["source_sha256"],
    "current_figure_sha256": page["figures_sha256"], "old_owner": group["old_owner"],
    "prior_accepted_manifest": prior_manifest_ref, "historical_identity_checks": identity_rows,
    "unavailability": {"reference": group["unavailability"], "actual_record_checked": True,
                       "exact_followup_name_rejection_and_empty_prefix_listing": True,
                       "scope": "Name rejection is not path-not-found. Current-manifest author membership/collection and seal remain pending."},
    "verbatim_current_embedded_stdout": stdout.getvalue(),
    "current_features": {"shape": list(features.shape), "mean_equals_sum_div_count_with_roundoff_tolerance": True,
                         "mean_reversal_max_abs_difference": float((features.mean(0) - namespace["reordered"].mean(0)).abs().max())},
    "c9_counterexample": {"input_powers_by_scenario_then_time": power.tolist(), "mean_log": log_mean.tolist(),
                          "total_power": totals.tolist(), "expected_mean_log": math.log(3), "atol": 1e-15,
                          "interpretation": "Equal mean-log summaries do not identify total powers; neither mel band is an isolated exact-frequency energy meter."},
    "c11_objective": {"targets": targets.tolist(), "shared_logits": shared_logits.tolist(),
                      "shared_mean_ce": float(shared_loss), "hand_minimum_shared_mean_ce": math.log(2),
                      "ordered_logits": ordered_logits.tolist(), "ordered_mean_ce": float(ordered_loss),
                      "same_logits_opposite_targets_mean_ce": float(swapped_target_loss),
                      "interpretation": "The direct supervised objective distinguishes opposite answers. No optimizer, weight update, trained order accuracy or generalization is established."},
    "api_used": {"path": "docs/review-tools/incremental_revision.py", "sha256": sha(ROOT / "docs/review-tools/incremental_revision.py"),
                 "methods": ["artifact", "document", "node", "owner", "at"]},
    "pending": ["Current appendix receipt/collection_index", "Current assignment window and phase seal", "Current historical sidecar/manifest assembly and complete gate"]
}
assert sha(ROOT / own_ref["path"]) == own_ref["sha256"]
with OUT.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"output": OUT.relative_to(ROOT).as_posix(), "sha256": sha(OUT),
                  "identity_checks": identity_rows, "stdout": stdout.getvalue(),
                  "mean_log": log_mean.tolist(), "total_power": totals.tolist(),
                  "shared_ce": float(shared_loss), "ordered_ce": float(ordered_loss)}, ensure_ascii=False))
