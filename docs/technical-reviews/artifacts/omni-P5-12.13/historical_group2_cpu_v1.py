"""Post-original prompted recheck of exactly group 2; no model execution."""

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
REQUEST = "docs/course-revision-20261029-omni/reviews/prior-fix-appendix-requests-v2.json"
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json"
FULL = "docs/course-revision-20261029-omni/reviews/P5-actual/technical-12.13/technical-original-v1.json"


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def document(path):
    return json.loads((ROOT / path).read_bytes())


def node(value, pointer):
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def main():
    torch.set_num_threads(1)
    group = document(REQUEST)["groups"][2]
    assert group["page_id"] == "12.13" and len(group["items"]) == 2
    assert [i["json_pointer"] for i in group["items"]] == ["/issues/0", "/issues/1"]
    full = document(FULL)
    frozen = document(FREEZE)
    page = next(p for p in frozen["pages"] if p["page_id"] == "12.13")
    body = (ROOT / page["snapshot"]).read_text()
    assert sha(page["snapshot"]) == full["source_sha256"] == page["source_sha256"]
    assert page["figures_sha256"] == full["figure_sha256"]
    assert sha(page["notebook"]) == full["notebook"]["sha256"] == page["notebook_sha256"]
    for name, digest in page["figures_sha256"].items():
        assert sha(name) == digest
    identity_checks = []
    for item in group["items"]:
        for label, ref in [("disposition", item["disposition_ref"]), ("original_report", item["original_report"]), ("historical_fix_verification", item["historical_fix_verification"]["report"])]:
            assert sha(ref["path"]) == ref["sha256"]
            identity_checks.append({"kind": label, "path": ref["path"], "sha256": ref["sha256"]})
        old = node(document(item["disposition_ref"]["path"]), item["disposition_ref"]["json_pointer"])
        issue = node(document(item["original_report"]["path"]), item["json_pointer"])
        assert old["final_necessity"] == item["necessity"] == "necessary"
        assert issue["necessary"] is True
        assert old["report"] == item["original_report"]
        assert old["json_pointer"] == item["json_pointer"]
        assert old["issue_id"] == item["issue_id"]
        assert item["historical_fix_verification"] in old["verification"]
    unavailable = group["unavailability"]
    assert sha(unavailable["report"]["path"]) == unavailable["report"]["sha256"]
    receipt = node(document(unavailable["report"]["path"]), unavailable["json_pointer"])
    assert receipt["reviewer_task"] == group["old_owner"] and receipt["unavailable"] is True
    assert receipt["events"][0]["response"] == "agent_name must use only lowercase letters, digits, and underscores"
    assert receipt["events"][1]["response"]["agents"] == []

    mean_quote = "它保留每個 mel 頻帶的平均 log 特徵，卻失去「先低後高」的順序。"
    supervision_quote = "用監督式訓練教模型回答先後時，還要提供帶有先後答案的資料，訓練目標才會要求模型利用這些位置。"
    assert mean_quote in body and supervision_quote in body
    assert "sequence = (bank @ spectrum.abs().square()).clamp(min=1e-8).log().T" in body
    assert "sequence.mean(0)" in body
    assert "它保留兩種頻率的總量" not in body
    assert "模型才學得會利用這些位置" not in body

    # Each row is a two-frame positive-power example, safely above the log floor.
    power = torch.tensor([[1.0, 9.0], [3.0, 3.0]], dtype=torch.float64)
    summary = power.clamp(min=1e-8).log().mean(dim=1)
    totals = power.sum(dim=1)
    assert torch.allclose(summary, torch.full_like(summary, math.log(3)), rtol=0, atol=1e-15)
    assert torch.equal(totals, torch.tensor([10.0, 6.0], dtype=torch.float64))
    assert not bool(totals[0] == totals[1])
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_task": "/root/omni_p5_technical_12_13",
        "context": "Prompted historical recheck after immutable independent full original; not a new independent initial finding.",
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-12.13/historical_group2_cpu_v1.py",
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "device": "CPU", "dtype": "float64", "threads": str(torch.get_num_threads())},
        "request": {"path": REQUEST, "sha256": sha(REQUEST), "json_pointer": "/groups/2"},
        "full_review": {"path": FULL, "sha256": sha(FULL), "created_at": full["created_at"]},
        "current_source_freeze": {"path": FREEZE, "sha256": sha(FREEZE)},
        "source_figure_notebook_unchanged_from_full_review": True,
        "current_source_sha256": page["source_sha256"],
        "current_figure_sha256": page["figures_sha256"],
        "current_notebook_sha256": page["notebook_sha256"],
        "requested_instance_count": 2,
        "retained_original_issue_pointers": [i["json_pointer"] for i in group["items"]],
        "duplicate_handling": "Two separate requested instances retained, with repeated report references; neither merged nor deduplicated by report/owner.",
        "historical_identity_checks": identity_checks,
        "unavailability": unavailable,
        "unavailability_event_meaning": "Name rejection plus actual empty prefix listing; no claim that rejection itself was path-not-found; events are preserved historical receipts, not newly emitted events.",
        "current_source_quotes": [mean_quote, supervision_quote],
        "counterexample": {"power_rows": power.tolist(), "log_means": summary.tolist(), "expected_log_mean": math.log(3), "totals": totals.tolist(), "log_mean_comparison_rtol": 0.0, "log_mean_comparison_atol": 1e-15, "floor_inactive": True},
        "result": "All assertions passed. Mean-log summary is not total power. Current wording explicitly restricts the order-answer data statement to supervised teaching. No training or neural inference executed.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
