"""19.20 的契約與歷史報告核對；不訓練、不重跑模型推論。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices


ROOT = Path(__file__).resolve().parents[4]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def main():
    freeze = read_json("docs/course-revision-20261016/freeze-08.json")
    page = next(p for p in freeze["pages"] if p["page_id"] == "19.20")
    source = "course/chapters/19.md"
    sections = lesson_slices((ROOT / source).read_text(encoding="utf-8"))
    body = sections["19.20"]
    source_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert source_sha == page["source_sha256"]
    assert body == (ROOT / page["snapshot"]).read_text(encoding="utf-8")
    assert "狀態追蹤正確率至少90%" in sections["19.15"]
    assert "Choice正確率至少90%" in sections["19.16"]
    assert "至少99%的語義選擇一致率" in sections["19.16"]
    assert "最後測試至少90%正確" in body
    assert "Choice至少90%" in body
    assert "重排語義一致至少99%" in body
    assert "這些數字是開訓前的要求，實際成績尚待驗證" in body

    public_path = "docs/selftrained/results/v2-final-public-results.json"
    public = read_json(public_path)
    historical = {}
    for name in ("moe", "dense"):
        architecture = public["architectures"][name]
        metric_path = f"docs/selftrained/results/public-raw/{name}/test/metrics.json"
        index_path = f"docs/selftrained/results/v2-{name}-public-final-test-stage-index.json"
        metric = read_json(metric_path)
        index = read_json(index_path)
        final = architecture["final_test"]
        assert final["metrics_sha256"] == digest(metric_path)
        assert metric["safe_weights_sha256"] == architecture["safe_weights"]["sha256"]
        assert metric["selected_checkpoint_sha256"] == architecture["selected_source"]["sha256"]
        assert final["run_id"] == index["run"]["attempt"]
        assert final["count"] == metric["count"] == index["completed_count"] == 3734
        assert metric["evaluation_complete"] is True
        assert metric["split"] == "test"
        assert metric["teacher_forcing_used_for_generation"] is False
        tool = metric["per_task_final_reply"]["tool_call"]["tool_roundtrip"]
        threshold = metric["thresholds"]["tool_roundtrip"]
        recomputed = tool["numerator"] / tool["denominator"]
        assert recomputed == tool["rate"]
        assert threshold == index["original_thresholds"]["tool_roundtrip"] == 0.95
        assert recomputed < threshold
        assert architecture["capability_acceptance_passed"] is False
        assert index["model_capability_acceptance"] is False
        historical[name] = {
            "run_id": final["run_id"],
            "evaluation_revision": final["revision"],
            "safe_weights_sha256": metric["safe_weights_sha256"],
            "selected_checkpoint_sha256": metric["selected_checkpoint_sha256"],
            "training_seed": index["checkpoint"]["origin"]["seed"],
            "test_rows": metric["count"],
            "tool_roundtrip": tool,
            "tool_source_groups": metric["per_task_final_reply"]["tool_call"]["source_groups"],
            "original_tool_minimum": threshold,
            "capability_acceptance_passed": architecture["capability_acceptance_passed"],
            "metric_path": metric_path,
            "metric_sha256": digest(metric_path),
            "index_path": index_path,
            "index_sha256": digest(index_path),
        }
    result = {
        "reviewer_task": "/root/tech_20261016_19_20",
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.20-audit.py",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "CPU JSON/text audit; no model execution or training",
        },
        "source_sha256": source_sha,
        "freeze_snapshot_matches": True,
        "prospective_contracts": {
            "B_state_tracking_min_accuracy": 0.90,
            "C_choice_min_accuracy": 0.90,
            "C_reordered_semantic_consistency_min": 0.99,
            "matches_prior_definitions": ["19.15", "19.16"],
            "observed_B_C_results": "not claimed or measured; pretraining acceptance requirements only",
        },
        "historical_A": {
            "public_path": public_path,
            "public_sha256": digest(public_path),
            "hf_revision": public["hf_revision"],
            "architectures": historical,
        },
        "result": "PASS: frozen source and prospective requirements agree; each historical A architecture fails original 95% tool-roundtrip criterion.",
        "scope": "Recomputed existing report fractions and checked versions only; historical GPU measurements were not rerun and no B/C performance was inferred.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
