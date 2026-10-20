import hashlib
import json
import platform
from pathlib import Path

record_path = Path("docs/course-experiments/results/contrastive.json")
code_path = Path("scripts/course_experiments/modalities.py")
record = json.loads(record_path.read_bytes())
results = record["results"]
summary = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-9/inspect-experiment.py",
    "environment": {"python": platform.python_version(), "device": "cpu; JSON inspection only"},
    "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [record_path, code_path]],
    "recorded_environment": {key: str(record[key]) for key in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status"]},
    "config": results["config"],
    "scope": results["scope"],
    "split_counts": {key: value["count"] for key, value in results["data"]["splits"].items()},
    "variants": {},
    "inspection_scope": "核對既有完整執行紀錄與訓練入口，沒有重跑訓練、沒有重新下載權重。"
}
assert record["evidence_status"] == "complete_run"
assert record["unfinished_schedules"] == []
for name, variant in results["variants"].items():
    training = variant["training"]
    history = training["history"]
    assert len(history) == 250 and history[-1]["step"] == 250
    assert all(row["effective_targets"] == 6 for row in history)
    evaluations = {}
    for split in ["validation", "test"]:
        evaluation = variant[split]
        for samples, metric in [("image_samples", "image_to_text_accuracy"), ("text_samples", "text_to_image_accuracy")]:
            observed = sum(sample["correct"] for sample in evaluation[samples]) / len(evaluation[samples])
            assert observed == evaluation[metric]
        evaluations[split] = {key: evaluation[key] for key in ["image_queries", "text_queries", "image_to_text_accuracy", "text_to_image_accuracy"]}
    summary["variants"][name] = {
        "history_count": len(history), "last_step": history[-1]["step"], "targets_per_update": 6,
        "initial_loss": training["initial_loss"], "final_loss": training["final_loss"],
        "evaluations": evaluations
    }
print(json.dumps(summary, ensure_ascii=False, indent=2))
