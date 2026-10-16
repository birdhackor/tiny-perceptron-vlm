"""Limited CPU review of frozen lesson 14.16; never runs training or final evaluation."""

import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


torch.set_num_threads(2)
body = lesson_slices((ROOT / "course/chapters/14.md").read_text())["14.16"]
freeze = json.loads((ROOT / "docs/course-revision-20261016/freeze-14.json").read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "14.16")
assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
assert body == (ROOT / page["snapshot"]).read_text()
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exec(re.search(r"```python\n(.*?)```", body, re.S)[1], {})
assert buffer.getvalue() == "早輪 0.2 後輪 0.4 懲罰 0.2\n早輪 0.5 後輪 0.4 懲罰 0.0\n"
assumed_losses = [1.2, 0.8, 0.6, 0.7, 0.9]
best_round = 1 + min(range(len(assumed_losses)), key=assumed_losses.__getitem__)
assert best_round == 3
violating_objective = 0.001 + 0.002 + max(0, 0.002 - 0.001)
nonviolating_objective = 1.0 + 1.0 + max(0, 1.0 - 1.0)
assert violating_objective < nonviolating_objective

report_path = ROOT / "docs/lesson-products/results/final-report-v1.json"
report = json.loads(report_path.read_text())
raw = ROOT / "outputs/lesson-products/final-report-v1.predictions.jsonl"
assert sha(raw) == report["predictions"]["sha256"]
assert raw.stat().st_size == report["predictions"]["bytes"]
selected = {"fixed3": (3, 6), "odd": (3, 4)}
rows = {mode: [] for mode in selected}
with raw.open() as stream:
    for line in stream:
        row = json.loads(line)
        if row.get("product") == "mechanism" and row["task"] == "boxes":
            assert row["split"] == "test"
            rows[row["mode"]].append(row)
assert {r["id"] for r in rows["fixed3"]} == {r["id"] for r in rows["odd"]}
summary = {}
for mode, depths in selected.items():
    group = rows[mode]
    assert len(group) == len({r["id"] for r in group}) == 2956
    assert len({r["family_id"] for r in group}) == 1478
    assert all(len(r["scores_by_depth"][str(d)]) == 19 for r in group for d in depths)
    summary[mode] = {}
    for depth in depths:
        losses, correct = [], 0
        for row in group:
            logits = row["scores_by_depth"][str(depth)]
            target = row["gold"]
            peak = max(logits)
            losses.append(peak + math.log(sum(math.exp(v - peak) for v in logits)) - logits[target])
            correct += max(range(len(logits)), key=logits.__getitem__) == target
        ce = math.fsum(losses) / len(group)
        primary = report["mechanisms"][mode]["evaluation"][str(depth)]["boxes"]
        assert correct == primary["correct"]
        assert abs(ce - primary["cross_entropy"]) < 2e-6
        assert abs(correct / len(group) - primary["accuracy"]) < 1e-7
        summary[mode][str(depth)] = {
            "count": len(group), "families": 1478, "correct": correct,
            "accuracy_percent": 100 * correct / len(group),
            "mean_ce_float64_independent": ce,
            "primary_mean_ce_float32": primary["cross_entropy"],
            "display": f"{ce:.4f}/{100 * correct / len(group):.2f}%",
        }

settings_path = ROOT / "docs/lesson-products/results/settings-freeze-v1.json"
settings = json.loads(settings_path.read_text())
assert sha(settings_path) == report["settings_freeze_sha256"]
receipt_checks = {}
for mode in selected:
    directory = ROOT / "outputs/lesson-products/mechanisms" / mode
    for name in ("last.pt", "training-receipt.json"):
        assert sha(directory / name) == settings["mechanisms"][mode][name]
    ckpt = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    receipt = json.loads((directory / "training-receipt.json").read_text())
    runner_name = "mechanism-fixed3-runner-result.json" if mode == "fixed3" else "mechanism-odd135-runner-result.json"
    runner = json.loads((ROOT / "docs/lesson-products/results" / runner_name).read_text())
    assert runner["execution"]["training"] == receipt
    assert receipt["mode"] == ckpt["metadata"]["mode"] == mode
    assert receipt["last_step"] == ckpt["metadata"]["step"] == 1500
    assert receipt["seed"] == settings["mechanisms"][mode]["seed"] == 8159
    assert receipt["batch_size"] == 128
    assert runner["status"] == runner["execution"]["status"] == "completed"
    receipt_checks[mode] = {
        "last_step": ckpt["metadata"]["step"], "mode": mode,
        "seed": receipt["seed"], "batch_size": receipt["batch_size"],
        "last_checkpoint_sha256": sha(directory / "last.pt"),
        "training_receipt_sha256": sha(directory / "training-receipt.json"),
        "initial_weights_sha256": receipt["initial_weights_sha256"],
        "existing_primary_training_environment": receipt["environment"],
        "existing_runner_status": runner["gpu_execution_status"],
    }
assert receipt_checks["fixed3"]["initial_weights_sha256"] == receipt_checks["odd"]["initial_weights_sha256"]

result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/backfill/14.16-check.py",
    "reviewer_task": "/root/backfill_tech_14_16",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "CPU", "platform": platform.platform()},
    "source_sha256": page["source_sha256"],
    "checked_primary_sources": {
        str(p.relative_to(ROOT)): sha(p) for p in [report_path, settings_path, raw,
            ROOT / "docs/lesson-products/results/mechanism-fixed3-runner-result.json",
            ROOT / "docs/lesson-products/results/mechanism-odd135-runner-result.json",
            ROOT / "tiny_perceptron/lesson_products/evaluation.py",
            ROOT / "tiny_perceptron/lesson_products/training.py",
            ROOT / "tiny_perceptron/lesson_products/model.py"]
    },
    "frozen_code_stdout": buffer.getvalue(),
    "hypothetical_minimum": {"losses": assumed_losses, "best_round": best_round},
    "numeric_counterexamples": {
        "ce_down_argmax_wrong": {"gold": 0, "probabilities_before": [0.2, 0.8], "probabilities_after": [0.4, 0.6], "ce_before": -math.log(0.2), "ce_after": -math.log(0.4), "both_argmax": 1},
        "mean_vs_each": {"losses_before": [1.0, 1.0], "losses_after": [0.0, 1.5], "mean_before": 1.0, "mean_after": 0.75, "second_sample_worsens": True},
        "finite_soft_penalty": {"lambda": 1.0, "objective": "L1+L2+lambda*max(0,L2-L1)", "violating_pair": [0.001, 0.002], "violating_objective": violating_objective, "nonviolating_pair": [1.0, 1.0], "nonviolating_objective": nonviolating_objective},
    },
    "score_reaggregation": summary,
    "checkpoint_and_receipt_checks": receipt_checks,
    "scope": "Personally executed frozen code, finite arithmetic, raw-score aggregation and local SHA/metadata checks only. Existing L4 training receipts inspected, no training/GPU dispatch, model forward pass, tuning, final-test restart, Modal call, upload or publication performed.",
    "result": "All assertions passed; four cited mechanism cells reproduce at displayed precision.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
