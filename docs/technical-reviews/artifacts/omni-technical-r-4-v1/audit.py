"""R.4 的有界 CPU／來源核對；不訓練、不下載、不聲稱重現歷史全測試。"""

import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import torch

from tiny_perceptron.lesson_products.inference import EFFORT, Product, auto_solve, check_manifest, rule_depth

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("audit-output.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(relative):
    return json.loads((ROOT / relative).read_bytes())


def ref(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": digest(path), "bytes": path.stat().st_size}


def section(relative, identifier):
    text = (ROOT / relative).read_text()
    match = re.search(r"^## " + re.escape(identifier) + r"\s.*?(?=^## |\Z)", text, re.M | re.S)
    if match is None:
        raise ValueError(identifier)
    return {"path": relative, "selector": "section:" + identifier,
            "start_line": text[:match.start()].count("\n") + 1,
            "sha256": hashlib.sha256(match.group().encode()).hexdigest()}


torch.set_num_threads(1)
freeze = read_json("docs/course-revision-20261029-omni/reviews/numbered-freeze-v3.json")
page = next(p for p in freeze["pages"] if p["page_id"] == "R.4")
assert digest(ROOT / page["snapshot"]) == page["source_sha256"]
assert section("course/README.md", "R.4")["sha256"] == page["source_sha256"]
assert page["figures_sha256"] == {}
body = (ROOT / page["snapshot"]).read_text()
assert "```" not in body and not re.search(r"!\[.*?\]\(", body)

sources = [
    "tiny_perceptron/lesson_products/inference.py",
    "tiny_perceptron/lesson_products/model.py",
    "tiny_perceptron/lesson_products/train.py",
    "tiny_perceptron/lesson_products/release.json",
    "docs/lesson-products/data-v2-manifest.json",
    "docs/lesson-products/results/initial-b-execution.json",
    "docs/lesson-products/results/initial-c-execution.json",
    "docs/lesson-products/results/final-report-v1.json",
    "docs/lesson-products/results/b-auto-policy.json",
    "docs/selftrained/v2-manifest.json",
    "docs/selftrained/results/training-raw/moe-pretrain/raw/train-receipt.json",
    "docs/selftrained/results/training-raw/moe-native/raw/train-receipt.json",
    "docs/selftrained/results/public-raw/moe/test/metrics.json",
    "LICENSE", "pyproject.toml", "assets/training/README.md",
    "docs/course-experiments/README.md",
]
origins = {}
for label, path in [
    ("a_pretrain", sources[10]), ("a_native", sources[11]),
]:
    record = read_json(path)
    origins[label] = {k: record[k] for k in ["schema", "architecture", "stage", "steps", "seed",
                                          "train_records", "validation_records", "test_used_for_selection",
                                          "origin", "selection"]}
    assert record["origin"]["kind"] == "all-neural-weights-random"
    assert record["test_used_for_selection"] is False

release = read_json("tiny_perceptron/lesson_products/release.json")
packages = {}
for product in ["b", "c"]:
    directory = ROOT / "outputs/omni-revision/baseline" / product
    manifest = check_manifest(directory, product)
    assert digest(directory / "artifact-manifest.json") == release["products"][product]["manifest_sha256"]
    receipt = json.loads((directory / "training-receipt.json").read_bytes())
    execution = read_json(f"docs/lesson-products/results/initial-{product}-execution.json")
    assert receipt["random_initialization"] is True and receipt["resume_source"] is None
    for key in ["initial_weights_sha256", "selected_weights_sha256", "data_manifest_sha256",
                "seed", "selected_step", "random_initialization"]:
        assert receipt[key] == execution["training"][key]
    origins[product] = {k: receipt[k] for k in ["product", "mode", "seed", "batch_size", "learning_rate",
                       "steps_this_attempt", "selected_step", "random_initialization", "resume_source",
                       "weights_changed", "data_manifest_sha256", "initial_weights_sha256", "selected_weights_sha256"]}
    packages[product] = {"path": str(directory.relative_to(ROOT)),
                         "manifest_sha256": digest(directory / "artifact-manifest.json"),
                         "files_verified": len(manifest["files"]), "files": manifest["files"]}

b = Product(ROOT / "outputs/omni-revision/baseline/b", "b", device="cpu")
c = Product(ROOT / "outputs/omni-revision/baseline/c", "c", device="cpu")
state = "兩盒追蹤;A=2;B=6;事件=交換;查詢=B"
cases = {
    "b_complete": b.solve(state, loops=3),
    "b_sum_once": b.solve("三數加法;x=1;y=2;z=3", loops=5),
    "c_missing_card": c.judge("卡片進度;申請編號=?", "info"),
    "auto_complete": auto_solve(b, c, state),
    "auto_missing": auto_solve(b, c, "兩盒追蹤;A=?;B=6;事件=交換;查詢=B"),
    "auto_unsupported": auto_solve(b, c, "天氣預測"),
    "auto_budget": auto_solve(b, c, state, effort="medium", max_loops=2),
}
assert cases["b_complete"]["answer"] == 2
assert cases["c_missing_card"]["selected"] == "no"
assert cases["auto_complete"]["status"] == "answered" and cases["auto_complete"]["b_loops"] == 3
assert cases["auto_missing"]["status"] == "clarification_needed" and cases["auto_missing"]["b_loops"] == 0
assert cases["auto_unsupported"]["status"] == "unsupported" and cases["auto_unsupported"]["b_loops"] == 0
assert cases["auto_budget"]["status"] == "budget_limited" and cases["auto_budget"]["b_loops"] == 2
policy = read_json("docs/lesson-products/results/b-auto-policy.json")
depths = [rule_depth(policy, "boxes", n) for n in range(1, 7)]
assert depths == [3, 3, 3, 3, 5, 5] and rule_depth(policy, "sum") == 1

a_metrics = read_json("docs/selftrained/results/public-raw/moe/test/metrics.json")
assert a_metrics["count"] == a_metrics["expected_count"] == 3734
assert a_metrics["evaluation_complete"] and not a_metrics["limited_smoke"]
assert sum(row["count"] for row in a_metrics["per_task_final_reply"].values()) == 3734
a_tasks = {k: {"count": a_metrics["per_task_final_reply"][k]["count"],
               "semantic": a_metrics["per_task_final_reply"][k]["semantic"]}
           for k in ["ocr", "tool_call", "voice_qa", "voice_topic_continuation"]}
final = read_json("docs/lesson-products/results/final-report-v1.json")
bc_metrics = {"b_loops5": {k: final["b"]["5"][k] for k in ["boxes", "sum"]},
              "c_choice": final["c"]["choice"]["after"],
              "c_wording": final["shifts"]["c_wording"], "scope": final["scope"]}
assert bc_metrics["b_loops5"]["sum"]["correct"] < bc_metrics["b_loops5"]["sum"]["count"]
assert bc_metrics["c_wording"]["choice"]["after"]["correct"] < bc_metrics["c_choice"]["count"]

context = [section("course/first-steps.md", "W.5")]
context += [section("course/chapters/19.md", identifier) for identifier in
            ["19.11", "19.12", "19.13", "19.17", "19.19", "19.20", "19.21", "19.22", "19.23", "19.24"]]
roadmap = {"new_product_module_exists": (ROOT / "tiny_perceptron/omni_products").exists(),
           "interpretation": "absence alone does not prove no training anywhere; R.4 status also uses assigned task's explicit not-yet-trained constraint and the exact linked CPU-reference slices"}
result = {
    "created_at": datetime.now(UTC).isoformat(),
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-r-4-v1/audit.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                    "threads": "1", "default_dtype": str(torch.get_default_dtype())},
    "freeze": {"path": "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3.json",
               "sha256": digest(ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v3.json"),
               "source_snapshot": page["snapshot"], "source_sha256": page["source_sha256"],
               "current_slice_equal": True, "figures_sha256": {}, "embedded_code_fences": 0},
    "inspected_files": [ref(path) for path in sources], "necessary_context_slices": context,
    "original_training_origins": origins, "verified_old_release": release, "verified_packages": packages,
    "actual_models": {"b": type(b.model).__name__, "c": type(c.model).__name__},
    "cpu_inference_cases": cases, "rule_depth_events1to6": depths, "EFFORT": EFFORT,
    "historical_original_aggregates": {"a_total": 3734, "a_tasks": a_tasks, "bc": bc_metrics},
    "historical_aggregate_verification": "Parsed original machine-generated metrics and checked counters/configuration; did not rerun the historical 3734 A, 2998 B or full C tests, and did not obtain the 56 MB per-item prediction file.",
    "roadmap": roadmap,
    "scope": "Only R.4 and named necessary linked context. Seven CPU inference cases plus rule lookup; no optimization, GPU, external-model inference, new-product training, latency comparison, or quality estimate. Historical sample counts are original denominators, not counts of this run."
}
with OUT.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"output": str(OUT.relative_to(ROOT)), "sha256": digest(OUT),
                  "environment": result["environment"], "b_sum_answer": cases["b_sum_once"]["answer"],
                  "statuses": {k: v.get("status", "direct") for k, v in cases.items()}}, ensure_ascii=False))
