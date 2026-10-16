"""Independent finite CPU checks of lesson 19.18; never starts formal evaluation."""

import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import torch

from scripts.reading_time import lesson_slices
from tiny_perceptron.lesson_products.inference import Product, auto_solve


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_bytes())


def main():
    result = {
        "reviewer_task": "/root/backfill_tech_19_18",
        "recorded_at": datetime.now(UTC).isoformat(),
        "command": "uv run --frozen --extra cpu python docs/technical-reviews/artifacts/revision-20261016/backfill/19.18-backfill-original-01-probe.py",
        "scope": "Recompute preexisting raw scores and 9 bounded CPU public-API requests. No training, tuning, or formal test entry point invoked. Historical timings are inspected primary receipts, not personally rerun timings.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    }
    freeze = read_json("docs/course-revision-20261016/freeze-14.json")
    lesson = next(page for page in freeze["pages"] if page["page_id"] == "19.18")
    body = lesson_slices((ROOT / lesson["source"]).read_text())["19.18"]
    assert hashlib.sha256(body.encode()).hexdigest() == lesson["source_sha256"]
    assert body == (ROOT / lesson["snapshot"]).read_text()
    result["frozen_section"] = lesson
    report = read_json("docs/lesson-products/results/final-report-v1.json")
    settings = read_json("docs/lesson-products/results/settings-freeze-v1.json")
    assert sha(ROOT / "docs/lesson-products/results/settings-freeze-v1.json") == report["settings_freeze_sha256"]
    manifest_path = ROOT / "outputs/lesson-products/data-v2/manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert sha(manifest_path) == report["data_manifest_sha256"]
    data_path = ROOT / "outputs/lesson-products/data-v2/b.jsonl"
    assert sha(data_path) == manifest["files"]["b.jsonl"]["sha256"]
    rows = {}
    with data_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "test":
                assert row["id"] not in rows
                rows[row["id"]] = row
    raw_path = ROOT / "outputs/lesson-products/final-report-v1.predictions.jsonl"
    assert sha(raw_path) == report["predictions"]["sha256"]
    raw = {}
    with raw_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["product"] == "b" and row["split"] == "test":
                assert row["id"] not in raw
                raw[row["id"]] = row
    assert raw.keys() == rows.keys()
    policy = read_json("outputs/lesson-products/products/b/auto-policy.json")
    assert policy["low_max_events"] == 0 and policy["medium_max_events"] == 4 and policy["sum_depth"] == 1
    for row_id, observed in raw.items():
        row = rows[row_id]
        assert observed["gold"] == row["gold"] and observed["task"] == row["task"]
        for depth, scores in observed["scores_by_depth"].items():
            assert observed["answers_by_depth"][depth] == max(range(len(scores)), key=scores.__getitem__)
    result["raw_identity"] = {"sha256": sha(raw_path), "bytes": raw_path.stat().st_size, "b_test_rows": len(rows), "task_counts": dict(Counter(row["task"] for row in rows.values())), "unique_family_count": len({row["family_id"] for row in rows.values()}), "all_depth_argmaxes_match": True}
    quality = {}
    for mode, fixed in (("low", 1), ("medium", 3), ("high", 5), ("auto", None)):
        total_loops, correct, accepted = 0, 0, 0
        histogram = Counter()
        for row_id, observed in raw.items():
            row = rows[row_id]
            if observed["c_choice"] not in ("boxes", "sum") or observed["c_info"] != "yes":
                continue
            accepted += 1
            if fixed is not None:
                depth = fixed
            elif observed["c_choice"] == "sum":
                depth = policy["sum_depth"]
            else:
                events = row["state"].split("事件=", 1)[1].split(";查詢=", 1)[0].split(",")
                depth = 1 if len(events) <= policy["low_max_events"] else 3 if len(events) <= policy["medium_max_events"] else 5
            histogram[depth] += 1
            total_loops += depth
            correct += observed["answers_by_depth"][str(depth)] == row["gold"]
        quality[mode] = {"count": len(rows), "correct": correct, "accepted": accepted, "accuracy": correct / len(rows), "mean_b_loops": total_loops / len(rows), "loop_histogram": dict(histogram)}
        primary = report["auto"][mode]
        assert correct == primary["correct_end_to_end"]
        assert accepted == primary["covered"]
        assert total_loops / len(rows) == primary["mean_actual_b_loops_all_complete"]
    assert [quality[m]["correct"] for m in quality] == [1598, 2927, 2930, 2926]
    assert [f'{quality[m]["accuracy"] * 100:.2f}%' for m in quality] == ["53.30%", "97.63%", "97.73%", "97.60%"]
    result["independent_quality_recalculation"] = quality
    result["auto_loop_reduction_percent"] = (3 - quality["auto"]["mean_b_loops"]) / 3 * 100
    oracle_correct = sum(any(row["answers_by_depth"][str(d)] == row["gold"] for d in (1, 3, 5)) for row in raw.values())
    result["offline_oracle_check"] = {"candidate_depths": [1, 3, 5], "correct": oracle_correct, "all_depths_wrong_stay_wrong": len(rows) - oracle_correct, "scope": "Accuracy upper bound for choosing one of the recorded outputs, with true labels available. Does not price search or predict generalization."}
    latency = report["latency"]
    sample = sorted(rows, key=lambda row_id: hashlib.sha256(("latency-v1:" + row_id).encode()).hexdigest())[:24]
    assert sample == latency["sample_ids"]
    assert latency["warmup_per_effort"] == 3 and latency["repetitions"] == 3
    assert all(value["requests"] == 24 * 3 for value in latency["measurements"].values())
    result["historical_latency_receipt"] = latency
    result["historical_latency_ms_rounded"] = {m: round(value["integrated_mean_seconds"] * 1000, 2) for m, value in latency["measurements"].items()}
    assert list(result["historical_latency_ms_rounded"].values()) == [3.63, 4.32, 4.99, 4.37]
    assert latency["measurements"]["auto"]["integrated_mean_seconds"] > latency["measurements"]["medium"]["integrated_mean_seconds"]
    result["provenance"] = {}
    for name, expected in settings["source_sha256s"].items():
        path = ROOT / "tiny_perceptron/lesson_products" / name
        result["provenance"][str(path.relative_to(ROOT))] = {"sha256": sha(path), "equals_final_test_freeze": sha(path) == expected}
        assert sha(path) == expected
    for product in ("b", "c"):
        for name, expected in settings["products"][product].items():
            path = ROOT / "outputs/lesson-products/products" / product / name
            assert sha(path) == expected
            result["provenance"][str(path.relative_to(ROOT))] = {"sha256": sha(path), "equals_final_test_freeze": True}
    torch.set_num_threads(2)
    b = Product(ROOT / "outputs/lesson-products/products/b", "b")
    c = Product(ROOT / "outputs/lesson-products/products/c", "c")
    result["environment"]["threads"] = str(torch.get_num_threads())
    result["environment"]["parameter_dtype"] = str(next(b.model.parameters()).dtype)
    api = []
    original_judge = c.judge
    calls = []

    def judge(state, question="choice", candidates=None):
        calls.append(question)
        return original_judge(state, question, candidates)

    c.judge = judge
    for task in ("boxes", "sum"):
        state = next(row["state"] for row in rows.values() if row["task"] == task)
        for effort in ("low", "medium", "high", "auto"):
            calls.clear()
            answer = auto_solve(b, c, state, effort)
            assert calls == ["choice", "info"]
            api.append({"task": task, "state": state, "effort": effort, "status": answer["status"], "b_loops": answer["b_loops"], "max_loops": answer["max_loops"], "c_calls": list(calls), "answer": answer["solution"]["answer"], "probability_sum_choice": sum(answer["choice"]["probabilities"].values())})
    calls.clear()
    answer = auto_solve(b, c, state, "medium", max_loops=2)
    assert calls == ["choice", "info"] and answer["status"] == "budget_limited" and answer["b_loops"] == 2
    api.append({"task": "sum", "effort": "medium", "max_loops": 2, "requested_loops": answer["requested_loops"], "actual_loops": answer["b_loops"], "status": answer["status"], "c_calls": list(calls)})
    try:
        b.solve("請幫我聊天")
    except ValueError as exc:
        result["outside_grammar_rejection"] = str(exc)
    else:
        raise AssertionError("B accepted unconstrained chat")
    result["bounded_cpu_api_requests"] = api
    old = read_json("docs/selftrained/results/v2-final-public-results.json")
    result["historical_A_primary_scope"] = {"scope": old["scope"], "hf_revision": old["hf_revision"], "architectures": {name: {"training_revision": value["training_revision"], "vision_clothing": value["final_test"]["per_task_final_reply"]["vision_clothing"]["exact"], "voice_qa": value["final_test"]["per_task_final_reply"]["voice_qa"]["exact"]} for name, value in old["architectures"].items()}, "inspection_limit": "Only this original result receipt was read for the limited image/audio capability statement; no historical production inference rerun."}
    result["result"] = "All assertions passed. Four table rows and all denominators reproduced from the existing primary score records; historical timing values and boundaries checked; 9 bounded CPU API requests confirmed two C calls and capped loops."
    target = Path(__file__).with_name("19.18-backfill-original-01-execution.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": result["result"], "quality": quality, "auto_loop_reduction_percent": result["auto_loop_reduction_percent"], "output": str(target.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
