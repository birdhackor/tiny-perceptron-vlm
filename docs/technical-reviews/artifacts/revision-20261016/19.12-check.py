"""19.12 independent audit: retained evidence and deterministic examples, no training."""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices
from scripts.selftrained import augment_ocr_v2, evaluate, prepare_text_tools, prepare_vision_ocr

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/technical-reviews/artifacts/revision-20261016/19.12-execution.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT / relative).read_text())


body = lesson_slices((ROOT / "course/chapters/19.md").read_bytes().decode())["19.12"]
page = next(x for x in read("docs/course-revision-20261016/freeze-08.json")["pages"] if x["page_id"] == "19.12")
assert hashlib.sha256(body.encode()).hexdigest() == page["source_sha256"]
assert page["figures_sha256"] == {}
result = {
    "reviewer_task": "/root/tech_20261016_19_12",
    "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.12-check.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU; metadata and deterministic helper checks only", "platform": platform.platform()},
    "scope": "Read original retained run metadata; did not rerun training, model inference, or historical final evaluation.",
    "source_sha256": page["source_sha256"],
    "original_archive_hash_checks": [],
    "architectures": {},
}
index = read("docs/selftrained/results/public-raw/index.json")
for item in index["files"]:
    actual = sha(ROOT / item["destination_path"])
    assert actual == item["sha256"]
    result["original_archive_hash_checks"].append({"path": item["destination_path"], "sha256": actual})

table = {
    "text": (464, 429, 560), "ocr": (252, 281, 324),
    "vision_clothing": (357, 356, 360), "vision_relation": (1427, 1426, 1440),
    "voice_qa": (42, 37, 90), "voice_topic_continuation": (26, 24, 60),
    "tool_call": (0, 7, 276), "tool_reply": (551, 468, 552),
    "tool_concept": (6, 3, 8), "tool_missing": (5, 8, 8),
    "tool_unavailable": (29, 24, 48), "tool_unsupported": (4, 4, 8),
}
controls = {"swap": (720, 707, 706, 715, 714), "roi": (150, 86, 110, 150, 150),
            "history_topic": (40, 31, 33, 40, 40), "history_format": (140, 103, 113, 138, 140),
            "history_choice": (80, 63, 0, 66, 69)}
selection = read("docs/selftrained/results/v2-final-training-source-selection.json")
assert selection["test_used_for_selection"] is False
public = read("docs/selftrained/results/v2-final-public-results.json")
for number, arch in enumerate(("moe", "dense")):
    prefix = f"docs/selftrained/results/public-raw/{arch}"
    metrics = read(prefix + "/test/metrics.json")
    protocol = read(prefix + "/freeze/frozen.json")
    receipt = read(prefix + "/test/receipt.json")
    journal = read(prefix + "/test/evaluation-receipt.json")
    stage = read(f"docs/selftrained/results/v2-{arch}-public-final-test-stage-index.json")
    assert metrics["count"] == metrics["expected_count"] == journal["completed_count"] == 3734
    assert metrics["evaluation_complete"] and not metrics["limited_smoke"]
    assert metrics["resumed_completed_count"] == 0 and journal["resumed_from"] is None
    assert metrics["teacher_forcing_used_for_generation"] is False
    assert protocol["test_once"] is True and protocol["selection"] == "validation only"
    assert selection["decided_at"] < protocol["created_at"] < receipt["started_at"]
    assert metrics["thresholds"] == protocol["thresholds"] == stage["original_thresholds"]
    assert metrics["weight_source"] == "safe_export"
    assert metrics["safe_weights_sha256"] == journal["checkpoint_sha256"] == protocol["safe_weights_sha256"]
    assert public["architectures"][arch]["selected_source"] == receipt["job"]["checkpoint"]
    assert sha(ROOT / (prefix + "/freeze/frozen.json")) == journal["protocol_sha256"] == receipt["job"]["protocol"]["sha256"]
    observed = {}
    for task, expected in table.items():
        v = metrics["per_task_final_reply"][task]
        assert (v["semantic"]["numerator"], v["semantic"]["denominator"], v["count"]) == (expected[number], expected[2], expected[2])
        observed[task] = {"correct": v["semantic"]["numerator"], "questions": v["count"], "source_groups": v["source_groups"], "recording_assets": v["recording_assets"], "image_assets": v["image_assets"]}
        assert public["architectures"][arch]["final_test"]["per_task_final_reply"][task] == v
    assert sum(x["questions"] for x in observed.values()) == 3734
    for kind, expected in controls.items():
        v = metrics["controls"]["paired"][kind]
        assert (v["pairs"], v["all_correct"], v["output_changed_when_gold_changed"]) == (expected[0], expected[1 + number], expected[3 + number])
    assert metrics["perception"]["ocr"]["exact_numerator"] == metrics["perception"]["ocr"]["denominator"] == 324
    assert metrics["voice_topic_continuation"]["end_to_end_correct"] == table["voice_topic_continuation"][number]
    assert metrics["voice_topic_continuation"]["recording_assets"] == 30
    class_counts = {str(i): metrics["stratified_final_reply"][f"vision_clothing/query_class={i}"]["count"] for i in range(3)}
    assert list(class_counts.values()) == [120, 120, 120]
    assert stage["model_capability_acceptance"] is False
    assert metrics["per_task_final_reply"]["tool_call"]["semantic"]["rate"] < protocol["thresholds"]["tool_roundtrip"] == 0.95
    historical = subprocess.check_output(["git", "show", receipt["revision"] + ":scripts/selftrained/evaluate.py"], cwd=ROOT).decode()
    assert hashlib.sha256(historical.encode()).hexdigest() == protocol["code_sha256"]["scripts/selftrained/evaluate.py"]
    adjusted = historical.replace('str(path.relative_to(root))', 'path.relative_to(root).as_posix()').replace('journal.receipt_path.open("rb")', 'journal.receipt_path.open("r+b")')
    assert adjusted == (ROOT / "scripts/selftrained/evaluate.py").read_text()
    result["architectures"][arch] = {
        "run_id": receipt["run_id"], "revision": receipt["revision"], "frozen_at": protocol["created_at"], "started_at": receipt["started_at"],
        "safe_weights_sha256": metrics["safe_weights_sha256"], "count": metrics["count"], "per_task": observed,
        "controls": {k: metrics["controls"]["paired"][k] for k in controls}, "ocr_entrance": metrics["perception"]["ocr"],
        "class_counts": class_counts, "voice_continuation": metrics["voice_topic_continuation"], "thresholds": protocol["thresholds"],
        "capability_acceptance": stage["model_capability_acceptance"],
        "historical_grader_sha256": protocol["code_sha256"]["scripts/selftrained/evaluate.py"],
        "historical_grader_equivalence": "Only current path.as_posix and receipt open mode portability changes; all scoring, aggregation, history and protocol logic unchanged.",
    }

rows = [x for x in prepare_text_tools.text_records() if x["split"] == "test"]
groups = defaultdict(list)
for row in rows:
    s = row["supervision"]
    groups[s["control_type"], s["pair_id"]].append(row)
group_sizes = defaultdict(Counter)
for (kind, _), members in groups.items():
    group_sizes[kind][len(members)] += 1
assert dict(group_sizes) == {"history_format": {2: 140}, "history_choice": {2: 80}, "history_topic": {3: 40}}
result["generated_text_control_group_sizes"] = {kind: dict(counts) for kind, counts in group_sizes.items()}
assert len(prepare_vision_ocr.CHARACTERS) == 12
assert {augment_ocr_v2.style("fixed-example", i)["font_family"] for i in range(5)} == {"Sans", "Serif"}
voice = read("docs/selftrained/provenance/voice-quality-summary.json")
voice_counts = {intent: voice["classes"][intent]["split_recordings"]["test"] for intent in ["address", "app_error", "card_issues"]}
assert sum(voice_counts.values()) == 30
result["finite_materials"] = {"ocr_characters": prepare_vision_ocr.CHARACTERS, "v2_train_families": ["Sans", "Serif"], "voice_test_recordings_by_intent": voice_counts}
result["baseline_derivation"] = {"always_bag_correct": 120, "questions": 360, "fraction": 120 / 360, "note": "Derived from gold distribution; no model run."}

code = re.search(r"```python\n(.*?)```", body, re.S).group(1)
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(code, {})
assert stdout.getvalue() == "按題合計 0.6666666666666666 每類等權重 0.5\n"
result["lesson_python_execution"] = {"code": code, "stdout": stdout.getvalue(), "derivation": "(12+0)/(12+6)=12/18=2/3; (1+0)/2=1/2"}

record = {"task": "tool_call", "messages": [{"role": "assistant", "content": "結果是5。"}], "supervision": {"expected_call": {"name": "calculator", "arguments": {"op": "add", "a": 2, "b": 3}}}}
trace = {"tool_call": record["supervision"]["expected_call"], "tool_result": {"ok": True, "result": 5}, "executed": True}
good = evaluate.score_reply(record, "結果是5。", trace)
bad_parameters = evaluate.score_reply(record, "結果是5。", {**trace, "tool_call": {"name": "calculator", "arguments": {"op": "add", "a": 1, "b": 4}}})
not_executed = evaluate.score_reply(record, "結果是5。", {**trace, "executed": False})
assert good["tool_roundtrip"] and not bad_parameters["tool_roundtrip"] and not not_executed["tool_roundtrip"]
result["synthetic_grader_boundary"] = {"correct_chain": good["tool_roundtrip"], "wrong_request_even_same_result": bad_parameters["tool_roundtrip"], "not_executed": not_executed["tool_roundtrip"], "scope": "Deterministic helper test, no trained model accuracy claim."}
result["result"] = "All assertions passed. Historical aggregate values, denominators, versions and limitations match 19.12; lesson arithmetic executed exactly."
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(result["result"])
print("Saved", OUT.relative_to(ROOT))
