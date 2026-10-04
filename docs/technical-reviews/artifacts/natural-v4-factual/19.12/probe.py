"""Independent bounded CPU validation of 19.12; no training or model downloads."""
import copy
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.capstone import CapstoneModel, build_dataset, default_config, prompt_ids
from scripts.course_experiments.capstone_deployment import image_counterfactuals

torch.set_num_threads(2)
inputs = {}
def read(path):
    p = ROOT / path
    data = p.read_bytes()
    inputs[path] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return json.loads(data)

def decoded(trace):
    ids = trace["generated_ids"]
    assert all(type(i) is int and 0 <= i < 264 for i in ids)
    specials = [i for i in ids if i < 8]
    assert not specials or (len(specials) == 1 and ids[-1] == specials[0])
    raw = bytes(i - 8 for i in ids if i >= 8).decode("utf-8", errors="replace")
    eos = bool(ids) and ids[-1] == 2
    assert trace["raw"] == raw and trace["eos"] == eos
    assert (trace["stop_reason"] == "eos") == eos
    return raw, eos

def action(raw, eos):
    if not eos:
        return {"status": "invalid", "reason": "unterminated_generation"}
    if raw.startswith("DIRECT:") and len(raw) > 7:
        return {"status": "direct", "content": raw[7:]}
    if raw.startswith("ASK:") and len(raw) > 4:
        return {"status": "ask", "content": raw[4:]}
    m = re.fullmatch(r"TOOL:([a-z_]+):([0-9]{1,3})\+([0-9]{1,3})", raw)
    if m:
        return {"status": "tool", "name": m[1], "a": int(m[2]), "b": int(m[3])}
    return {"status": "invalid", "reason": "malformed_action"}

def score(path, rows):
    report = read(path)
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(report["records"]) == report["count"]
    assert set(by_id) == {r["id"] for r in report["records"]}
    tasks = defaultdict(lambda: {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
    derived, errors = {}, []
    for record in report["records"]:
        row = by_id[record["id"]]
        assert record["task"] == row["task"] and record["expected_action"] == row["answer"]
        raw, eos = decoded(record["action_trace"])
        assert record["action_trace"]["prompt_ids"] == prompt_ids(row)
        parsed = action(raw, eos)
        assert parsed == record["parsed_action"]
        ac = eos and raw == row["answer"]
        gold = row["answer"].split(":", 1)[1]
        if row["answer"].startswith("TOOL:"):
            operands = gold.split(":")[1].split("+")
            gold = str(sum(map(int, operands)))
        assert gold == record["expected_final"]
        answer = None
        if parsed["status"] in ("direct", "ask"):
            answer = parsed["content"]
            assert record["runtime"] is None and record["final_trace"] is None
        if parsed["status"] == "tool":
            expected_runtime = (
                {"status": "error", "reason": "calculator_unavailable"} if not row["available"] else
                {"status": "error", "reason": "tool_not_allowlisted"} if parsed["name"] != "calculator" else
                {"status": "ok", "result": str(parsed["a"] + parsed["b"])}
            )
            assert record["runtime"] == expected_runtime
            if expected_runtime["status"] == "ok":
                final = action(*decoded(record["final_trace"]))
                if final["status"] == "direct":
                    answer = final["content"]
                followup = dict(row, image=None, audio=None)
                followup["user"] = f"原題：{parsed['a']}+{parsed['b']}。計算器回報：{expected_runtime['result']}。請回答。"
                assert record["final_trace"]["prompt_ids"] == prompt_ids(followup)
        assert answer == record["answer"]
        e2e = ac and answer == gold
        assert ac == record["action_correct"] and e2e == record["end_to_end_correct"]
        tasks[row["task"]]["count"] += 1
        tasks[row["task"]]["action_correct"] += int(ac)
        tasks[row["task"]]["end_to_end_correct"] += int(e2e)
        derived[row["id"]] = {"action": raw, "answer": answer, "correct": e2e}
        if not e2e:
            errors.append({"id": row["id"], "task": row["task"], "user": row["user"], "expected_action": row["answer"], "generated_action": raw, "runtime": record["runtime"], "answer": answer, "expected_final": gold})
    tasks = dict(tasks)
    assert tasks == report["by_task"]
    summary = {"count": len(rows), "action_correct": sum(t["action_correct"] for t in tasks.values()), "end_to_end_correct": sum(t["end_to_end_correct"] for t in tasks.values()), "by_task": tasks}
    assert summary["action_correct"] == report["action_correct"] and summary["end_to_end_correct"] == report["end_to_end_correct"]
    return report, summary, derived, errors

splits, manifest = build_dataset()
data = read("docs/course-experiments/capstone-evidence/deployment/data.json")
assert splits == data["splits"] and manifest == data["manifest"]
counts = dict(sorted(Counter(row["task"] for row in splits["test"]).items()))
assert sum(counts.values()) == 90
families = {k: {r["family"] for r in rows} for k, rows in splits.items()}
assert not (families["train"] & families["validation"] or families["train"] & families["test"] or families["validation"] & families["test"])

base = "docs/course-experiments/capstone-evidence/"
names = ["untrained", "pretrain", "sft", "joint", "dpo", "joint-ptq4", "joint-ptq8", "ptq4", "ptq8"]
paths = {name: base + "deployment/test-" + name + ".json" for name in names}
paths.update({"ce": base + "student/test-ce.json", "kd": base + "student/test-kd.json", "kd-ptq4": base + "student/test-kd-ptq4.json"})
reports, summaries, derived, failures = {}, {}, {}, {}
for name, path in paths.items():
    reports[name], summaries[name], derived[name], failures[name] = score(path, splits["test"])
expected = [0, 0, 45, 78, 78, 78, 78, 78, 78, 62, 61, 61]
assert [summaries[k]["end_to_end_correct"] for k in paths] == expected
assert summaries["joint"]["action_correct"] == 80

joint = reports["joint"]["records"]
nonrefusal = [r for r in joint if r["task"] != "safety"]
assert len(nonrefusal) == 87 and sum(r["end_to_end_correct"] for r in nonrefusal) == 75
assert all("不能提供他人密碼" not in decoded(r["action_trace"])[0] for r in nonrefusal)
joint_specials = {task: dict(Counter(row["answer"] for row in splits["test"] if row["task"] == task)) for task in ["concept", "image_color", "image_shape", "audio", "rag", "missing", "safety", "style"]}

swapped_rows = image_counterfactuals(splits["test"])
swaps, image_summary, image_derived, _ = score(base + "deployment/test-joint-image-swaps.json", swapped_rows)
pairs = read(base + "deployment/test-joint-image-pairs.json")
pair_tasks = defaultdict(lambda: {"count": 0, "both_correct": 0})
for p in pairs["pairs"]:
    orig = derived["joint"][p["original_id"]]
    new = image_derived[p["swapped_id"]]
    assert p["original_generated"] == orig["action"] and p["swapped_generated"] == new["action"]
    both = orig["correct"] and new["correct"]
    assert both == p["both_end_to_end_correct"]
    task = next(r["task"] for r in splits["test"] if r["id"] == p["original_id"])
    pair_tasks[task]["count"] += 1
    pair_tasks[task]["both_correct"] += int(both)
assert sum(v["both_correct"] for v in pair_tasks.values()) == pairs["both_end_to_end_correct"] == 27
audio_rows = []
for row in splits["test"]:
    if row["task"] == "joint":
        new = copy.deepcopy(row)
        new["audio"]["pitch"] = "low" if row["audio"]["pitch"] == "high" else "high"
        new["answer"] = f"DIRECT:{row['image']['color']},{new['audio']['pitch']}"
        new["id"] += "-audio-swap"
        audio_rows.append(new)
_, audio_summary, audio_derived, _ = score(base + "deployment/test-joint-audio-swaps.json", audio_rows)
assert all(derived["joint"][r["id"].removesuffix("-audio-swap")]["correct"] and audio_derived[r["id"]]["correct"] for r in audio_rows)

validation = {}
for name, folder in [("joint", "joint"), ("dpo", "dpo")]:
    _, validation[name], _, _ = score(base + folder + "/validation.json", splits["validation"])
selection = read("docs/course-experiments/capstone-selection.json")
assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"]
assert validation["joint"]["end_to_end_correct"] == 75 and validation["dpo"]["end_to_end_correct"] == 71
for name in validation:
    assert selection["candidates"][name]["validation_end_to_end_correct"] == validation[name]["end_to_end_correct"]

changes = {}
for fp, quant in [("joint", "joint-ptq4"), ("joint", "joint-ptq8"), ("dpo", "ptq4"), ("dpo", "ptq8"), ("kd", "kd-ptq4")]:
    ids = []
    for a, b in zip(reports[fp]["records"], reports[quant]["records"], strict=True):
        assert a["id"] == b["id"]
        if any((a[field] or {}).get("generated_ids") != (b[field] or {}).get("generated_ids") for field in ["action_trace", "final_trace"]):
            ids.append(a["id"])
    changes[fp + "->" + quant] = {"count": len(ids), "ids": ids, "all_previously_wrong": all(not derived[fp][id]["correct"] for id in ids)}
assert [v["count"] for v in changes.values()] == [1, 0, 2, 0, 6]
ce_extra = [id for id in derived["ce"] if derived["ce"][id]["correct"] and not derived["kd"][id]["correct"]]
assert len(ce_extra) == 1
ce_kd_example = {name: next(r for r in failures.get(name, []) if r["id"] == ce_extra[0]) if name == "kd" else derived[name][ce_extra[0]] for name in ["ce", "kd"]}

benchmark = read(base + "deployment/generation-benchmark.json")
timings = {}
for name, mode in benchmark["modes"].items():
    measured = [r for r in mode["iterations"] if r["phase"] == "measured"]
    assert len(measured) == 10 and len(mode["iterations"]) == 13
    times = [r["seconds"] for r in measured]
    assert times == mode["seconds"] and statistics.median(times) == mode["median_seconds"]
    for r in mode["iterations"]:
        assert decoded(r["trace"]) == ("DIRECT:15", True) and len(r["trace"]["generated_ids"]) == 10
    timings[name] = {"median_ms": statistics.median(times) * 1000, "rounded_ms": round(statistics.median(times) * 1000, 3), "warmup": 3, "measured": 10}
assert [timings[k]["rounded_ms"] for k in ["full", "cache"]] == [68.054, 59.978]
cache = read(base + "deployment/cache-consistency.json")
first = {}
for row in splits["validation"]:
    first.setdefault(row["task"], row)
assert [r["id"] for r in cache["records"]] == [first[task]["id"] for task in sorted(first)]
assert cache["count"] == 12
for r in cache["records"]:
    decoded(r["full_trace"]); decoded(r["cached_trace"])
    assert r["full_trace"]["generated_ids"] == r["cached_trace"]["generated_ids"]
    assert r["full_trace"]["stop_reason"] == r["cached_trace"]["stop_reason"]

torch.manual_seed(42)
moe = CapstoneModel()
student = CapstoneModel(replace(default_config(dense=True), width=48))
parameters = {"moe": sum(p.numel() for p in moe.parameters()), "student": sum(p.numel() for p in student.parameters())}
assert parameters == {"moe": 328128, "student": 79920}
a = torch.zeros(1, 3, 16, 16)
b = a.clone(); a[0, 0, 0, 0] = 1; b[0, 0, 0, 1] = 1
pool_equal = torch.equal(torch.nn.functional.adaptive_avg_pool2d(a, (4, 4)), torch.nn.functional.adaptive_avg_pool2d(b, (4, 4)))
assert pool_equal and not torch.equal(a, b)
pretrain = read(base + "pretrain/train-report.json")
assert pretrain["steps"] == pretrain["requested_steps"] == 300 and pretrain["schedule_completed"]
student_report = read("docs/course-experiments/results/capstone_student.json")
dpo_report = read("docs/course-experiments/results/capstone_preference.json")
assert student_report["results"]["teacher_checkpoint_sha256"] == selection["candidates"]["dpo"]["checkpoint_sha256"]
for mode in ["ce", "kd"]:
    train = read(base + "student/" + mode + "/train-report.json")
    assert train["steps"] == 350 and train["parameters"]["parameters"] == 79920
deployment = read("docs/course-experiments/results/capstone_deployment.json")
assert deployment["gpu"] == "NVIDIA L4" and deployment["seed"] == 42
natural_selection = read("docs/natural-assistant/v4/selection.json")
asr_selection = read("docs/natural-assistant/v4/asr-selection.json")
assert natural_selection["selected_variant"] == "base" and asr_selection["selected_variant"] == "turbo"

result = {"environment": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu", "cuda_available": str(torch.cuda.is_available())}, "limits": "Actual CPU arithmetic/current dataset/current architecture and independent rescoring of fixed GPU records; no own GPU inference/training/timing replication. Cache logits lack full tensors, so only recorded allclose is inspected, not independently recalculated.", "inputs": inputs, "counts": counts, "manifest_sha": manifest["sha256"], "record_rescoring": summaries, "joint_failures": failures["joint"], "labels": joint_specials, "nonrefusal": {"count": 87, "correct": 75, "refusal_sentence_occurrences": 0}, "image_pairs": dict(pair_tasks), "audio_pairs": {"count": 18, "both_correct": 18}, "validation": validation, "id_changes": changes, "ce_kd_difference": ce_kd_example, "timing": timings, "cache_raw_ids_equal": {"count": 12, "all_equal": True}, "parameter_count": parameters, "pooling_collision": {"different_pixels_same_4x4_summary": pool_equal}, "mean_derivation": {"micro_e2e": 78 / 90, "equal_task_e2e": sum(t["end_to_end_correct"] / t["count"] for t in summaries["joint"]["by_task"].values()) / 12}, "selection": {"capstone": "joint, 75/84 versus DPO 71/84", "natural": natural_selection["selected_variant"], "asr": asr_selection["model"]}}
print(json.dumps(result, ensure_ascii=False, indent=2))
