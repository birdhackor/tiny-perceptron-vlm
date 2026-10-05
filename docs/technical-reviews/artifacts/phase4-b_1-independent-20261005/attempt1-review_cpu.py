"""Independent, bounded CPU audit of B.1; no training/model sampling."""
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from scripts.course_experiments.applications import _parse_json_action, _prompt_ids, _tool_records
from scripts.course_experiments.common import records_sha256, split_records
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.retrieval import call_tool

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (ART / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


raw_dir = ART / "original-data"
code_dir = ART / "code"
raw_dir.mkdir(exist_ok=True)
code_dir.mkdir(exist_ok=True)
original_result = ROOT / "docs/course-experiments/results/tools.json"
original_dataset = ROOT / "outputs/application-smoke/tools/dataset.json"
result_copy = raw_dir / "tools.json"
dataset_copy = raw_dir / "dataset.json"
shutil.copyfile(original_result, result_copy)
shutil.copyfile(original_dataset, dataset_copy)
assert sha(original_result) == sha(result_copy)
assert sha(original_dataset) == sha(dataset_copy)
result = json.loads(result_copy.read_bytes())
dataset_record = next(x for x in result["artifacts"] if x["path"] == "dataset.json")
assert sha(dataset_copy) == dataset_record["sha256"]

code_paths = [
    "scripts/course_experiments/applications.py", "scripts/course_experiments/common.py",
    "scripts/course_experiments/run.py", "tiny_perceptron/retrieval.py",
    "tiny_perceptron/data.py", "tiny_perceptron/model.py",
]
code_hashes = {}
for relative in code_paths:
    original = ROOT / relative
    target = code_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, target)
    assert sha(original) == sha(target) == result["code_sha256"][relative]
    code_hashes[relative] = sha(target)

environment = {"python": platform.python_version(), "torch": str(torch.__version__),
               "device": "CPU; CUDA_VISIBLE_DEVICES empty", "threads": str(torch.get_num_threads()),
               "platform": platform.platform(), "executable": sys.executable}
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
save("environment.json", environment)
save("provenance.json", {
    "raw_result_original": str(original_result.relative_to(ROOT)),
    "raw_result_copy": str(result_copy.relative_to(ROOT)), "raw_result_sha256": sha(result_copy),
    "raw_dataset_original": str(original_dataset.relative_to(ROOT)),
    "raw_dataset_copy": str(dataset_copy.relative_to(ROOT)), "raw_dataset_sha256": sha(dataset_copy),
    "dataset_recorded_run_hash_matches": True,
    "dataset_origin_scope": "Existing bounded local smoke-run dataset; exact bytes match original L4 run artifact SHA. No training data download.",
    "original_run": {k: result[k] for k in ["revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "unfinished_schedules"]},
    "modal": {k: result["modal"][k] for k in ["run_id", "batch_id"]},
    "hf": {k: result["hf"][k] for k in ["repo", "revision", "prefix", "result_revision"]},
    "code_sha256": code_hashes,
    "read_json_pointers": ["/revision", "/device", "/seed", "/torch_version", "/python_version", "/gpu",
        "/step_scale", "/unfinished_schedules", "/artifacts/*/{path,bytes,sha256}", "/code_sha256/<six necessary files>",
        "/modal/{run_id,batch_id}", "/hf/{repo,revision,prefix,result_revision}",
        "/results/training/{steps,planned_steps,step_scale,records,records_sha256}", "/results/split",
        "/results/model/config", "/results/samples/*/{family,operation,question,expected,max_steps,trace}",
        "/results/samples/*/trace/*/{step,generation,executed,parsed,tool_result,finite_result}",
        "/results/samples/*/trace/*/generation/{messages,input_ids,samples}",
        "/results/samples/*/trace/*/generation/samples/0/{generated,generated_ids,eos,invalid_special_tokens}",
        "dataset.json /train/*, /validation/*, /test/*: family, operation, a, b, answer, messages"],
    "avoided": "No author result summaries, extra notes, old reviews, TRAINING/DATA/worklog, checkpoints or weight loading.",
})

dataset = json.loads(dataset_copy.read_bytes())
print("dataset top-level keys/types:", [(k, type(v).__name__) for k, v in dataset.items()])
regenerated = split_records(_tool_records(), result["seed"])
assert regenerated == dataset
encoded = (json.dumps(regenerated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
assert hashlib.sha256(encoded).hexdigest() == dataset_record["sha256"]
family_sets = {k: {r["family"] for r in rows} for k, rows in dataset.items()}
assert not (family_sets["train"] & family_sets["validation"] or family_sets["train"] & family_sets["test"] or family_sets["validation"] & family_sets["test"])
all_rows = [r for rows in dataset.values() for r in rows]
assert len(all_rows) == 210
for row in all_rows:
    assert all(ord(c) < 128 for c in row["messages"][1]["content"])
    if row["operation"] == "copy":
        assert row["messages"][1]["content"] == f"COPY:{row['answer']}"
        assert json.loads(row["messages"][2]["content"]) == {"done": True, "answer": row["answer"]}
    else:
        assert 0 <= row["a"] <= 9 and 0 <= row["b"] <= 9
        assert row["family"] == f"pair:{min(row['a'],row['b'])}:{max(row['a'],row['b'])}"
        assert row["answer"] == call_tool(json.loads(row["messages"][2]["content"]))
        assert row["messages"][3]["content"] == f"TOOL_RESULT:{row['answer']}"
for name, rows in dataset.items():
    assert len(rows) == result["results"]["split"][name]["records"]
    assert records_sha256(rows) == result["results"]["split"][name]["sha256"]
    assert len(family_sets[name]) == result["results"]["split"][name]["families"]
assert records_sha256(dataset["train"]) == result["results"]["training"]["records_sha256"]

tok = ByteTokenizer()
episodes = result["results"]["samples"]
assert len(episodes) == len(dataset["test"]) == 20
raw_audit = []
tools = copy_tasks = completed_correctly = 0
for row, episode in zip(dataset["test"], episodes, strict=True):
    assert episode["family"] == row["family"]
    assert episode["question"] == row["messages"][1]["content"]
    history = copy.deepcopy(row["messages"][:2])
    events = []
    executions = []
    final_answer = None
    for event in episode["trace"]:
        generation = event["generation"]
        assert generation["messages"] == history
        assert generation["input_ids"] == _prompt_ids(history, tok)
        sample = generation["samples"][0]
        ids = sample["generated_ids"]
        decoded = tok.decode(ids[:-1] if ids and ids[-1] == tok.eos_id else ids)
        assert decoded == sample["generated"]
        assert bool(ids and ids[-1] == tok.eos_id) == sample["eos"]
        entry = {"step": event["step"], "generated": decoded, "executed_recorded": event["executed"]}
        try:
            action = _parse_json_action(decoded)
        except ValueError as error:
            assert event["executed"] is False
            entry["independent_parse_error"] = type(error).__name__
            events.append(entry)
            break
        assert action == event["parsed"]
        if "done" in action:
            assert event["executed"] is False
            final_answer = action["answer"]
            entry["completion_answer"] = final_answer
            events.append(entry)
            break
        recalculated = call_tool(action)
        assert event["executed"] is True
        assert recalculated == event["tool_result"]
        assert action == {"name": row["operation"], "arguments": {"a": row["a"], "b": row["b"]}}
        entry["independent_tool_result"] = recalculated
        executions.append(recalculated)
        history += [{"role": "assistant", "content": decoded}, {"role": "user", "content": f"TOOL_RESULT:{recalculated}"}]
        events.append(entry)
    tools += len(executions)
    if row["operation"] == "copy":
        copy_tasks += 1
        assert not executions
    completed_correctly += int(final_answer == row["answer"] and (not executions if row["operation"] == "copy" else executions and final_answer == executions[-1]))
    raw_audit.append({"question": episode["question"], "family": row["family"], "events": events})
assert tools == 18 and copy_tasks == 2
assert completed_correctly == 19

fence = ROOT / "outputs/course-revision-20261005/phase4-b_1-factual-original/fence-1.py"
out = io.StringIO()
with contextlib.redirect_stdout(out):
    exec(compile(fence.read_bytes(), str(fence), "exec"), {})
assert '"name": "multiply"' in out.getvalue() and '尚未執行乘法' in out.getvalue()
request = {"name": "multiply", "arguments": {"a": 123, "b": 45}}
assert json.loads(json.dumps(request, ensure_ascii=False)) == request
assert call_tool(request) == 5535
wrong_request = {"name": "multiply", "arguments": {"a": 123, "b": 54}}
assert call_tool(wrong_request) == 6642 != 5535
for bad in [{"name": "delete_all", "arguments": {"a": 123, "b": 45}}, {"name": "multiply", "arguments": {"a": True, "b": 45}}, {"name": "multiply", "arguments": {"a": "123", "b": 45}}]:
    try:
        call_tool(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid request accepted")
example = {"user": "123乘45", "assistant_tool_request": request}
assert example["user"] not in {r["messages"][1]["content"] for r in all_rows}
assert not any(r["operation"] != "copy" and (r["a"] == 123 or r["b"] == 45) for r in all_rows)
# Verify SFT supervision targets the two assistant outputs, not user/result turns.
x, labels = render_chat(dataset["train"][0]["messages"], tok)
supervised = tok.decode(labels[labels != -100].tolist())
assistant_text = "".join(m["content"] for m in dataset["train"][0]["messages"] if m["role"] == "assistant")
assert supervised == assistant_text

help_run = subprocess.run([sys.executable, "scripts/course_experiments/run.py", "--help"], cwd=ROOT, text=True, capture_output=True, timeout=20)
assert help_run.returncode == 0 and "--device {cpu,cuda}" in help_run.stdout
assets_run = subprocess.run([sys.executable, "scripts/course_experiments/run.py", "--list-assets", "tools"], cwd=ROOT, text=True, capture_output=True, timeout=20)
assert assets_run.returncode == 0 and not assets_run.stdout.strip()
(ART / "recipe-help.txt").write_text(help_run.stdout, encoding="utf-8")
(ART / "original-fence-repeated-stdout.txt").write_text(out.getvalue(), encoding="utf-8")
save("raw-episode-audit.json", raw_audit)
summary = {
    "original_fence": "executed unchanged; serialization only, no training/call_tool",
    "mutation_results": {"123_times_45": 5535, "123_times_54": 6642, "unknown_bool_string_requests": "all rejected"},
    "dataset": {k: {"records": len(v), "families": len(family_sets[k]), "sha256": records_sha256(v)} for k, v in dataset.items()},
    "regenerated_dataset_file_sha256": hashlib.sha256(encoded).hexdigest(),
    "original_samples_recomputed": {"episodes": len(episodes), "arithmetic_tool_calls": tools, "copy_tasks_without_calls": copy_tasks, "correct_completed": completed_correctly},
    "sft_supervised_text_matches_assistant_turns": True,
    "cli": {"help_exit": help_run.returncode, "list_assets_exit": assets_run.returncode, "tools_asset_count": 0},
    "coverage": "json dumps/loads round-trip; original unchanged fence; call_tool correct and wrong numeric arguments plus name/type rejection; _tool_records and split_records entire bounded dataset; records_sha256 full split hashes; _parse_json_action and _prompt_ids raw 20 episodes; ByteTokenizer decode exact stored output IDs; render_chat assistant targets; CLI help/list-assets only. No model training, new generation, or weight loading.",
}
save("cpu-results.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
