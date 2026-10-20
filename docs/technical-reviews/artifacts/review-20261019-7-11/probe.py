"""核對對話目標、梯度與實驗配方；配方追蹤不執行訓練。"""

import contextlib
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from scripts.course_experiments import common, run, text
from scripts.prepare_data import generate_records
from tiny_perceptron.data import ByteTokenizer, pad_batch, render_chat, toy_conversations
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(42)
conversations = toy_conversations()
examples = [render_chat(messages) for messages in conversations[:2]]
x, y, valid = pad_batch(examples)
model = TinyLM(ModelConfig(width=8))
before = {name: value.detach().clone() for name, value in model.named_parameters()}
logits = model(x, valid=valid)["logits"]
logits.retain_grad()
loss = masked_loss(logits, y)
selected = y != -100
manual_loss = -logits.log_softmax(-1)[selected, y[selected]].mean()
loss.backward()
parameters = dict(model.named_parameters())
original = conversations[0]
messages = [message.copy() for message in original]
messages[1]["content"] = "3"
exercise_x, exercise_y = render_chat(messages)
padded_x, padded_y, padded_valid = pad_batch([examples[0], (examples[1][0][:-1], examples[1][1][:-1])])
demo = {
    "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu"},
    "conversation_count": len(conversations),
    "ordered_questions": [row[0]["content"] for row in conversations],
    "first_answers": [row[1]["content"] for row in conversations[:2]],
    "x": x.tolist(), "y": y.tolist(), "valid": valid.tolist(),
    "shape": list(x.shape), "logits_shape": list(logits.shape),
    "dtype": str(x.dtype), "target_count": int(selected.sum()),
    "loss": float(loss.detach()), "explicit_answer_mean_nll": float(manual_loss.detach()),
    "ignored_logits_gradient_max": float(logits.grad[~selected].abs().max()),
    "answer_logits_gradient_nonzero": bool((logits.grad[selected] != 0).any()),
    "all_parameter_gradients_finite": all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters.values()),
    "parameters_with_nonzero_gradient": sum(bool((p.grad != 0).any()) for p in parameters.values()),
    "parameters_changed_by_backward": [name for name, p in parameters.items() if not torch.equal(p, before[name])],
    "copied_dictionaries_are_distinct": all(a is not b for a, b in zip(messages, original)),
    "original_answer_after_copy_mutation": original[1]["content"],
    "exercise_targets": exercise_y[exercise_y != -100].tolist(),
    "original_targets": examples[0][1][examples[0][1] != -100].tolist(),
    "exercise_input_prefix_unchanged": torch.equal(exercise_x[:-1], examples[0][0][:-1]),
    "tokenizer": {"vocab_size": ByteTokenizer.vocab_size, "zero_byte": ord("0"), "three_byte": ord("3"), "zero_ids": ByteTokenizer().encode("0"), "three_ids": ByteTokenizer().encode("3"), "eos": ByteTokenizer.eos_id},
    "padding_probe": {"x": padded_x.tolist(), "y": padded_y.tolist(), "valid": padded_valid.tolist()},
}
assert demo["shape"] == [2, 10] and demo["logits_shape"] == [2, 10, 264]
assert demo["target_count"] == 4 and demo["conversation_count"] == 16
assert demo["ignored_logits_gradient_max"] == 0 and demo["answer_logits_gradient_nonzero"]
assert demo["all_parameter_gradients_finite"] and demo["parameters_with_nonzero_gradient"] > 0
assert demo["parameters_changed_by_backward"] == []
assert demo["exercise_targets"] == [59, 2] and demo["original_targets"] == [56, 2]
assert demo["original_answer_after_copy_mutation"] == "0" and demo["exercise_input_prefix_unchanged"]
assert torch.allclose(loss, manual_loss, atol=1e-6, rtol=0)
assert not padded_valid[1, -1] and padded_y[1, -1] == -100 and padded_x[1, -1] == 0

records = generate_records("attributes-sft")
parts = common.split_records(records, seed=42)
groups = {key: set(row["family"] for row in rows) for key, rows in parts.items()}
split_manifest = {}
for key, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    split_manifest[key] = {"records": len(rows), "families": len(groups[key]), "sha256": digest(raw)}
assert not (groups["train"] & groups["test"] or groups["train"] & groups["validation"] or groups["test"] & groups["validation"])
training_calls = []
initial_states = []
real_new_lm = common.new_lm


def tracked_new_lm(ctx, **kwargs):
    created = real_new_lm(ctx, **kwargs)
    initial_states.append({name: value.detach().clone() for name, value in created.state_dict().items()})
    return created


def tracked_fit(model, rows, ctx, **kwargs):
    defaults = inspect.signature(common.fit_lm).parameters
    training_calls.append({
        "name": kwargs["name"], "mode": kwargs["mode"], "steps": kwargs["steps"],
        "records": len(rows), "records_sha256": common.records_sha256(rows),
        "batch_size": kwargs.get("batch_size", defaults["batch_size"].default),
        "lr": kwargs.get("lr", defaults["lr"].default),
        "width": model.config.width, "layers": model.config.layers, "seed": ctx.seed,
        "first_record": rows[0],
    })
    return {"steps": kwargs["steps"], "records": len(rows)}


ctx = SimpleNamespace(seed=42, device="cpu", output=Path(__file__).parent, step_scale=1.0)
with patch.object(text, "new_lm", tracked_new_lm), patch.object(text, "fit_lm", tracked_fit), patch.object(text, "_evaluations", return_value={}), patch.object(text, "_save_splits", return_value=split_manifest), patch.object(text, "write_json"), patch.object(text, "save_checkpoint"), patch.object(text, "_natural_chat_pilot", return_value={"scope": "追蹤時略過外部資料與訓練"}):
    text.run_sft(ctx)
same_initial = all(torch.equal(initial_states[0][name], initial_states[1][name]) for name in initial_states[0])
assert same_initial and [call["steps"] for call in training_calls] == [900, 250, 900]
assert [call["mode"] for call in training_calls] == ["sft", "text", "sft"]
assert all(call["records"] == 45 and call["width"] == 64 and call["layers"] == 2 and call["batch_size"] == 16 and call["lr"] == 0.003 for call in training_calls)
assert training_calls[0]["records_sha256"] == training_calls[2]["records_sha256"]

record_path = ROOT / "docs/course-experiments/results/sft.json"
record = json.loads(record_path.read_bytes())
results = record["results"]
for split, calculated in split_manifest.items():
    assert all(results["data"][split][key] == value for key, value in calculated.items())
for observed, saved in zip(training_calls, [results["training"], results["pretrain_then_sft"]["pretraining"], results["pretrain_then_sft"]["sft"]]):
    assert all(observed[key] == saved[key] for key in ("steps", "records", "records_sha256"))

cli_calls = []


def fake_execute(*args, **kwargs):
    cli_calls.append({"args": [str(value) for value in args], "kwargs": kwargs})
    return {"experiment_id": "sft", "elapsed_seconds": 0.0, "evidence_status": "追蹤時未訓練", "artifacts": []}


with patch.object(sys, "argv", ["run", "--experiment", "sft", "--device", "cuda"]), patch.object(run, "execute", fake_execute), contextlib.redirect_stdout(io.StringIO()):
    run.main()
assets = run.list_assets("sft")
spec = run.experiment_spec("sft")
fetch_list = subprocess.run([str(ROOT / ".venv/bin/python"), "scripts/fetch_training_assets.py", "--list"], cwd=ROOT, text=True, capture_output=True, check=True)
asset_rows = [json.loads(row) for row in fetch_list.stdout.splitlines()]
selected_asset = next(row for row in asset_rows if row["id"] == "ultrachat-sft")
source_paths = ["tiny_perceptron/data.py", "tiny_perceptron/model.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/run.py", "scripts/prepare_data.py", "scripts/fetch_training_assets.py", "docs/course-experiments/plan.json", "assets/training/manifest.json"]
code_versions = {name: {"sha256": digest((ROOT / name).read_bytes()), "recorded_sha256": record["code_sha256"].get(name)} for name in source_paths}
historical = subprocess.run(["git", "show", record["revision"] + ":scripts/course_experiments/common.py"], cwd=ROOT, text=True, capture_output=True)
out = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-11/probe.py",
    "demo": demo,
    "recipe_trace": {"scope": "實際建立模型與資料；攔截訓練、評估、存檔與 UltraChat pilot，不能當成訓練重現", "same_initial_parameters": same_initial, "training_calls": training_calls},
    "split_manifest": split_manifest,
    "example_attribute_question": next(row for row in records if row["messages"][0]["content"] == "color=green;shape=square;pitch=low;shape?"),
    "record_receipt": {
        "path": str(record_path.relative_to(ROOT)), "sha256": digest(record_path.read_bytes()),
        "run": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
        "elapsed_seconds_rounded": round(record["elapsed_seconds"], 1),
        "training": [{key: entry[key] for key in ("steps", "records", "records_sha256", "effective_tokens", "parameters", "seconds")} for entry in [results["training"], results["pretrain_then_sft"]["pretraining"], results["pretrain_then_sft"]["sft"]]],
        "historical_source_lookup": {"command": "git show " + record["revision"] + ":scripts/course_experiments/common.py", "returncode": historical.returncode, "stderr": historical.stderr, "scope": "紀錄 revision 的該路徑無法重建；本次以逐檔 code_sha256 核對配方"},
    },
    "code_versions": code_versions,
    "cli_trace": {"calls": cli_calls, "spec": spec, "assets": assets, "fetch_list_command": ".venv/bin/python scripts/fetch_training_assets.py --list", "fetch_list_returncode": fetch_list.returncode, "selected_asset": selected_asset, "scope": "驗證參數解析、派送、資料依賴與預設輸出位置；沒有下載或啟動 CUDA 訓練"},
}
print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
