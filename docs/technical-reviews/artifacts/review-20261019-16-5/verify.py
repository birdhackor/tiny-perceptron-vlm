"""核對遮罩算例、文件邊界與已保存的實測數值；不執行訓練。"""

import ast
import copy
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _mechanism_examples, _packing_batch
from tiny_perceptron.attention import attention_mask, manual_attention
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat, shifted
from tiny_perceptron.model import ModelConfig, TinyLM, loss_sum


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


torch.set_num_threads(1)
torch.manual_seed(27)
position = torch.arange(4)
segments = torch.tensor([[0, 0, 1, 1]])
causal = attention_mask(position, position)
isolated = attention_mask(position, position, segments=segments)
changed_segments = attention_mask(position, position, segments=torch.tensor([[0, 0, 0, 1]]))
valid = torch.tensor([[True, True, True, False]])
padded_mask = attention_mask(position, position, valid=valid, segments=segments)
expected = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
assert isolated[0, 0].int().tolist() == expected
assert causal[0, 0, -1].int().tolist() == [1, 1, 1, 1]
assert changed_segments[0, 0].int().tolist() == [
    [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]
]
assert padded_mask[0, 0].int().tolist() == [
    [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]
]
assert not bool((isolated & ~causal).any())
assert not bool((padded_mask & ~isolated).any())
assert torch.equal(isolated, attention_mask(position, position, segments=torch.tensor([[12, 12, 3, 3]])))
batch_mask = attention_mask(position, position, segments=torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]]))
assert list(batch_mask.shape) == [2, 1, 4, 4]

q = torch.zeros(1, 1, 4, 2)
k = torch.zeros_like(q)
v = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
_, weights = manual_attention(q, k, v, isolated)
assert list(weights.shape) == [1, 1, 4, 4]
assert torch.equal(weights.masked_select(~isolated), torch.zeros(10))

config = ModelConfig(vocab_size=40, width=8, heads=2, layers=2, max_length=16)
model = TinyLM(config).eval()
ids = torch.tensor([[1, 14, 2, 1, 19, 2]])
positions = torch.tensor([[0, 1, 2, 0, 1, 2]])
document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
altered = ids.clone()
altered[:, :3] = (altered[:, :3] + 7) % config.vocab_size
with torch.no_grad():
    original = model(ids, positions=positions, segments=document_ids)["logits"]
    changed = model(altered, positions=positions, segments=document_ids)["logits"]
    causal_original = model(ids, positions=positions)["logits"]
    causal_changed = model(altered, positions=positions)["logits"]
isolated_error = float((original[:, 3:] - changed[:, 3:]).abs().max())
causal_error = float((causal_original[:, 3:] - causal_changed[:, 3:]).abs().max())
assert isolated_error == 0.0
assert causal_error > 0.0
sdpa_model = copy.deepcopy(model)
for block in sdpa_model.blocks:
    block.attention.backend = "sdpa"
with torch.no_grad():
    sdpa_logits = sdpa_model(ids, positions=positions, segments=document_ids)["logits"]
assert torch.allclose(original, sdpa_logits, atol=1e-6, rtol=1e-6)

chapter = (ROOT / "course/chapters/03.md").read_text()
simple_code = next(code for code in re.findall(r"```python\n(.*?)```", chapter, flags=re.S) if "class CausalAttention" in code)
simple_namespace = {"torch": torch}
exec(compile(simple_code, "course/chapters/03.md:CausalAttention", "exec"), simple_namespace)
simple_output = simple_namespace["CausalAttention"](8, heads=2)(torch.zeros(1, 4, 8))
assert list(simple_output.shape) == [1, 4, 8]
assert "attention_mask(" not in simple_code
assert ".tril()[None, None]" in simple_code

trace = []


def traced_mask(queries, keys, valid=None, segments=None):
    result = attention_mask(queries, keys, valid, segments)
    trace.append({"queries": queries.tolist(), "keys": keys.tolist(), "allowed": result.int().tolist()})
    return result


with torch.no_grad(), patch("tiny_perceptron.attention.attention_mask", side_effect=traced_mask):
    state = model(ids[:, :3])["cache"]
    model(ids[:, 3:4], cache=state)
assert trace[-1]["queries"] == [3]
assert trace[-1]["allowed"] == [[[[1, 1, 1, 1]]]]
tril_row = torch.ones(1, 4, dtype=torch.bool).tril().int().tolist()
assert tril_row == [[1, 0, 0, 0]]

examples = [shifted([1, 20, 21, 2]), shifted([1, 30, 2])]
pad, pack = _packing_batch(examples, SimpleNamespace(device="cpu"))
assert pack[1].tolist() == [[20, 21, 2, 30, 2]]
assert pack[3].tolist() == [[0, 1, 2, 0, 1]]
assert pack[4].tolist() == [[0, 0, 0, 1, 1]]
assert pad[1][1, -1].item() == IGNORE
uniform_logits = torch.zeros(2, 3, 40)
total, count = loss_sum(uniform_logits, pad[1])
assert count.item() == 5

records = [{"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}]
mechanism = _mechanism_examples(records, 128)
assert len(mechanism[0][0]) == 7
chat_x, chat_y = render_chat(records[0]["messages"])
tokenizer = ByteTokenizer()
assert chat_y[chat_y != IGNORE].tolist() == tokenizer.encode("answer") + [tokenizer.eos_id]
samplers = [random.Random(42), random.Random(42)]
draws = [[sampler.choices(range(45), k=3) for _ in range(40)] for sampler in samplers]
assert draws[0] == draws[1]

record_path = ROOT / "docs/course-experiments/results/efficiency.json"
record = json.loads(record_path.read_bytes())
packing = record["results"]["packing"]
updates = packing["actual_updates"]
historic_revision = "a5133f444e0dca2347635f74f8ac83a27bb203d4"
source_path = "scripts/course_experiments/architecture.py"
historic = subprocess.check_output(["git", "show", f"{historic_revision}:{source_path}"], cwd=ROOT)
assert digest(historic) == record["code_sha256"][source_path]
functions = {"_packing_batch", "_packing_probe", "_packing_updates", "_mechanism_examples", "_gradients", "_sync", "_heldout", "run_efficiency"}
historic_tree = ast.parse(historic)
historic_functions = {node.name: node for node in historic_tree.body if isinstance(node, ast.FunctionDef) and node.name in functions}
current_tree = ast.parse((ROOT / source_path).read_bytes())
current_functions = {node.name: node for node in current_tree.body if isinstance(node, ast.FunctionDef) and node.name in functions}
function_matches = {name: ast.dump(node) == ast.dump(current_functions[name]) for name, node in historic_functions.items()}
assert all(function_matches.values())
code_hash_checks = {}
for name in ["tiny_perceptron/attention.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py"]:
    observed = digest((ROOT / name).read_bytes())
    assert observed == record["code_sha256"][name]
    code_hash_checks[name] = observed

assert packing["document_prefix_lengths"] == [7, 7]
assert packing["padded_positions"] == packing["packed_positions"] == packing["effective_input_positions"] == 14
rounding = {
    "logit_max_error": (packing["logit_max_error"], "1.90735e-06"),
    "gradient_max_error": (packing["gradient_max_error"], "1.78814e-07"),
    "second_document_error_without_isolation": (packing["second_document_error_without_isolation"], "5.42937"),
    "weight_max_error_after_updates": (updates["weight_max_error_after_updates"], "5.97909e-06"),
}
for value, expected_string in rounding.values():
    assert f"{value:.6g}" == expected_string
branches = {}
for name in ["padded", "packed"]:
    branch = updates[name]
    assert branch["optimizer_updates"] == branch["steps"] == 40
    assert branch["effective_tokens"] == 845
    milliseconds = branch["warm_step_median_seconds"] * 1000
    assert f"{milliseconds:.3f}" == {"padded": "10.799", "packed": "11.191"}[name]
    heldout = {}
    for split, expected_matches, expected_records in [("validation", 0, 5), ("test", 1, 10)]:
        values = branch["heldout"][split]
        assert sum(sample["exact"] for sample in values["samples"]) == values["matches"] == expected_matches
        assert len(values["samples"]) == values["records"] == expected_records
        heldout[split] = {"matches": values["matches"], "records": values["records"]}
    sample = next(sample for sample in branch["heldout"]["test"]["samples"] if sample["expected"] == "circle" and sample["messages"][0]["content"].endswith("shape?"))
    assert sample["generated"] == "low"
    branches[name] = {"updates": 40, "effective_targets": 845, "median_milliseconds": milliseconds, "formatted_milliseconds": f"{milliseconds:.3f}", "heldout": heldout, "shape_sample": sample}
baseline = record["results"]["models"]["mha"]
for split, matches, records_count in [("validation", 2, 5), ("test", 6, 10)]:
    values = baseline["heldout"][split]
    assert sum(sample["exact"] for sample in values["samples"]) == values["matches"] == matches
    assert values["records"] == records_count
assert branches["packed"]["median_milliseconds"] > branches["padded"]["median_milliseconds"]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-5/verify.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "threads": str(torch.get_num_threads())},
    "result": "全部斷言通過；CPU 機制核對與保存的 L4 結果查核分開，未重跑 L4 訓練。",
    "mask": {"shape": list(isolated.shape), "batch_shape": list(batch_mask.shape), "dtype": str(isolated.dtype), "isolated": isolated[0, 0].int().tolist(), "causal_last_row": causal[0, 0, -1].int().tolist(), "changed_segments": changed_segments[0, 0].int().tolist(), "invalid_last_key": padded_mask[0, 0].int().tolist(), "monotonic_intersection": True, "segment_label_renaming_invariant": True, "dense_weights_shape": list(weights.shape)},
    "attention_routes": {"manual_sdpa_logit_max_error": float((original - sdpa_logits).abs().max()), "tolerance": "atol=rtol=1e-6", "simple_class_output_shape": list(simple_output.shape), "simple_class_uses_tril_only": True, "simple_class_source_sha256": digest((ROOT / "course/chapters/03.md").read_bytes())},
    "cache": {"last_layer_call": trace[-1], "tril_one_by_four": tril_row},
    "eos_and_positions": {"input_ids": ids.tolist(), "model_positions": positions.tolist(), "isolated_second_document_error": isolated_error, "causal_second_document_error": causal_error},
    "packing_targets": {"input": pack[0].tolist(), "labels": pack[1].tolist(), "positions": pack[3].tolist(), "segments": pack[4].tolist(), "padded_labels": pad[1].tolist(), "effective_targets": count.item(), "uniform_loss_sum": total.item(), "assistant_input_length": len(mechanism[0][0]), "assistant_only_input": mechanism[0][0].tolist(), "assistant_only_labels": mechanism[0][1].tolist(), "sft_supervised_labels": chat_y[chat_y != IGNORE].tolist(), "same_seed_batches_equal": draws[0] == draws[1]},
    "empirical_record": {"path": record_path.relative_to(ROOT).as_posix(), "sha256": digest(record_path.read_bytes()), "environment": {key: record[key] for key in ["python_version", "torch_version", "gpu", "device", "seed"]}, "model_config": baseline["model"]["config"], "dataset": record["results"]["dataset"], "packing_probe": {key: value for key, value in packing.items() if key != "actual_updates"}, "numeric_rounding": {key: {"raw": value, "formatted": expected_string} for key, (value, expected_string) in rounding.items()}, "branches": branches, "baseline_scores": {split: {key: baseline["heldout"][split][key] for key in ["matches", "records"]} for split in ["validation", "test"]}, "max_document_positions": 128 // 3, "warmup_updates_excluded": 3, "measured_updates": 40 - 3, "parameter_max_error": updates["weight_max_error_after_updates"], "raw_step_latencies_saved": False},
    "original_experiment_code": {"path": source_path, "retrieved_with": f"git show {historic_revision}:{source_path}", "revision": historic_revision, "sha256": digest(historic), "recorded_sha256": record["code_sha256"][source_path], "function_ast_matches_current": function_matches, "other_code_sha256": code_hash_checks, "function_excerpts": {name: ast.get_source_segment(historic.decode(), node) for name, node in historic_functions.items()}},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
