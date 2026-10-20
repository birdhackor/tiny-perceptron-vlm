"""核對共享表的數值、更新、模型轉換與既存實驗紀錄；不重新訓練模型。"""

import copy
import hashlib
import io
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from scripts.course_experiments.architecture import _clone_config
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.quantization import replace_linear_layers

torch.set_num_threads(1)
torch.manual_seed(42)


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


embedding = nn.Embedding(3, 2)
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
tied = nn.Linear(2, 3, bias=False)
tied.weight = embedding.weight
copied = nn.Linear(2, 3, bias=False)
with torch.no_grad():
    copied.weight.copy_(embedding.weight)
h = torch.tensor([[2.0, 1.0]])
toy = {
    "shape": list(embedding.weight.shape),
    "table_parameters": embedding.weight.numel(),
    "original": [tied(h).tolist(), copied(h).tolist()],
    "identity": [tied.weight is embedding.weight, copied.weight is embedding.weight],
}
with torch.no_grad():
    embedding.weight[1, 1] += 1
toy["exercise"] = [tied(h).tolist(), copied(h).tolist()]
assert toy["original"] == [[[2.0, 1.0, 3.0]], [[2.0, 1.0, 3.0]]]
assert toy["exercise"] == [[[2.0, 2.0, 3.0]], [[2.0, 1.0, 3.0]]]
assert toy["identity"] == [True, False]

pair = nn.Module()
pair.embedding, pair.output = embedding, tied
pair.zero_grad()
loss = embedding(torch.tensor([0])).sum() + tied(h).sum()
loss.backward()
gradient = embedding.weight.grad.clone()
before = embedding.weight.detach().clone()
optimizer = torch.optim.SGD(pair.parameters(), lr=0.1)
optimizer.step()
expected_gradient = torch.tensor([[3.0, 2.0], [2.0, 1.0], [2.0, 1.0]])
assert torch.equal(gradient, expected_gradient)
assert len(list(pair.parameters())) == 1
assert torch.allclose(embedding.weight, before - 0.1 * gradient)
updates = {
    "unique_parameters": len(list(pair.parameters())),
    "names_without_deduplication": [name for name, _ in pair.named_parameters(remove_duplicate=False)],
    "gradient_from_both_roles": gradient.tolist(),
    "one_sgd_update_matches": bool(torch.allclose(embedding.weight, before - 0.1 * gradient)),
}
copied_before = copied.weight.detach().clone()
optimizer.zero_grad()
tied(h).sum().backward()
optimizer.step()
updates["copied_table_unchanged"] = torch.equal(copied.weight, copied_before)
assert updates["copied_table_unchanged"]
with torch.no_grad():
    no_grad_result = embedding.weight * 2
updates["parameter_is_trainable"] = isinstance(embedding.weight, nn.Parameter) and embedding.weight.requires_grad
updates["no_grad_result_requires_grad"] = no_grad_result.requires_grad
assert not no_grad_result.requires_grad

record_path = "docs/course-experiments/results/modern.json"
record = json.loads((ROOT / record_path).read_bytes())
variants = record["results"]["variants"]
base_config = ModelConfig(**variants["baseline"]["model"]["config"])
torch.manual_seed(record["seed"])
initial = TinyLM(base_config)
context = SimpleNamespace(seed=record["seed"], device="cpu")
baseline, baseline_copied = _clone_config(initial, context)
shared, shared_copied = _clone_config(initial, context, tied=True)
counts = [sum(p.numel() for p in m.parameters()) for m in (baseline, shared)]
assert counts == [141568, 124672]
assert counts[0] - counts[1] == 264 * 64 == 16896
assert shared.output.weight is shared.embedding.weight
assert torch.equal(shared.output.weight, baseline.embedding.weight)
assert not torch.equal(shared.output.weight, baseline.output.weight)
assert "output.weight" not in shared_copied
assert set(baseline_copied) - set(shared_copied) == {"output.weight"}
assert all(torch.equal(v, baseline.state_dict()[k]) for k, v in shared.state_dict().items() if k != "output.weight")
ids = torch.tensor([[1, 12, 13, 2]])
with torch.no_grad():
    output_difference = (shared(ids)["logits"] - baseline(ids)["logits"]).abs().max().item()
assert output_difference > 0
model_check = {
    "counts": counts,
    "saved_parameters": counts[0] - counts[1],
    "config": variants["baseline"]["model"]["config"],
    "shared_identity": shared.output.weight is shared.embedding.weight,
    "shared_output_uses_baseline_embedding": torch.equal(shared.output.weight, baseline.embedding.weight),
    "shared_output_equals_baseline_output": torch.equal(shared.output.weight, baseline.output.weight),
    "other_initial_tables_equal": True,
    "synthetic_input_ids": ids.tolist(),
    "initial_max_logit_difference": output_difference,
}

state_buffer = io.BytesIO()
torch.save(shared.state_dict(), state_buffer)
state_buffer.seek(0)
state = torch.load(state_buffer, weights_only=True)
reloaded_tied = TinyLM(replace(base_config, tied=True))
reloaded_tied.load_state_dict(state)
reloaded_untied = TinyLM(base_config)
reloaded_untied.load_state_dict(state)
quantized = replace_linear_layers(copy.deepcopy(shared), bits=4)
conversions = {
    "tied_constructor_after_state_load": reloaded_tied.output.weight is reloaded_tied.embedding.weight,
    "untied_constructor_after_state_load": reloaded_untied.output.weight is reloaded_untied.embedding.weight,
    "untied_loaded_values_equal": torch.equal(reloaded_untied.output.weight, reloaded_untied.embedding.weight),
    "quantized_output_type": type(quantized.output).__name__,
    "quantized_output_has_weight_parameter": hasattr(quantized.output, "weight"),
    "quantized_embedding_still_parameter": isinstance(quantized.embedding.weight, nn.Parameter),
}
assert conversions["tied_constructor_after_state_load"]
assert not conversions["untied_constructor_after_state_load"]
assert conversions["untied_loaded_values_equal"]
assert not conversions["quantized_output_has_weight_parameter"]

audit = {}
expected = {
    "baseline": {"initial": 5.76007, "validation": 2.26331, "test": 2.29163},
    "tied": {"initial": 43.87647, "validation": 2.36276, "test": 2.38422},
}
for name in ("baseline", "tied"):
    variant = variants[name]
    audit[name] = {"model": variant["model"], "training": {k: v for k, v in variant["training"].items() if k != "history"}}
    metrics = {"initial": variant["training"]["initial"], **{key: variant["heldout"][key] for key in ("validation", "test")}}
    audit[name]["metrics"] = {}
    for key, metric in metrics.items():
        recomputed = metric["nll_sum"] / metric["effective_tokens"]
        assert abs(recomputed - metric["nll"]) < 1e-12
        assert round(recomputed, 5) == expected[name][key]
        audit[name]["metrics"][key] = {k: v for k, v in metric.items() if k != "samples"}
        audit[name]["metrics"][key]["recomputed_from_sum_and_count"] = recomputed
    assert variant["training"]["steps"] == variant["training"]["optimizer_updates"] == 240
    assert variant["training"]["skipped_updates"] == 0
assert variants["tied"]["heldout"]["validation"]["nll"] > variants["baseline"]["heldout"]["validation"]["nll"]
assert variants["tied"]["heldout"]["test"]["nll"] > variants["baseline"]["heldout"]["test"]["nll"]

paths = [
    "tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/training.py",
    "tiny_perceptron/quantization.py", "scripts/course_experiments/architecture.py",
    "scripts/course_experiments/common.py", record_path,
]
asset = ROOT / "assets/training/tinystories-v1.tar.gz"
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-6/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu"},
    "toy": toy, "updates": updates, "model": model_check, "conversions": conversions,
    "record_audit": audit,
    "record_environment": {k: record[k] for k in ("revision", "device", "gpu", "seed", "torch_version", "python_version")},
    "dataset": record["results"]["dataset"],
    "code_hashes": {path: digest(path) for path in paths},
    "code_matches_record": {path: digest(path) == record["code_sha256"].get(path) for path in paths if path in record["code_sha256"]},
    "dataset_archive_is_lfs_pointer": asset.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"),
    "limitations": "核對既存紀錄的分母、數值與配置；未重新執行原始 CUDA 訓練，也未重新計算原始資料上的 NLL。合成輸入只核對起始輸出不同。",
    "result": "全部斷言通過",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
