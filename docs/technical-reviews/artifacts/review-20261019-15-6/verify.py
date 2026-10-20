"""核對列加總、練習變體與 MoEFFN 的 dispatch / PAD 行為。"""

import hashlib
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.architecture import _forward, _routing
from tiny_perceptron.data import pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import MoEFFN

torch.set_num_threads(1)
torch.manual_seed(1729)


def combine(rows, contributions):
    result = torch.zeros(3, 2)
    pointer = result.data_ptr()
    returned = result.index_add_(0, rows, contributions)
    overwritten = torch.zeros(3, 2)
    for row, value in zip(rows, contributions, strict=True):
        overwritten[row] = value
    return {
        "rows": rows.tolist(),
        "contributions": contributions.tolist(),
        "sum": result.tolist(),
        "overwrite": overwritten.tolist(),
        "same_object_returned": returned is result,
        "storage_pointer_unchanged": result.data_ptr() == pointer,
        "result_shape": list(result.shape),
        "index_dtype": str(rows.dtype),
        "contribution_dtype": str(contributions.dtype),
    }


base = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
examples = [
    combine(torch.tensor([2, 0, 2]), base),
    combine(torch.tensor([1, 0, 1]), base),
    combine(torch.tensor([1, 0, 1]), base[[2, 1, 0]]),
]
expected_sum = [
    [[2.0, 2.0], [0.0, 0.0], [4.0, 4.0]],
    [[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]],
    [[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]],
]
expected_overwrite = [
    [[2.0, 2.0], [0.0, 0.0], [3.0, 3.0]],
    [[2.0, 2.0], [3.0, 3.0], [0.0, 0.0]],
    [[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]],
]
for observed, summed, overwritten in zip(examples, expected_sum, expected_overwrite, strict=True):
    assert observed["sum"] == summed
    assert observed["overwrite"] == overwritten
    assert observed["same_object_returned"] and observed["storage_pointer_unchanged"]

dispatch_checks = []
for top_k in (1, 2):
    module = MoEFFN(width=4, experts=4, top_k=top_k)
    with torch.no_grad():
        module.router.weight.copy_(torch.tensor([[4.0] * 4, [3.0] * 4, [-3.0] * 4, [-4.0] * 4]))
    x = torch.arange(1.0, 25.0).reshape(2, 3, 4) / 24
    calls = [0] * 4
    handles = []
    for index, expert in enumerate(module.experts):
        def count_inputs(_module, args, expert_index=index):
            calls[expert_index] += len(args[0])
        handles.append(expert.register_forward_pre_hook(count_inputs))
    actual, _, chosen = module(x)
    for handle in handles:
        handle.remove()
    flat = x.reshape(-1, 4)
    weights, indices = module.router(flat).softmax(-1).topk(top_k, dim=-1)
    if top_k > 1:
        weights = weights / weights.sum(-1, keepdim=True)
    reference = torch.zeros_like(flat)
    for row in range(len(flat)):
        for slot in range(top_k):
            reference[row] += module.experts[int(indices[row, slot])](flat[row]) * weights[row, slot]
    torch.testing.assert_close(actual.reshape(-1, 4), reference, rtol=1e-6, atol=1e-7)
    expected_calls = torch.bincount(chosen.flatten(), minlength=4).tolist()
    assert calls == expected_calls and sum(calls) == 6 * top_k
    dispatch_checks.append({
        "top_k": top_k,
        "chosen": chosen.tolist(),
        "expert_input_rows": calls,
        "expected_input_rows": expected_calls,
        "output_shape": list(actual.shape),
        "max_absolute_error_against_tokenwise_sum": float((actual.detach().reshape(-1, 4) - reference.detach()).abs().max()),
    })

model = TinyLM(ModelConfig(width=4, heads=1, layers=2, max_length=8, experts=4, top_k=2))
model.eval()
batch = [
    (torch.tensor([3, 4, 5]), torch.tensor([4, 5, 6])),
    (torch.tensor([7]), torch.tensor([8])),
]
ids, _, valid = pad_batch(batch)
captured = []
router_rows = [0] * len(model.blocks)
expert_rows = [0] * len(model.blocks)
handles = []
for layer, block in enumerate(model.blocks):
    def capture(module, args, result):
        captured.append((module, args[0].detach(), result[2].detach(), result[1].detach()))
    handles.append(block.ffn.register_forward_hook(capture))
    def count_router(_module, args, layer_index=layer):
        router_rows[layer_index] += len(args[0])
    handles.append(block.ffn.router.register_forward_pre_hook(count_router))
    for expert in block.ffn.experts:
        def count_expert(_module, args, layer_index=layer):
            expert_rows[layer_index] += len(args[0])
        handles.append(expert.register_forward_pre_hook(count_expert))
with torch.no_grad():
    raw_result = model(ids, valid=valid)
for handle in handles:
    handle.remove()
assert router_rows == [6, 6]
assert expert_rows == [12, 12]

valid_auxiliary = []
valid_counts = []
layer_details = []
with torch.no_grad():
    for module, inputs, chosen, raw_aux in captured:
        probability = module.router(inputs[valid]).float().softmax(-1)
        selected = chosen[valid.reshape(-1)]
        load = F.one_hot(selected, len(module.experts)).float().mean((0, 1))
        auxiliary = len(module.experts) * (load * probability.mean(0)).sum()
        valid_auxiliary.append(auxiliary)
        counts = torch.bincount(selected.flatten(), minlength=len(module.experts))
        valid_counts.append(counts)
        layer_details.append({
            "raw_auxiliary_including_pad": float(raw_aux),
            "valid_auxiliary": float(auxiliary),
            "valid_dispatch_counts": counts.tolist(),
            "valid_mean_router_probability": probability.mean(0).tolist(),
        })
    expected_auxiliary = torch.stack(valid_auxiliary).sum()
    corrected = _forward(model, ids, valid)
    torch.testing.assert_close(corrected["auxiliary"], expected_auxiliary, rtol=1e-6, atol=1e-7)
    routing = _routing(model, batch, SimpleNamespace(device="cpu"))
assert routing["effective_input_tokens"] == 4 and routing["padding_excluded"]
for observed, counts in zip(routing["layers"], valid_counts, strict=True):
    assert observed["dispatch_counts"] == counts.tolist()
    assert observed["dispatch_denominator"] == 8

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-6/verify.py",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "platform": platform.platform(),
        "seed": "1729",
        "threads": "1",
    },
    "repository_code_sha256": {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in ["tiny_perceptron/modern.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/architecture.py"]
    },
    "row_examples": examples,
    "dispatch_checks": dispatch_checks,
    "padding": {
        "input_ids": ids.tolist(),
        "valid": valid.tolist(),
        "raw_router_input_rows_per_layer": router_rows,
        "raw_expert_input_rows_per_layer": expert_rows,
        "layers": layer_details,
        "raw_auxiliary": float(raw_result["auxiliary"]),
        "expected_valid_auxiliary": float(expected_auxiliary),
        "script_valid_auxiliary": float(corrected["auxiliary"]),
        "routing": routing,
    },
    "result": "全部斷言通過；算例精確相等；浮點模型核對使用 rtol=1e-6、atol=1e-7。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
