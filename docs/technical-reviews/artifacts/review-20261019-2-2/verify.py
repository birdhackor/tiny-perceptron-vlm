"""核對凍結教材的拼接、求導記錄與固定視窗輸入。"""

import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.train_simple import examples
from tiny_perceptron.simple import ContextMLP


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source = ROOT / "docs/course-review-20261019/freeze-01/sources/2.2.md"
assert digest(source) == "d6df938a10d357274b140377b897aea134047be6aed8f894d2a1a8384db0b219"
code = re.search(r"```python\n(.*?)```", source.read_text(), re.S)[1]
namespace = {"__name__": "__main__"}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(source), "exec"), namespace)
embedding, ids, x, context = (namespace[name] for name in ("embedding", "ids", "x", "context"))
weight_after_initialization = embedding.weight.detach().clone()
assert ids.dtype == torch.int64 and x.dtype == torch.float32
assert list(ids.shape) == [2, 3]
assert list(x.shape) == [2, 3, 2]
assert list(context.shape) == [2, 6]
assert context.tolist() == [[1, 10, 2, 20, 3, 30], [3, 30, 2, 20, 1, 10]]
assert torch.equal(context.reshape_as(x), x)
assert x.requires_grad and context.requires_grad
assert type(x.grad_fn).__name__ == "EmbeddingBackward0"
assert type(context.grad_fn).__name__ == "ViewBackward0"
assert embedding.weight.grad is None
assert torch.equal(embedding.weight.detach(), weight_after_initialization)
calls = [node.func for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
assert not any(isinstance(node, ast.Attribute) and node.attr in {"backward", "step"} for node in calls)

changed_ids = ids.clone()
changed_ids[0] = torch.tensor([2, 1, 3])
changed = embedding(changed_ids).reshape(2, -1)
assert changed.tolist() == [[2, 20, 1, 10, 3, 30], [3, 30, 2, 20, 1, 10]]
assert context.sum(1).tolist() == changed.sum(1).tolist() == [66, 66]
position_weights = torch.tensor([1., 0., 0., 0., 2., 0.])
weighted = context @ position_weights
assert weighted.tolist() == [7, 5]

general_shapes = []
for batch, positions, width in [(1, 1, 1), (2, 3, 2), (3, 5, 4)]:
    values = torch.arange(batch * positions * width).reshape(batch, positions, width)
    flattened = values.reshape(batch, -1)
    assert list(flattened.shape) == [batch, positions * width]
    assert torch.equal(flattened.reshape_as(values), values)
    general_shapes.append({"input": list(values.shape), "output": list(flattened.shape)})

gradient_mode = {"before": torch.is_grad_enabled()}
with torch.no_grad():
    gradient_mode["inside"] = torch.is_grad_enabled()
gradient_mode["after"] = torch.is_grad_enabled()
assert gradient_mode == {"before": True, "inside": False, "after": True}

vocabulary = {"我": 2, "想": 3, "喝": 4, "水": 5}
documents = ["我想喝水"]
windows = []
experiment_file = ROOT / "scripts/course_experiments/text.py"
tree = ast.parse(experiment_file.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_simple_examples")
experiment_namespace = {"torch": torch}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(experiment_file), "exec"), experiment_namespace)
for length in [1, 3, 5]:
    actual, labels = examples(documents, vocabulary, length)
    experiment_inputs, experiment_labels = experiment_namespace["_simple_examples"](documents, vocabulary, length)
    assert torch.equal(actual, experiment_inputs)
    assert torch.equal(labels, experiment_labels)
    expected_rows = []
    encoded = [2, 3, 4, 5]
    for position in range(len(encoded) + 1):
        prefix = encoded[:position]
        expected_rows.append(([0] * max(0, length - len(prefix)) + prefix)[-length:])
    assert actual.tolist() == expected_rows
    assert labels.tolist() == [2, 3, 4, 5, 0]
    windows.append({"T": length, "inputs": actual.tolist(), "targets": labels.tolist()})

model = ContextMLP(6, context=3, width=2)
assert list(model(torch.tensor([[2, 3, 4]])).shape) == [1, 6]
try:
    model(torch.tensor([[0, 2, 3, 4, 5]]))
except ValueError as error:
    mismatch = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("不同視窗長度應被固定輸入模型拒絕")

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-2-2/verify.py",
    "result": "所有斷言通過；未執行訓練、反向求導或參數更新。",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": str(x.dtype)},
    "source_sha256": digest(source),
    "original_stdout": stdout.getvalue(),
    "original_ids": ids.tolist(),
    "embedding_values": x.tolist(),
    "context_values": context.tolist(),
    "total_elements": x.numel(),
    "inferred_length": x.numel() // ids.shape[0],
    "changed_ids": changed_ids.tolist(),
    "changed_context": changed.tolist(),
    "original_sums": context.sum(1).tolist(),
    "changed_sums": changed.sum(1).tolist(),
    "position_weights": position_weights.tolist(),
    "position_weighted_results": weighted.tolist(),
    "general_shapes": general_shapes,
    "gradient_mode": gradient_mode,
    "grad_functions": {"lookup": type(x.grad_fn).__name__, "reshape": type(context.grad_fn).__name__},
    "parameter_grad_is_none": embedding.weight.grad is None,
    "parameter_unchanged_after_lookup": torch.equal(embedding.weight.detach(), weight_after_initialization),
    "windows": windows,
    "fixed_window_mismatch": mismatch,
    "repository_sha256": {name: digest(ROOT / name) for name in ["scripts/train_simple.py", "scripts/course_experiments/text.py", "tiny_perceptron/simple.py"]},
}
print(json.dumps(result, ensure_ascii=False, indent=2))
