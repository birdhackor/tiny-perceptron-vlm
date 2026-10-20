import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch
from torch import nn


root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-01/sources/2.3.md"
raw = source.read_bytes()
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(source), "exec"), namespace)
layer, x, y = (namespace[name] for name in ("layer", "x", "y"))
expected = torch.tensor([[10.0, 2.0]])
assert torch.equal(y, expected)
before = {name: value.detach().clone() for name, value in layer.named_parameters()}
assert all(value.grad is None for value in layer.parameters())
assert all(torch.equal(before[name], value) for name, value in layer.named_parameters())
baseline = {
    "input": x.tolist(),
    "weight": layer.weight.tolist(),
    "weight_shape": list(layer.weight.shape),
    "bias": layer.bias.tolist(),
    "bias_shape": list(layer.bias.shape),
    "output": y.tolist(),
    "manual": namespace["manual"].tolist(),
    "transpose_shape": list(layer.weight.T.shape),
    "parameter_names": list(dict(layer.named_parameters())),
    "requires_grad": {name: value.requires_grad for name, value in layer.named_parameters()},
    "gradients": {name: value.grad for name, value in layer.named_parameters()},
    "original_stdout": captured.getvalue(),
}

try:
    x @ layer.weight
except RuntimeError as error:
    missing_transpose = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("未轉置的非方形矩陣應拒絕相乘")

square_input = torch.tensor([[1.0, 2.0]])
square_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
correct_square = square_input @ square_weight.T
wrong_square = square_input @ square_weight
assert correct_square.tolist() == [[5.0, 11.0]]
assert wrong_square.tolist() == [[7.0, 10.0]]

batched_input = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
batched_output = layer(batched_input)
independent_output = torch.stack(
    [torch.stack([layer(row) for row in sample]) for sample in batched_input]
)
assert tuple(batched_output.shape) == (2, 4, 2)
assert torch.equal(batched_output, independent_output)
changed_input = batched_input.clone()
changed_input[0, 0, 2] += 1
changed_output = layer(changed_input)
expected_delta = torch.zeros_like(batched_output)
expected_delta[0, 0, 0] = 2
expected_delta[0, 0, 1] = 1
assert torch.equal(changed_output - batched_output, expected_delta)

context_input = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                              [[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]])
context = context_input.reshape(2, -1)
context_layer = nn.Linear(6, 2)
with torch.no_grad():
    context_layer.weight.copy_(torch.tensor([[1., 0., 2., 0., 3., 0.],
                                            [0., 1., 0., 2., 0., 3.]]))
    context_layer.bias.zero_()
context_output = context_layer(context)
assert context_output.tolist() == [[14.0, 140.0], [10.0, 100.0]]
changed_context = context_input.clone()
changed_context[0, 0, 0] += 1
changed_context_output = context_layer(changed_context.reshape(2, -1))
assert (changed_context_output - context_output).tolist() == [[1.0, 0.0], [0.0, 0.0]]

with torch.no_grad():
    layer.weight[0, 2] = 3
exercise_output = layer(x)
assert exercise_output.tolist() == [[14.0, 2.0]]
assert (exercise_output - y).tolist() == [[4.0, 0.0]]

print(json.dumps({
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version),
                    "device": str(x.device), "dtype": str(x.dtype)},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "baseline": baseline,
    "missing_transpose": missing_transpose,
    "square_transpose": {"input": square_input.tolist(), "weight": square_weight.tolist(),
                         "correct": correct_square.tolist(), "without_transpose": wrong_square.tolist()},
    "batched": {"input": batched_input.tolist(), "output": batched_output.tolist(),
                "shape": list(batched_output.shape), "independent_equal": True,
                "single_position_delta": expected_delta.tolist()},
    "context": {"input": context_input.tolist(), "flattened": context.tolist(),
                "weight": context_layer.weight.tolist(), "bias": context_layer.bias.tolist(),
                "output": context_output.tolist(),
                "single_sample_delta": (changed_context_output - context_output).tolist()},
    "exercise": {"weight": layer.weight.tolist(), "output": exercise_output.tolist(),
                 "delta": (exercise_output - y).tolist()},
    "result": "全部斷言通過"
}, ensure_ascii=False, indent=2))
