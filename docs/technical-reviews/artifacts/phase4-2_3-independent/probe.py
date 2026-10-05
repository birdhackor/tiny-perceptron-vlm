"""Bounded CPU verification for current 2.3; no model/data downloads or training run."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.simple import ContextMLP

torch.set_num_threads(1)
torch.set_default_device("cpu")
torch.manual_seed(42)
assert torch.version.cuda is None and not torch.cuda.is_available()
base = Path(__file__).parent
raw = (base / "original/fence-1.py").read_bytes()
namespace = {"__name__": "__main__"}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    exec(compile(raw, "original/fence-1.py", "exec"), namespace)
layer, x, y, manual = [namespace[key] for key in ("layer", "x", "y", "manual")]
assert torch.equal(y, torch.tensor([[10.0, 2.0]]))
assert torch.equal(y, manual)
assert layer.weight.shape == (2, 3) and layer.bias.shape == (2,)
assert layer.weight.requires_grad and layer.bias.requires_grad
assert y.requires_grad
assert layer.weight.grad is None and layer.bias.grad is None
assert torch.equal(layer.weight, torch.tensor([[1., 0., 2.], [0., -1., 1.]]))
assert torch.equal(layer.bias, torch.tensor([1., 0.]))
records = {
    "environment": {"python": sys.version, "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version),
                    "cuda_build": str(torch.version.cuda), "device": str(x.device),
                    "threads": torch.get_num_threads()},
    "original": {"fence_sha256": hashlib.sha256(raw).hexdigest(),
                 "stdout": capture.getvalue(), "input_shape": list(x.shape),
                 "weight_shape": list(layer.weight.shape), "bias_shape": list(layer.bias.shape),
                 "output": y.detach().tolist(), "manual": manual.detach().tolist(),
                 "max_absolute_error": (y - manual).abs().max().item(),
                 "both_parameter_grads_are_none": True,
                 "original_parameters_unchanged_after_forward": True},
}

# The lesson's exercise changes one multiplier, not another output row.
with torch.no_grad():
    layer.weight[0, 2] = 3.0
altered = layer(x)
assert torch.equal(altered, torch.tensor([[14.0, 2.0]]))
records["weight_exercise"] = {"changed_index": [0, 2], "old": 2.0, "new": 3.0,
                              "output": altered.detach().tolist(),
                              "delta": (altered - y).detach().tolist()}
with torch.no_grad():
    layer.weight[0, 2] = 2.0

# Independent scalar construction checks the two leading axes and shared recipes.
rank3 = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
expected = torch.tensor([[[float(row[0]) + 2 * float(row[2]) + 1,
                           -float(row[1]) + float(row[2])] for row in batch] for batch in rank3])
actual = layer(rank3)
assert actual.shape == (2, 4, 2) and torch.equal(actual, expected)
perturbed = rank3.clone()
perturbed[0, 2, 1] += 5.0
delta = layer(perturbed) - actual
expected_delta = torch.zeros_like(actual)
expected_delta[0, 2, 1] = -5.0
assert torch.equal(delta, expected_delta)
records["last_axis"] = {"input_shape": list(rank3.shape), "output_shape": list(actual.shape),
                        "output": actual.detach().tolist(),
                        "independent_scalar_max_absolute_error": (actual - expected).abs().max().item(),
                        "single_input_change": "rank3[0,2,1] += 5",
                        "only_changed_output": [0, 2, 1], "output_delta": -5.0}

# Non-square failure and square silent error are different behaviors.
try:
    x @ layer.weight
except RuntimeError as error:
    wrong_shape_error = str(error)
else:
    raise AssertionError("Non-square omitted transpose unexpectedly succeeded")
square_x = torch.tensor([[1., 2.]])
square_w = torch.tensor([[1., 3.], [2., 4.]])
square_correct = square_x @ square_w.T
square_wrong = square_x @ square_w
assert torch.equal(square_correct, torch.tensor([[7., 10.]]))
assert torch.equal(square_wrong, torch.tensor([[5., 11.]]))
records["transpose"] = {"non_square_error": wrong_shape_error,
                        "square_input": square_x.tolist(), "square_weight": square_w.tolist(),
                        "correct": square_correct.tolist(), "omitted_transpose": square_wrong.tolist()}

# The actual repository ContextMLP flattens only after the batch axis.
model = ContextMLP(4, context=3, width=2)
with torch.no_grad():
    model.embedding.weight.copy_(torch.tensor([[0., 0.], [1., 10.], [2., 20.], [3., 30.]]))
    model.hidden.weight.copy_(torch.tensor([[1., 0., 0., 0., 2., 0.], [0., 1., 0., 0., 0., 0.]]))
    model.hidden.bias.zero_()
ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
features = model.embedding(ids).flatten(1)
mixed = model.hidden(features)
assert torch.equal(features, torch.tensor([[1., 10., 2., 20., 3., 30.], [3., 30., 2., 20., 1., 10.]]))
assert torch.equal(mixed, torch.tensor([[7., 10.], [5., 30.]]))
repository_output = model(ids)
assert torch.equal(repository_output, model.output(torch.tanh(mixed)))
records["repository_context"] = {"ids_shape": list(ids.shape),
                                 "flattened_shape": list(features.shape),
                                 "hidden_weight_shape": list(model.hidden.weight.shape),
                                 "hidden_output": mixed.detach().tolist(),
                                 "forward_expansion_matches": True,
                                 "repository_code_sha256": hashlib.sha256((ROOT / "tiny_perceptron/simple.py").read_bytes()).hexdigest()}

# Separate single-step probe supports trainability, not the lesson's having trained.
optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
before_weight, before_bias = layer.weight.detach().clone(), layer.bias.detach().clone()
loss = (layer(x) - torch.zeros(1, 2)).square().mean()
loss.backward()
expected_weight_grad = torch.tensor([[10., 20., 40.], [2., 4., 8.]])
expected_bias_grad = torch.tensor([10., 2.])
assert torch.equal(layer.weight.grad, expected_weight_grad)
assert torch.equal(layer.bias.grad, expected_bias_grad)
optimizer.step()
expected_weight = before_weight - 0.01 * expected_weight_grad
expected_bias = before_bias - 0.01 * expected_bias_grad
assert torch.allclose(layer.weight, expected_weight, rtol=0, atol=1e-6)
assert torch.allclose(layer.bias, expected_bias, rtol=0, atol=1e-6)
records["separate_one_step"] = {"loss": loss.item(), "loss_denominator": "mean over 1 sample x 2 scalar outputs = 2",
                                "weight_grad": expected_weight_grad.tolist(), "bias_grad": expected_bias_grad.tolist(),
                                "learning_rate": 0.01, "weight_after": layer.weight.detach().tolist(),
                                "bias_after": layer.bias.detach().tolist(),
                                "scope": "One isolated CPU SGD step; no training effectiveness or evaluation claim"}
records["tolerance"] = "Original/exercise/rank3/scalar/transpose values are exact float32 integers; exact equality. SGD update uses atol=1e-6 and rtol=0. Original torch.allclose uses defaults rtol=1e-5, atol=1e-8."
print(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False))
