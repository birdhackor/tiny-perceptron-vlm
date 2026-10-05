"""Bounded CPU checks for 3.4, keeping the lesson's original fence intact."""
import contextlib
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.attention import CausalAttention

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
OUT = Path(__file__).resolve().parent
torch.set_default_device("cpu")
torch.set_printoptions(precision=12)
namespace = {}
original_stdout = io.StringIO()
fence = OUT / "original/fence-1.py"
with contextlib.redirect_stdout(original_stdout):
    exec(compile(fence.read_bytes(), str(fence), "exec"), namespace)
q, k, v, weights, output = [namespace[name] for name in ("q", "k", "v", "weights", "output")]
denominator = math.exp(1) + math.exp(0)
scalar_weights = [math.exp(1) / denominator, math.exp(0) / denominator]
scalar_output = [scalar_weights[0] * a + scalar_weights[1] * b for a, b in zip([10, 20, 30], [-1, -2, -3])]
expected = torch.tensor([scalar_output], dtype=torch.float64)
base_error = float((output.double() - expected).abs().max())
assert base_error < 3e-6
assert not any(namespace[name].requires_grad for name in ("q", "k", "v", "weights", "output"))
assert output.shape == (1, 3)

q, k, v = [value.double() for value in (q, k, v)]
weights = (q @ k.T).softmax(dim=-1)
output = weights @ v
assert torch.allclose(output, expected, atol=1e-12, rtol=0)
changed_v = v.clone()
changed_v[0] *= 2
changed_weights = (q @ k.T).softmax(dim=-1)
changed_output = changed_weights @ changed_v
exercise_expected = [scalar_weights[0] * a + scalar_weights[1] * b for a, b in zip([20, 40, 60], [-1, -2, -3])]
assert torch.equal(changed_weights, weights)
assert torch.allclose(changed_output, torch.tensor([exercise_expected], dtype=torch.float64), atol=1e-12, rtol=0)
assert [round(x, 3) for x in scalar_output] == [7.042, 14.083, 21.125]
assert [round(x, 3) for x in exercise_expected] == [14.352, 28.704, 43.057]

only_v_swapped = weights @ v.flip(0)
jointly_swapped = (q @ k.flip(0).T).softmax(-1) @ v.flip(0)
assert not torch.allclose(only_v_swapped, output)
assert torch.allclose(jointly_swapped, output, atol=1e-12, rtol=0)

three_queries = torch.tensor([[1., 0.], [0., 1.], [2., 0.]], dtype=torch.float64)
scores_three = three_queries @ k.T
weights_three = scores_three.softmax(-1)
output_three = weights_three @ v
assert scores_three.shape == (3, 2)
assert output_three.shape == (3, 3)
assert torch.allclose(weights_three.sum(-1), torch.ones(3, dtype=torch.float64), atol=1e-12, rtol=0)
wrong_axis = scores_three.softmax(0)
assert not torch.allclose(wrong_axis.sum(-1), torch.ones(3, dtype=torch.float64))

collision_v = v.clone()
delta = torch.tensor([1., 2., 3.], dtype=torch.float64)
collision_v[0] += delta
collision_v[1] -= scalar_weights[0] / scalar_weights[1] * delta
collision_error = float((weights @ collision_v - output).abs().max())
assert collision_error < 1e-12 and not torch.equal(collision_v, v)

torch.manual_seed(7)
layer = CausalAttention(width=2, heads=1)
x = torch.tensor([[[1., 2.], [3., 4.]]])
projections = {name: getattr(layer, name)(x) for name in ("q", "k", "v")}
parameter_info = {name: {"shape": list(getattr(layer, name).weight.shape), "requires_grad": getattr(layer, name).weight.requires_grad, "grad_is_none": getattr(layer, name).weight.grad is None} for name in projections}
assert len({getattr(layer, name).weight.data_ptr() for name in projections}) == 3
assert all(item["requires_grad"] and item["grad_is_none"] for item in parameter_info.values())
assert all(value.shape == x.shape for value in projections.values())

result = {
    "environment": {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "torch_threads": str(torch.get_num_threads())},
    "original_fence_sha256": hashlib.sha256(fence.read_bytes()).hexdigest(),
    "original_fence_stdout": original_stdout.getvalue(),
    "original_float32_max_abs_error_vs_independent_scalar": base_error,
    "denominator": denominator,
    "weights_scalar": scalar_weights,
    "output_scalar": scalar_output,
    "changed_output_scalar": exercise_expected,
    "only_v_swapped_output": only_v_swapped.tolist(),
    "joint_k_v_swap_output": jointly_swapped.tolist(),
    "three_query_scores": scores_three.tolist(),
    "three_query_weights": weights_three.tolist(),
    "three_query_output": output_three.tolist(),
    "wrong_axis_row_sums": wrong_axis.sum(-1).tolist(),
    "collision_original_v": v.tolist(),
    "collision_different_v": collision_v.tolist(),
    "collision_max_abs_output_difference": collision_error,
    "q_k_v_parameter_info": parameter_info,
    "q_k_v_projection_shapes": {name: list(value.shape) for name, value in projections.items()},
    "scope": "Original forward-only demo, independent scalar arithmetic, exercise, row correspondence, non-square axis/shape variation, constructive non-invertibility, and three separate trainable projections. No backward, optimizer, weight update, full training, data/model download, or GPU execution.",
}
(OUT / "probe-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
