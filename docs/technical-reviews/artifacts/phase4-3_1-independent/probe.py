"""Independent, bounded CPU checks of the current lesson 3.1 and its exercises."""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import platform
import sys
import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
original = (BASE / "original/fence-1.py").read_bytes()
namespace = {"__name__": "__main__"}
exec(compile(original, "frozen-current-3.1-fence-1.py", "exec"), namespace)
values, weights, output = [namespace[key] for key in ("values", "weights", "output")]
assert tuple(values.shape) == (2, 2)
assert tuple(weights.shape) == (2,)
assert tuple(output.shape) == (2,)
assert not any(x.requires_grad for x in (values, weights, output))
assert output.grad_fn is None
frozen_values = values.clone()
exact = [Decimal("0.9") * Decimal(2) + Decimal("0.1") * Decimal(0),
         Decimal("0.9") * Decimal(0) + Decimal("0.1") * Decimal(4)]
assert exact == [Decimal("1.8"), Decimal("0.4")]
original_error = [abs(float(actual) - float(expected)) for actual, expected in zip(output, exact)]
assert max(original_error) < 1e-6

swapped = original.replace(b"weights = torch.tensor([0.9, 0.1])", b"weights = torch.tensor([0.1, 0.9])")
swapped = swapped.replace(b"torch.tensor([1.8, 0.4])", b"torch.tensor([0.2, 3.6])")
(BASE / "swapped-fence.py").write_bytes(swapped)
swap_namespace = {"__name__": "__main__"}
exec(compile(swapped, "swapped-current-3.1-fence-1.py", "exec"), swap_namespace)
swapped_exact = [Decimal("0.1") * Decimal(2), Decimal("0.9") * Decimal(4)]
swapped_error = [abs(float(actual) - float(expected)) for actual, expected in zip(swap_namespace["output"], swapped_exact)]
assert max(swapped_error) < 1e-6
assert torch.equal(values, frozen_values)
assert torch.equal(swap_namespace["values"], frozen_values)

endpoint_outputs = [(torch.tensor(w) @ values).tolist() for w in ([1.0, 0.0], [0.0, 1.0])]
assert endpoint_outputs == [[2.0, 0.0], [0.0, 4.0]]
three_positions = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
three_weights = torch.tensor([0.2, 0.3, 0.5])
three_output = three_weights @ three_positions
assert tuple(three_output.shape) == (2,)
assert torch.allclose(three_output, torch.tensor([4.6, 7.4]))
assert torch.allclose(three_output, (three_weights[:, None] * three_positions).sum(dim=0))
try:
    torch.tensor([0.9, 0.1]) @ three_positions
except RuntimeError as error:
    mismatch = str(error)
else:
    raise AssertionError("position-count mismatch must be rejected")

contributions = torch.tensor([0.9, 0.1]) * torch.tensor([1.0, 100.0])
assert torch.allclose(contributions, torch.tensor([0.9, 10.0]))
assert contributions[1] > contributions[0]
wrong_output = output + torch.tensor([0.0, 0.01])
assert not torch.allclose(wrong_output, torch.tensor([1.8, 0.4]))
assert torch.allclose(output + torch.tensor([1e-6, 0.0]), torch.tensor([1.8, 0.4]))
layer = torch.nn.Linear(2, 1, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[-2.0, 4.0]]))
assert layer.weight.min().item() < 0 and layer.weight.sum().item() == 2.0
assert layer(torch.tensor([1.0, 2.0])).item() == 6.0

result = {
    "original_fence_sha256": hashlib.sha256(original).hexdigest(),
    "swapped_fence_sha256": hashlib.sha256(swapped).hexdigest(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu",
                    "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()),
                    "dtype": str(values.dtype), "threads": str(torch.get_num_threads()),
                    "python_executable": sys.executable},
    "axis_and_units": {"values_axis_0": "position; 2 original positions",
                       "values_axis_1": "feature; 2 arbitrary unitless numbers per position",
                       "weights_axis_0": "position; 2 dimensionless proportions",
                       "output_axis_0": "feature; position axis is reduced",
                       "normalization_denominator": "sum(weights)=1; no additional division by position count"},
    "original_output": output.tolist(), "exact_original": [str(x) for x in exact],
    "original_error_against_exact_decimal": original_error,
    "allclose_default": {"rtol": 1e-5, "atol": 1e-8, "rule": "abs(input_i-other_i) <= atol + rtol*abs(other_i) for every i",
                         "reference_1.8_tolerance": 1e-8+1e-5*1.8,
                         "reference_0.4_tolerance": 1e-8+1e-5*0.4,
                         "original_passed": True, "second_feature_plus_0.01_passed": False,
                         "first_feature_plus_0.000001_passed": True},
    "swapped_output": swap_namespace["output"].tolist(), "exact_swapped": [str(x) for x in swapped_exact],
    "swapped_error_against_exact_decimal": swapped_error, "source_values_unchanged": True,
    "endpoints": endpoint_outputs, "three_position_output": three_output.tolist(),
    "mismatch_error": mismatch, "counterexample_contributions": contributions.tolist(),
    "linear_weight": layer.weight.tolist(), "linear_weight_sum": layer.weight.sum().item(),
    "linear_output": layer(torch.tensor([1.0, 2.0])).item(),
    "original_is_training": False, "original_tensor_requires_grad": False, "original_output_grad_fn": None,
    "scope": "Numeric and API demonstration only; no models, training, downloads, or semantic understanding evaluation."
}
(BASE / "probe-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
