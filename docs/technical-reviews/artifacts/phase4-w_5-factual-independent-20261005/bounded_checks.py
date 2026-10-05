"""Independent, bounded CPU checks of W.5; no training or external data."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from torch import nn


torch.set_default_device("cpu")
torch.set_num_threads(1)
assert torch.version.cuda is None
assert not torch.cuda.is_available()

recipes_int = [[2, 1, 0], [1, 2, 1]]
weights_int = [[10, 0], [20, 1], [30, 4]]
independent_integer_products = [
    [sum(row[k] * weights_int[k][j] for k in range(3)) for j in range(2)]
    for row in recipes_int
]
assert independent_integer_products == [[40, 1], [80, 6]]

recipes = torch.tensor(recipes_int, dtype=torch.float32)
weights = torch.tensor(weights_int, dtype=torch.float32)
result = recipes @ weights
assert result.shape == (2, 2)
assert result.tolist() == independent_integer_products
assert torch.equal(result, torch.matmul(recipes, weights))
assert weights.T.shape == (2, 3)
assert weights.T.tolist() == [[10.0, 20.0, 30.0], [0.0, 1.0, 4.0]]

# Only change one store's recipe: the other store must retain its two outputs.
changed_recipes = recipes.clone()
changed_recipes[0, 2] = 2.0
changed_result = changed_recipes @ weights
assert changed_result.tolist() == [[100.0, 9.0], [80.0, 6.0]]

# Incompatible contracted axes: a 2x3 matrix cannot multiply a 2x2 matrix.
try:
    recipes @ weights[:2]
except RuntimeError as error:
    mismatch_error = str(error)
else:
    raise AssertionError("Missing expected incompatible-shape RuntimeError")

layer = nn.Linear(3, 2)
assert layer.weight.shape == (2, 3)
assert layer.bias.shape == (2,)
assert torch.is_grad_enabled()
with torch.no_grad():
    assert not torch.is_grad_enabled()
    copied = layer.weight.copy_(weights.T)
    assert copied is layer.weight
    layer.bias.copy_(torch.tensor([5.0, 0.0]))
    assert (layer.weight * 2).requires_grad is False
assert torch.is_grad_enabled()
assert layer.weight.requires_grad is True
assert layer.weight.grad_fn is None
assert layer.weight.grad is None
assert layer.bias.grad is None

original_linear = layer(recipes)
assert original_linear.tolist() == [[45.0, 1.0], [85.0, 6.0]]
assert original_linear.requires_grad is True
assert isinstance(original_linear.tolist(), list)
assert all(isinstance(row, list) for row in original_linear.tolist())
assert all(isinstance(value, float) for row in original_linear.tolist() for value in row)
with torch.no_grad():
    layer.bias.copy_(torch.tensor([8.0, 0.0]))
changed_linear = layer(recipes)
assert changed_linear.tolist() == [[48.0, 1.0], [88.0, 6.0]]
assert (changed_linear - original_linear).tolist() == [[3.0, 0.0], [3.0, 0.0]]

base = Path(__file__).resolve().parent
module_paths = {
    "torch/nn/modules/linear.py": Path(torch.nn.modules.linear.__file__),
    "torch/autograd/grad_mode.py": Path(torch.autograd.grad_mode.__file__),
    "torch/_torch_docs.py": Path(torch.__file__).parent / "_torch_docs.py",
    "torch/_tensor_docs.py": Path(torch.__file__).parent / "_tensor_docs.py",
}
source_comparison = []
for relative, installed in module_paths.items():
    downloaded = base / "sources" / relative.replace("/", "--")
    installed_sha = hashlib.sha256(installed.read_bytes()).hexdigest()
    downloaded_sha = hashlib.sha256(downloaded.read_bytes()).hexdigest()
    source_comparison.append({
        "relative_path": relative,
        "installed_path": str(installed),
        "installed_sha256": installed_sha,
        "downloaded_sha256": downloaded_sha,
        "exact_match": installed_sha == downloaded_sha,
    })
    assert installed_sha == downloaded_sha

facts = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "pure_integer_sums": independent_integer_products,
    "original_matrix_product": result.tolist(),
    "original_shape": list(result.shape),
    "changed_one_store": changed_result.tolist(),
    "incompatible_shape_error": mismatch_error,
    "original_linear": original_linear.tolist(),
    "bias_8_linear": changed_linear.tolist(),
    "bias_change_differences": (changed_linear - original_linear).tolist(),
    "no_grad_restored": torch.is_grad_enabled(),
    "linear_forward_still_requires_grad": original_linear.requires_grad,
    "parameter_gradients_uncomputed": layer.weight.grad is None and layer.bias.grad is None,
    "official_source_matches_installed_bytes": source_comparison,
    "arithmetic_comparison": "Exact integer sums and exact representable float32 values; tolerance 0.",
}
print(json.dumps(facts, ensure_ascii=False, indent=2))
