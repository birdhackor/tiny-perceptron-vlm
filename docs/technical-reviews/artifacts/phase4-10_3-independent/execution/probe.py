"""Independent bounded CPU checks for current section 10.3; no optimizer or training."""
import ast
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.multimodal import patchify, scene

ARTIFACT = Path(__file__).resolve().parents[1]
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

environment = {
    "python": platform.python_version(), "python_executable": sys.executable,
    "torch": torch.__version__, "torch_git_version": torch.version.git_version,
    "device": "cpu", "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()), "threads": torch.get_num_threads(),
    "cwd": str(Path.cwd()), "platform": platform.platform(),
}
official_comparisons = {}
for cls, name in [(nn.Linear, "linear.py"), (nn.Embedding, "sparse.py"), (torch.manual_seed, "random.py")]:
    installed = inspect.getsourcefile(cls)
    snapshot = ARTIFACT / "sources" / name
    official_comparisons[name] = {
        "installed_path": installed, "installed_sha256": sha(installed),
        "snapshot_sha256": sha(snapshot), "same_bytes": Path(installed).read_bytes() == snapshot.read_bytes(),
    }
    assert official_comparisons[name]["same_bytes"]

torch.manual_seed(0)
image = scene()[None]
patches = patchify(image, 4)
layer = nn.Linear(48, 8)
before = {name: p.detach().clone() for name, p in layer.named_parameters()}
features = layer(patches)
manual = patches @ layer.weight.T + layer.bias
torch.testing.assert_close(features, manual, rtol=0, atol=1e-7)
assert list(patches.shape) == [1, 16, 48]
assert list(features.shape) == [1, 16, 8]
assert list(layer.weight.shape) == [8, 48]
assert list(layer.bias.shape) == [8]
assert all(p.requires_grad for p in layer.parameters())
assert all(p.grad is None for p in layer.parameters())
assert all(torch.equal(before[n], p) for n, p in layer.named_parameters())
original_forward = {
    "image_shape": list(image.shape), "patches_shape": list(patches.shape),
    "features_shape": list(features.shape), "weight_shape": list(layer.weight.shape),
    "bias_shape": list(layer.bias.shape), "parameter_count": sum(p.numel() for p in layer.parameters()),
    "manual_formula_max_absolute_error": (features - manual).abs().max().item(),
    "all_parameters_require_grad": True, "gradients_after_forward": "all None",
    "parameter_updates_after_forward": 0,
}

# Directly validate row-major patch and within-patch RGB/row/column axes.
sentinel = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
sp = patchify(sentinel, 4)
for index in range(16):
    row, col = divmod(index, 4)
    expected = sentinel[0, :, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4].reshape(48)
    assert torch.equal(sp[0, index], expected)
assert torch.equal(sp[0, 1], sentinel[0, :, 0:4, 4:8].reshape(48))

# A repeated patch must receive the same affine map at each position.
repeated = patches[:, 5:6].expand(1, 16, 48)
repeated_output = layer(repeated)
assert torch.equal(repeated_output[:, 0], repeated_output[:, 15])
perturbed = repeated.clone()
perturbed[:, 3, 0] += 1
changed = layer(perturbed)
other_positions = [i for i in range(16) if i != 3]
assert torch.equal(changed[:, other_positions], repeated_output[:, other_positions])
torch.testing.assert_close(changed[:, 3] - repeated_output[:, 3], layer.weight[:, 0][None], rtol=1e-5, atol=1e-7)

# Re-seeding reproduces this initialization within this installed CPU environment.
torch.manual_seed(0)
repeat_layer = nn.Linear(48, 8)
assert torch.equal(layer.weight, repeat_layer.weight)
assert torch.equal(layer.bias, repeat_layer.bias)
torch.manual_seed(1)
different_layer = nn.Linear(48, 8)
assert not torch.equal(layer.weight, different_layer.weight)

# The required short variation changes width, keeping the 16 patch positions.
torch.manual_seed(0)
wide_layer = nn.Linear(48, 16)
wide_features = wide_layer(patches)
assert list(wide_features.shape) == [1, 16, 16]
assert list(wide_layer.weight.shape) == [16, 48]

# Execute the explicit two-input collision, with exact binary-representable 0.5.
collision_input = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
collision_output = collision_input @ torch.tensor([0.5, 0.5]) + 0.0
assert torch.equal(collision_output, torch.tensor([0.5, 0.5]))

# Contrast integer-index lookup with the affine operation; no semantic evaluation.
table = nn.Embedding(6, 8)
ids = torch.tensor([[2, 4, 2]])
assert torch.equal(table(ids), table.weight[ids])

# Gradients establish trainability only. There is deliberately no parameter update.
features.sum().backward()
assert list(layer.weight.grad.shape) == [8, 48]
assert list(layer.bias.grad.shape) == [8]
assert torch.equal(layer.bias.grad, torch.full((8,), 16.0))
assert all(torch.equal(before[n], p) for n, p in layer.named_parameters())
source_tree = ast.parse((ARTIFACT / "inputs/fence-1.py").read_bytes())
calls = [ast.unparse(n.func) for n in ast.walk(source_tree) if isinstance(n, ast.Call)]
assert not any(x.endswith(".backward") or x.endswith(".step") for x in calls)

result = {
    "environment": environment, "official_snapshot_comparisons": official_comparisons,
    "original_forward": original_forward,
    "patch_axes": {"positions_checked": 16, "values_per_position_checked": 48,
                   "order": "patch row, patch column; then channel, pixel row, pixel column",
                   "second_patch_top_left_rgb_values": sp[0, 1, [0, 16, 32]].tolist()},
    "shared_projection": {"equal_input_equal_output": True, "unmodified_positions_exactly_equal": 15,
                          "single_pixel_delta_matches_weight_column": True},
    "seed_reproducibility": {"same_seed_same_weight_and_bias": True, "different_seed_different_weight": True,
                             "scope": "same PyTorch version, CPU platform, RNG call sequence"},
    "exercise": {"features_shape": list(wide_features.shape), "weight_shape": list(wide_layer.weight.shape)},
    "collision": {"inputs": collision_input.tolist(), "weights": [0.5, 0.5], "bias": 0,
                  "outputs": collision_output.tolist(), "equality": "exact float32"},
    "text_embedding_lookup": {"ids": ids.tolist(), "equals_weight_indexing": True},
    "gradient_variant": {"weight_grad_shape": list(layer.weight.grad.shape), "bias_grad_values": layer.bias.grad.tolist(),
                         "parameter_updates": 0, "scope": "one backward call only; not training"},
    "original_fence_calls": calls,
    "assertions": "all passed", "weights_saved": False,
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
