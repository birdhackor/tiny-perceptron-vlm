"""Fresh 20.6 review: exact fence plus bounded independent CPU perturbations."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
from tiny_perceptron.alignment import LoRALinear

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda),
    "cuda_available": str(torch.cuda.is_available()),
    "device": "cpu",
    "threads": str(torch.get_num_threads()),
}
(ART / "cpu-environment.json").write_text(json.dumps(environment, indent=2) + "\n")

# This is the extracted original fence, unchanged; no bootstrap or training runner.
original = ART / "extraction/fence-1.py"
ns = {"__name__": "__main__"}
exec(compile(original.read_bytes(), "course/chapters/20.md#20.6:original-fence-1", "exec"), ns)
assert sum(p.numel() for p in ns["layer"].parameters() if p.requires_grad) == 14
assert torch.equal(ns["base_before"], ns["layer"].base.weight)
assert not torch.equal(ns["branch_before"], ns["layer"].b)

observations = []
for rank in (2, 1):
    torch.manual_seed(42)
    layer = LoRALinear(nn.Linear(4, 3), rank=rank, alpha=2)
    before = {n: p.detach().clone() for n, p in layer.named_parameters()}
    optimizer = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    x = torch.ones(1, 4)
    loss_before = layer(x).square().mean()
    loss_before.backward()
    assert all(torch.equal(before[n], p) for n, p in layer.named_parameters())
    assert layer.base.weight.grad is None and layer.base.bias.grad is None
    assert torch.count_nonzero(layer.a.grad).item() == 0  # B starts at zero.
    assert torch.count_nonzero(layer.b.grad).item() > 0
    optimizer.step()
    assert torch.equal(before["base.weight"], layer.base.weight)
    assert torch.equal(before["base.bias"], layer.base.bias)
    assert torch.equal(before["a"], layer.a)
    assert not torch.equal(before["b"], layer.b)
    count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert count == rank * (4 + 3)
    loss_after = layer(x).square().mean()
    assert loss_after < loss_before
    observations.append({
        "rank": rank, "alpha": 2, "alpha_over_rank": 2 / rank,
        "A_shape": list(layer.a.shape), "B_shape": list(layer.b.shape),
        "output_shape": list(layer(x).shape), "trainable_parameters": count,
        "base_weight_parameters": layer.base.weight.numel(),
        "base_bias_parameters": layer.base.bias.numel(),
        "backward_only_parameters_unchanged": True,
        "base_weight_and_bias_unchanged_after_step": True,
        "A_unchanged_first_step_due_zero_B": True, "B_changed_after_step": True,
        "loss_before": float(loss_before.detach()), "loss_after": float(loss_after.detach()),
    })

# Nonzero correction checks orientation, scaling, and unchanged affine bias.
layer = LoRALinear(nn.Linear(4, 3), rank=2, alpha=6)
with torch.no_grad():
    layer.b.copy_(torch.arange(6).reshape(3, 2) / 10)
x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 4
expected = F.linear(x, layer.merged_weight(), layer.base.bias)
actual = layer(x)
torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
results = {
    "original_fence_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    "original_fence_exit": "all original print outcomes and assertions passed",
    "perturbations": observations,
    "merged_weight_max_absolute_error": float((actual - expected).abs().max()),
    "saving_condition": "rank*(in_features+out_features) < in_features*out_features; W excludes the frozen bias",
    "scope": "One local SGD update for ranks 2 and 1; no images, model loading, data downloads, GPU, or full training.",
}
(ART / "cpu-results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
