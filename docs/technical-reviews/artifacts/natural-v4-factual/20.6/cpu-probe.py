"""Independent bounded CPU verification of section 20.6 and needed mechanics."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.alignment import LoRALinear

torch.set_num_threads(1)
environment = {
    "python": sys.version.split()[0], "torch": torch.__version__,
    "torch_git_version": torch.version.git_version, "device": "cpu",
    "cuda_available": str(torch.cuda.is_available()),
}
print("ENVIRONMENT", json.dumps(environment))
assert torch.__version__ == "2.14.1+cpu"

def section(path, identity):
    raw = (ROOT / path).read_text()
    start = re.search(r"^## " + re.escape(identity) + r" .+$", raw, re.M).start()
    end = re.search(r"^## ", raw[start + 1:], re.M)
    return raw[start:start + 1 + end.start()] if end else raw[start:]

for path, identity in [("course/chapters/20.md", "20.6"),
                       ("course/chapters/08.md", "8.8"),
                       ("course/chapters/08.md", "8.13"),
                       ("course/chapters/11.md", "11.2")]:
    raw = section(path, identity)
    print("EXACT_PUBLISHED_BLOCK", identity, hashlib.sha256(raw.encode()).hexdigest())
    block = re.search(r"```python\n(.*?)\n```", raw, re.S).group(1)
    exec(compile(block, path + "#" + identity, "exec"), {})

for rank in (2, 1):
    torch.manual_seed(42)
    layer = LoRALinear(nn.Linear(4, 3), rank=rank, alpha=rank)
    x = torch.ones(1, 4, requires_grad=True)
    snapshots = {name: p.detach().clone() for name, p in layer.named_parameters()}
    assert all(snapshots[name].data_ptr() != p.data_ptr() for name, p in layer.named_parameters())
    params = [p for p in layer.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.1)
    y = layer(x)
    expected_y = torch.nn.functional.linear(x, layer.base.weight, layer.base.bias)
    expected_y = expected_y + (x @ layer.a.T @ layer.b.T) * (layer.alpha / layer.rank)
    assert torch.equal(y, expected_y)
    loss = y.square().mean()
    # Independently derive dL/dy = 2y / (one row * three outputs).
    gy = 2 * y.detach() / 3
    expected_b_grad = gy.T @ (x.detach() @ layer.a.detach().T)
    expected_a_grad = (gy @ layer.b.detach()).T @ x.detach()
    expected_x_grad = gy @ layer.base.weight.detach() + gy @ layer.b.detach() @ layer.a.detach()
    loss.backward()
    errors = {
        "A_grad": float((layer.a.grad - expected_a_grad).abs().max()),
        "B_grad": float((layer.b.grad - expected_b_grad).abs().max()),
        "x_grad_through_frozen_base": float((x.grad - expected_x_grad).abs().max()),
    }
    assert all(value < 1e-7 for value in errors.values())
    assert layer.base.weight.grad is None and layer.base.bias.grad is None
    assert layer.a.grad.abs().max() == 0 and layer.b.grad.abs().max() > 0
    gradients = {name: p.grad.detach().clone() for name, p in layer.named_parameters() if p.requires_grad}
    optimizer.step()
    update_errors = {
        name: float((p.detach() - (snapshots[name] - 0.1 * gradients[name])).abs().max())
        for name, p in layer.named_parameters() if p.requires_grad
    }
    assert all(value < 1e-8 for value in update_errors.values())
    assert torch.equal(snapshots["base.weight"], layer.base.weight)
    assert torch.equal(snapshots["base.bias"], layer.base.bias)
    assert not torch.equal(snapshots["b"], layer.b)
    assert torch.equal(snapshots["a"], layer.a)
    first_b_change = float((snapshots["b"] - layer.b).abs().max())
    optimizer.zero_grad(set_to_none=True)
    layer(torch.ones(1, 4)).square().mean().backward()
    second_a_grad = float(layer.a.grad.abs().max())
    assert second_a_grad > 0
    print("RANK_PROBE", json.dumps({
        "rank": rank, "alpha": rank, "scaling": layer.alpha / layer.rank,
        "A_shape": list(layer.a.shape), "B_shape": list(layer.b.shape),
        "trainable_elements": sum(p.numel() for p in params),
        "optimizer_elements": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
        "input": x.detach().tolist(), "initial_output": y.detach().tolist(),
        "loss_mean_over": 3, "initial_loss": float(loss.detach()),
        "base_weight_and_bias_unchanged": True, "first_A_unchanged": True,
        "first_B_max_change": first_b_change, "second_A_grad_max": second_a_grad,
        "gradient_max_errors": errors, "SGD_update_max_errors": update_errors,
        "input_gradient": x.grad.tolist(), "snapshot_has_independent_storage": True,
    }))

# Fix the factors and vary only alpha, including the prerequisite exercise.
torch.manual_seed(0)
scaled = LoRALinear(nn.Linear(4, 3), rank=2, alpha=2)
with torch.no_grad():
    scaled.b.fill_(1)
delta = scaled.merged_weight() - scaled.base.weight
for alpha, multiple in [(4, 2), (6, 3)]:
    scaled.alpha = alpha
    actual = scaled.merged_weight() - scaled.base.weight
    error = float((actual - multiple * delta).abs().max().detach())
    assert torch.allclose(actual, multiple * delta, atol=1e-6, rtol=0)
    print("SCALING_PROBE", json.dumps({"rank": 2, "alpha": alpha, "multiple": multiple, "max_error": error, "atol": 1e-6, "rtol": 0}))
print("BOUNDED_CPU_CHECKS_PASSED")
