"""Bounded CPU checks for lesson 11.2; no data download or model training."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tiny_perceptron.model import TinyLM, ModelConfig
from tiny_perceptron.multimodal import MultiModalLM

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(19)
out = {"environment": {"python": sys.version, "torch": torch.__version__,
    "torch_git_version": torch.version.git_version, "device": "cpu", "cuda_build": str(torch.version.cuda)},
    "scope": "Parameter listing, synthetic forward/backward, manual scalar optimizer steps only; no neural training or scoring."}

def run_original(code):
    namespace = {"__name__": "__main__"}
    exec(compile(code, "original-fence.py", "exec"), namespace)
    model, optimizer = namespace["model"], namespace["optimizer"]
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    listed = [p for _, p in parameters]
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert [id(p) for p in listed] == [id(p) for p in grouped]
    assert not optimizer.state
    assert all(p.grad is None for p in model.parameters())
    return model, optimizer, {"names_shapes_numel": [
        {"name": name, "shape": list(p.shape), "numel": p.numel()} for name, p in parameters],
        "trainable_elements": sum(p.numel() for p in listed),
        "optimizer_elements": sum(p.numel() for p in grouped),
        "same_parameter_objects_and_order": True, "optimizer_state_count": len(optimizer.state),
        "all_grad_none": True, "blocks": len(model.language.blocks)}

raw = (HERE / "fence-1.py").read_bytes()
model, optimizer, out["original"] = run_original(raw)
assert out["original"]["trainable_elements"] == 136
assert tuple(model.image_projector.weight.shape) == (8, 16)
before = {name: p.detach().clone() for name, p in model.named_parameters()}
assert all(torch.equal(p.detach(), before[name]) for name, p in model.named_parameters())
out["original"]["unchanged_without_forward_backward_step"] = True

needle = b"model.image_projector.requires_grad_(True)\n"
assert raw.count(needle) == 1
exercise = raw.replace(needle, needle + b"model.language.blocks[-1].requires_grad_(True)\n")
(HERE / "exercise-fence.py").write_bytes(exercise)
emodel, eoptimizer, out["exercise"] = run_original(exercise)
assert out["exercise"]["trainable_elements"] == 976
block_parts = {name: p.numel() for name, p in emodel.language.blocks[-1].named_parameters()}
out["exercise"]["block_parts"] = block_parts
out["exercise"]["block_elements"] = sum(block_parts.values())
assert sum(block_parts.values()) == 840
assert all(name.startswith(("language.blocks.0.", "image_projector.")) for name, p in emodel.named_parameters() if p.requires_grad)

# A second block changes [-1]'s index, rather than opening every block.
two_blocks = exercise.replace(b"ModelConfig(width=8)", b"ModelConfig(width=8, layers=2)")
_, _, out["two_blocks_variant"] = run_original(two_blocks)
assert out["two_blocks_variant"]["trainable_elements"] == 976
assert all(x["name"].startswith(("language.blocks.1.", "image_projector.")) for x in out["two_blocks_variant"]["names_shapes_numel"])

# Unfreezing after construction does not alter the saved optimizer objects.
model.language.blocks[-1].requires_grad_(True)
out["unfreeze_after_optimizer"] = {"current_trainable_elements": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "optimizer_elements": sum(p.numel() for g in optimizer.param_groups for p in g["params"])}
assert out["unfreeze_after_optimizer"] == {"current_trainable_elements": 976, "optimizer_elements": 136}

# Exact chain-rule example: y=(3*w+1)^2 at w=2 gives dy/dw=42.
fixed = nn.Linear(1, 1)
with torch.no_grad():
    fixed.weight.fill_(3)
    fixed.bias.fill_(1)
fixed.requires_grad_(False)
fixed.eval()
w = nn.Parameter(torch.tensor([[2.0]]))
y = fixed(w).square().sum()
y.backward()
with torch.no_grad():
    detached = fixed(w).square().sum()
out["frozen_chain"] = {"eval_training_flag": fixed.training, "loss": y.item(),
    "upstream_gradient": w.grad.item(), "frozen_parameter_grads_none": all(p.grad is None for p in fixed.parameters()),
    "normal_output_requires_grad": y.requires_grad, "no_grad_output_requires_grad": detached.requires_grad}
assert y.item() == 49 and w.grad.item() == 42
assert all(p.grad is None for p in fixed.parameters()) and not detached.requires_grad

def scalar_case(gradient, decay, frozen=False):
    p = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=decay)
    p.grad = None if gradient is None else torch.tensor([gradient], dtype=torch.float64)
    if frozen:
        p.requires_grad_(False)
    stale_preserved = p.grad is not None
    opt.step()
    return {"value": p.item(), "state_count": len(opt.state), "requires_grad": p.requires_grad,
            "gradient_tensor_preserved_by_freeze": stale_preserved}

out["none_gradient"] = scalar_case(None, 0.2)
out["zero_gradient_decay"] = scalar_case(0, 0.2)
out["stale_gradient_after_freeze"] = scalar_case(1, 0, frozen=True)
out["cleared_gradient_after_freeze"] = scalar_case(None, 0.2, frozen=True)
assert out["none_gradient"]["value"] == out["cleared_gradient_after_freeze"]["value"] == 2
assert abs(out["zero_gradient_decay"]["value"] - 1.96) < 1e-12
assert out["stale_gradient_after_freeze"]["value"] < 2

p = nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0)
p.grad = torch.ones_like(p)
opt.step()
first = p.item()
p.grad.zero_()
opt.step()
second = p.item()
before_step = opt.state[p]["step"].item()
before_moment = opt.state[p]["exp_avg"].clone()
p.grad = None
opt.step()
out["historical_direction"] = {"after_nonzero_gradient": first, "after_zero_gradient": second,
    "after_none_gradient": p.item(), "state_step_before_none": before_step,
    "state_step_after_none": opt.state[p]["step"].item(),
    "moment_unchanged_on_none": torch.equal(before_moment, opt.state[p]["exp_avg"])}
assert second < first and p.item() == second and opt.state[p]["step"].item() == before_step
assert torch.equal(before_moment, opt.state[p]["exp_avg"])

frozen_p = nn.Parameter(torch.tensor([2.0]))
active_p = nn.Parameter(torch.tensor([4.0]))
frozen_p.grad = torch.ones_like(frozen_p)
frozen_p.requires_grad_(False)
filtered = torch.optim.AdamW([p for p in [frozen_p, active_p] if p.requires_grad], lr=0.1)
active_p.grad = torch.zeros_like(active_p)
filtered.step()
out["excluded_parameter"] = {"frozen_value": frozen_p.item(), "optimizer_contains_frozen": any(p is frozen_p for g in filtered.param_groups for p in g["params"])}
assert frozen_p.item() == 2 and not out["excluded_parameter"]["optimizer_contains_frozen"]

out["hashes"] = {"original_fence": hashlib.sha256(raw).hexdigest(), "exercise_fence": hashlib.sha256(exercise).hexdigest()}
(HERE / "probe-results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False, indent=2))
