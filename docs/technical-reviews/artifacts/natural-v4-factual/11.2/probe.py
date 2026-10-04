"""Bounded CPU checks performed by /root/v4_review_coordinator/factual_v4_11_2."""
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

import torch
from torch import nn
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import MultiModalLM

torch.set_num_threads(1)
torch.manual_seed(20261004)
base = Path(__file__).resolve().parent
root = Path.cwd()
source = (base / "source-original.md").read_text()
code = re.findall(r"```python\n(.*?)```", source, re.S)[0]
result = {
    "reviewer_task": "/root/v4_review_coordinator/factual_v4_11_2",
    "environment": {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_commit": torch.version.git_version, "device": "cpu",
        "cuda_available": str(torch.cuda.is_available()), "dtype": "float32",
        "threads": str(torch.get_num_threads()), "seed": "20261004",
    },
    "source_sha256": hashlib.sha256((base / "source-original.md").read_bytes()).hexdigest(),
}

def run_example(text):
    ns, out = {}, io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(text, "course/chapters/11.md#11.2", "exec"), ns)
    model, optimizer = ns["model"], ns["optimizer"]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    info = {
        "stdout": out.getvalue(), "trainable": ns["trainable"],
        "shapes": {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad},
        "optimizer_numel": sum(p.numel() for p in params),
        "optimizer_uses_same_objects": all(a is b for a, b in zip(params, ns["parameters_to_update"], strict=True)),
        "named_parameters_use_same_objects": all(any(p is q for q in params) for n, p in model.named_parameters() if p.requires_grad),
        "all_grad_none": all(p.grad is None for p in model.parameters()),
        "optimizer_state_entries": len(optimizer.state),
        "language_layers": len(model.language.blocks),
    }
    return model, optimizer, info

m, opt, main = run_example(code)
assert main["trainable"] == [("image_projector.weight", 128), ("image_projector.bias", 8)]
assert main["optimizer_numel"] == 136 and main["optimizer_state_entries"] == 0
assert main["all_grad_none"] and main["optimizer_uses_same_objects"] and main["named_parameters_use_same_objects"]
result["exact_lesson_code"] = main
variant = code.replace("trainable = []", "model.language.blocks[-1].requires_grad_(True)\ntrainable = []", 1)
mv, ov, exercise = run_example(variant)
assert exercise["optimizer_numel"] == 976 and exercise["language_layers"] == 1
added = [n for n, p in mv.named_parameters() if p.requires_grad and n.startswith("language.")]
assert added and all(n.startswith("language.blocks.0.") for n in added)
exercise["added_language_numel"] = sum(p.numel() for n, p in mv.named_parameters() if n in added)
assert exercise["added_language_numel"] == 840
result["exercise_code"] = exercise

def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}

def differences(model, before):
    return {n: float((p.detach() - before[n]).abs().max()) for n, p in model.named_parameters()}

def one_forward(model, no_grad_language=False):
    # A fixed synthetic feature input; no dataset or quality evaluation.
    feature = torch.arange(32, dtype=torch.float32).reshape(1, 2, 16) / 32
    projected = model.image_projector(feature)
    if no_grad_language:
        with torch.no_grad():
            logits = model.language(embeddings=projected)["logits"]
    else:
        logits = model.language(embeddings=projected)["logits"]
    return logits.square().mean()

m.eval()
before = snapshot(m)
loss = one_forward(m)
loss.backward()
grads = {n: None if p.grad is None else float(p.grad.abs().max()) for n, p in m.named_parameters()}
assert all(v is None for n, v in grads.items() if not n.startswith("image_projector."))
assert all(grads[n] > 0 for n in ("image_projector.weight", "image_projector.bias"))
assert all(v == 0 for v in differences(m, before).values())  # Backward alone cannot change parameters.
opt.step()
delta = differences(m, before)
assert all(v == 0 for n, v in delta.items() if not n.startswith("image_projector."))
assert all(delta[n] > 0 for n in ("image_projector.weight", "image_projector.bias"))
result["frozen_language_route"] = {
    "training_mode": m.training, "loss": float(loss.detach()), "loss_requires_grad": loss.requires_grad,
    "input_shape": [1, 2, 16], "logits_shape": [1, 2, 264], "mean_denominator": 528,
    "gradient_maxima": {n: v for n, v in grads.items() if v is not None},
    "frozen_parameters_with_grad": [n for n, v in grads.items() if v is not None and not n.startswith("image_projector.")],
    "changed_parameters": {n: v for n, v in delta.items() if v > 0},
    "unchanged_frozen_tensors": sum(v == 0 for n, v in delta.items() if not n.startswith("image_projector.")),
    "backward_only_parameter_delta": 0,
}
opt.zero_grad(set_to_none=True)
cut_loss = one_forward(m, no_grad_language=True)
assert not cut_loss.requires_grad
try:
    cut_loss.backward()
except RuntimeError as e:
    cut_error = str(e)
else:
    raise AssertionError("no_grad language route unexpectedly supported backward")
assert all(p.grad is None for p in m.image_projector.parameters())
result["no_grad_language_route"] = {"loss_requires_grad": cut_loss.requires_grad, "backward_error": cut_error, "projector_grad_none": True}

late, late_opt, _ = run_example(code)
late.language.blocks[-1].requires_grad_(True)
late_before = snapshot(late)
late_loss = one_forward(late)
late_loss.backward()
late_grads = {n: float(p.grad.abs().max()) for n, p in late.named_parameters() if n.startswith("language.blocks.0.") and p.grad is not None}
assert late_grads and any(v > 0 for v in late_grads.values())
late_opt.step()
late_delta = differences(late, late_before)
assert all(v == 0 for n, v in late_delta.items() if n.startswith("language.blocks.0."))
assert any(v > 0 for n, v in late_delta.items() if n.startswith("image_projector."))
result["late_unfreeze"] = {
    "trainable_numel_after_unfreeze": sum(p.numel() for p in late.parameters() if p.requires_grad),
    "optimizer_numel_after_unfreeze": sum(p.numel() for g in late_opt.param_groups for p in g["params"]),
    "language_block_grad_maxima": late_grads, "language_block_max_delta": max(v for n, v in late_delta.items() if n.startswith("language.blocks.0.")),
    "projector_max_delta": max(v for n, v in late_delta.items() if n.startswith("image_projector.")),
}
all_model = MultiModalLM(TinyLM(ModelConfig(width=8)))
all_opt = torch.optim.AdamW(all_model.parameters(), lr=0.001)
all_count = sum(p.numel() for g in all_opt.param_groups for p in g["params"])
all_model.requires_grad_(False)
all_model.image_projector.requires_grad_(True)
result["post_creation_freeze"] = {
    "optimizer_numel_before": all_count,
    "optimizer_numel_after": sum(p.numel() for g in all_opt.param_groups for p in g["params"]),
    "trainable_numel_after": sum(p.numel() for p in all_model.parameters() if p.requires_grad),
}
assert result["post_creation_freeze"]["optimizer_numel_after"] == all_count > 136

fixed = nn.Linear(2, 1)
with torch.no_grad():
    fixed.weight.copy_(torch.tensor([[2., -3.]]))
    fixed.bias.fill_(1.)
fixed.requires_grad_(False).eval()
x = torch.tensor([[4., 5.]], requires_grad=True)
y = fixed(x)
y.backward()
assert y.item() == -6 and torch.equal(x.grad, torch.tensor([[2., -3.]]))
assert fixed.weight.grad is None and fixed.bias.grad is None
result["hand_calculation_chain"] = {"y": y.item(), "x_gradient": x.grad.tolist(), "fixed_weight_grad": None, "fixed_bias_grad": None}

zero, unused = nn.Parameter(torch.tensor(1.)), nn.Parameter(torch.tensor(1.))
zero_opt = torch.optim.AdamW([zero, unused], lr=0.001)
(zero * 0).backward()
assert zero.grad.item() == 0 and unused.grad is None
zero_opt.step()
result["zero_vs_none"] = {"zero_grad": zero.grad.item(), "unused_grad": unused.grad, "zero_param_after": zero.item(), "unused_param_after": unused.item(), "weight_decay": zero_opt.param_groups[0]["weight_decay"]}
assert zero.item() < 1 and unused.item() == 1

# Scope check: disabling autograd does not clear an existing gradient, and an
# optimizer reads .grad rather than requiring requires_grad=True at step time.
stale = nn.Parameter(torch.tensor(1.))
stale_opt = torch.optim.AdamW([stale], lr=0.001)
(stale.square()).backward()
stale.requires_grad_(False)
stale_opt.step()
result["stale_gradient_scope"] = {"requires_grad": stale.requires_grad, "retained_grad": stale.grad.item(), "parameter_after_step": stale.item()}
assert stale.item() < 1 and stale.grad.item() == 2

assert platform.python_version() == "3.13.5" and torch.__version__ == "2.14.1+cpu"
(base / "probe-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("ALL BOUNDED CPU ASSERTIONS PASSED")
