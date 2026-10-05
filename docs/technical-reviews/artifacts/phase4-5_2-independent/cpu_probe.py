"""Bounded CPU audit of section 5.2; synthetic logits only, no data/model download."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from tiny_perceptron.model import loss_sum

torch.set_num_threads(1)
torch.set_default_device("cpu")
BASE = Path(__file__).resolve().parent
DTYPE = torch.float64
ATOL = 1e-12
assert torch.version.cuda is None and not torch.cuda.is_available()
labels = torch.tensor([[1, 2, 3, 4], [1, -100, -100, -100]])
result = {"environment": {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu", "threads": str(torch.get_num_threads())}, "tolerances": {"double_absolute": ATOL, "original_float32_absolute": 1e-6}, "provenance": {"labels": labels.tolist(), "labels_units": "class IDs per [sample, position]; -100 is ignored", "logits_shape": [2, 4, 5], "logits_units": "unnormalized class scores", "sample_count": 2, "position_count": 8, "valid_target_count": 5, "candidate_count": 5, "model_used": "none", "optimizer_demo": "SGD on a 5-number synthetic shared logit vector; no tutorial model training"}}

# Re-execute precisely the original fence and inspect the objects left in its namespace.
namespace = {"__name__": "__main__"}
original = (BASE / "original" / "fence-1.py").read_bytes()
exec(compile(original, "5.2:original-fence-1", "exec"), namespace)
original_logits = namespace["logits"]
original_grad = original_logits.grad
expected_grad = torch.zeros_like(original_logits)
for sample, position in (labels != -100).nonzero().tolist():
    expected_grad[sample, position] = 0.2 / 5
    expected_grad[sample, position, labels[sample, position]] -= 1 / 5
assert torch.allclose(original_grad, expected_grad, atol=1e-7, rtol=0)
assert torch.equal(original_logits.detach(), torch.zeros_like(original_logits))
result["original_inspection"] = {"fence_sha256": hashlib.sha256(original).hexdigest(), "total": namespace["total"].item(), "count": namespace["count"].item(), "mean": namespace["mean_loss"].item(), "scalar_shapes": {key: list(namespace[key].shape) for key in ("count", "total", "mean_loss")}, "item_python_types": {key: type(namespace[key].item()).__name__ for key in ("count", "total", "mean_loss")}, "logits_leaf": original_logits.is_leaf, "mean_has_grad_fn": namespace["mean_loss"].grad_fn is not None, "count_requires_grad": namespace["count"].requires_grad, "logits_changed_by_backward": False, "valid_gradient_target": original_grad[0, 0, 1].item(), "valid_gradient_non_target": original_grad[0, 0, 0].item(), "ignored_gradient_max_abs": original_grad[labels == -100].abs().max().item(), "update_count": 0}

# The prescribed exercise also runs as the original float32 fence with only its label edited.
edited = original.replace(b"[1, -100, -100, -100]", b"[1, 2, -100, -100]")
assert edited != original
(BASE / "exercise-edited-fence.py").write_bytes(edited)
raw_exercise = []
for name, source, expected_count in (("edited", edited, 6), ("restored", original, 5)):
    ns = {"__name__": "__main__"}
    exec(compile(source, f"5.2:exercise-{name}", "exec"), ns)
    assert ns["count"].item() == expected_count
    assert abs(ns["total"].item() - expected_count * math.log(5)) < 1e-6
    assert abs(ns["mean_loss"].item() - math.log(5)) < 1e-6
    assert ns["logits"].shape[-1] == 5
    raw_exercise.append({"state": name, "fence_sha256": hashlib.sha256(source).hexdigest(), "count": ns["count"].item(), "total": ns["total"].item(), "mean": ns["mean_loss"].item(), "dtype": str(ns["logits"].dtype), "candidate_count": ns["logits"].shape[-1]})
result["original_dtype_exercise"] = raw_exercise

exercise = []
for new_label in (2, -100):
    variant_labels = labels.clone()
    variant_labels[1, 1] = new_label
    variant_logits = torch.zeros(2, 4, 5, dtype=DTYPE, requires_grad=True)
    total, count = loss_sum(variant_logits, variant_labels)
    mean = total / count
    mean.backward()
    n = 6 if new_label == 2 else 5
    assert count.item() == n
    assert abs(total.item() - n * math.log(5)) < ATOL
    assert abs(mean.item() - math.log(5)) < ATOL
    assert torch.equal(variant_logits.grad[variant_labels == -100], torch.zeros_like(variant_logits.grad[variant_labels == -100]))
    exercise.append({"changed_label": new_label, "count": n, "total": total.item(), "mean": mean.item(), "candidates": variant_logits.shape[-1]})
result["exercise_change_and_restore"] = exercise

# Same loss values can conceal different weights: compare per-position gradients.
token_logits = torch.zeros(2, 4, 5, dtype=DTYPE, requires_grad=True)
token_total, token_count = loss_sum(token_logits, labels)
(token_total / token_count).backward()
sample_logits = torch.zeros_like(token_logits, requires_grad=True)
terms = []
for sample in range(2):
    total, count = loss_sum(sample_logits[sample], labels[sample])
    terms.append(total / count)
sample_loss = torch.stack(terms).mean()
sample_loss.backward()
assert abs(token_logits.grad[0, 0, 1].item() - (-0.8 / 5)) < ATOL
assert abs(sample_logits.grad[0, 0, 1].item() - (-0.8 / 8)) < ATOL
assert abs(sample_logits.grad[1, 0, 1].item() - (-0.8 / 2)) < ATOL
assert not torch.allclose(token_logits.grad, sample_logits.grad, atol=ATOL, rtol=0)
nonuniform = torch.zeros(2, 4, 5, dtype=DTYPE)
nonuniform[1, 0, 1] = 2.0
nt, nc = loss_sum(nonuniform, labels)
ns = torch.stack([loss_sum(nonuniform[i], labels[i])[0] / loss_sum(nonuniform[i], labels[i])[1] for i in range(2)]).mean()
assert abs((nt / nc).item() - ns.item()) > 1e-4
result["weighting"] = {"token_weights": [1 / 5] * 5, "per_sample_weights": [1 / 8] * 4 + [1 / 2], "target_gradient_token_long_and_short": [token_logits.grad[0, 0, 1].item(), token_logits.grad[1, 0, 1].item()], "target_gradient_sample_long_and_short": [sample_logits.grad[0, 0, 1].item(), sample_logits.grad[1, 0, 1].item()], "uniform_token_mean": (token_total / token_count).item(), "uniform_sample_mean": sample_loss.item(), "nonuniform_token_mean": (nt / nc).item(), "nonuniform_sample_mean": ns.item(), "nonuniform_change": "Only short sample target logit [1,0,1] changed from 0 to 2; same labels, denominator, and candidate axis."}

# Gradient linearity requires the same parameter point and global denominator.
initial = torch.tensor([0.2, -0.1, 0.3, -0.2, 0.1], dtype=DTYPE)
batch = initial.clone().requires_grad_()
bt, bc = loss_sum(batch.expand(2, 4, 5), labels)
(bt / bc).backward()
micro = initial.clone().requires_grad_()
for sample in range(2):
    total, _ = loss_sum(micro.expand(4, 5), labels[sample])
    (total / bc).backward()
assert torch.allclose(batch.grad, micro.grad, atol=ATOL, rtol=0)
batch_before_step = batch.detach().clone()
micro_before_step = micro.detach().clone()
assert torch.equal(batch_before_step, initial) and torch.equal(micro_before_step, initial)
torch.optim.SGD([batch], lr=0.1).step()
torch.optim.SGD([micro], lr=0.1).step()
assert torch.allclose(batch, micro, atol=ATOL, rtol=0)
sequential_sample = initial.clone().requires_grad_()
optimizer = torch.optim.SGD([sequential_sample], lr=0.1)
for sample in range(2):
    optimizer.zero_grad(set_to_none=True)
    total, _ = loss_sum(sequential_sample.expand(4, 5), labels[sample])
    (total / bc).backward()
    optimizer.step()
sequential_answer = initial.clone().requires_grad_()
optimizer = torch.optim.SGD([sequential_answer], lr=0.1)
for target in labels[labels != -100]:
    optimizer.zero_grad(set_to_none=True)
    total, _ = loss_sum(sequential_answer[None], target[None])
    (total / bc).backward()
    optimizer.step()
assert not torch.allclose(batch, sequential_sample, atol=ATOL, rtol=0)
assert not torch.allclose(batch, sequential_answer, atol=ATOL, rtol=0)
result["accumulation_and_steps"] = {"shared_initial_logits": initial.tolist(), "global_valid_target_denominator": bc.item(), "batch_gradient": batch.grad.tolist(), "micro_gradient": micro.grad.tolist(), "gradient_max_abs_diff": (batch.grad - micro.grad).abs().max().item(), "unchanged_until_step": True, "single_batch_steps": 1, "two_micro_backward_single_step": 1, "single_step_final": batch.detach().tolist(), "accumulated_single_step_final": micro.detach().tolist(), "sample_immediate_steps": 2, "sample_immediate_final": sequential_sample.detach().tolist(), "answer_immediate_steps": 5, "answer_immediate_final": sequential_answer.detach().tolist(), "scope": "This checks arithmetic and the recomputation boundary only. It does not establish training quality or a preferred batch size."}

# Direct helper contract: the zero-denominator case is rejected before division.
try:
    loss_sum(torch.zeros(1, 1, 5), torch.full((1, 1), -100))
except ValueError as error:
    result["zero_valid_contract"] = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("loss_sum should reject all-ignored labels")
result["assertions"] = "All bounded CPU assertions passed. Original fence executed; label edit and restoration executed; sample/token weights and gradients checked; accumulation and immediate steps checked."
print(json.dumps(result, ensure_ascii=False, indent=2))
(BASE / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
