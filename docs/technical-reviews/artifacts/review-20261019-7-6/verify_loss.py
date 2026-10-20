import hashlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch
from tiny_perceptron.model import loss_sum, masked_loss

torch.set_num_threads(1)
torch.set_default_device("cpu")
a = torch.tensor([[[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
labels = torch.tensor([[1, 2]])
b = torch.cat([a, torch.zeros(1, 3, 4)], dim=1)
extended = torch.tensor([[1, 2, IGNORE, IGNORE, IGNORE]])
first_probability = math.exp(2) / (math.exp(2) + 3)
first_loss = -math.log(first_probability)
uniform_loss = math.log(4)
expected_mean = (first_loss + uniform_loss) / 2
expected_changed_mean = (first_loss + 2 * uniform_loss) / 3
before_sum, before_count = loss_sum(a, labels)
after_sum, after_count = loss_sum(b, extended)
before, after = masked_loss(a, labels), masked_loss(b, extended)
probabilities = a.softmax(dim=-1)
per_position = F.cross_entropy(b.reshape(-1, 4), extended.reshape(-1), reduction="none")
assert torch.equal(before, after)
assert before_count.item() == after_count.item() == 2
assert torch.equal(before_sum, after_sum)
assert abs(before.item() - expected_mean) < 5e-7
assert abs(probabilities[0, 0, 1].item() - first_probability) < 1e-7
assert probabilities[0, 1].tolist() == [0.25] * 4

with_grad = b.clone().requires_grad_()
masked_loss(with_grad, extended).backward()
ignored_gradient = with_grad.grad[0, 2:].clone()
assert torch.equal(ignored_gradient, torch.zeros_like(ignored_gradient))
variations = [
    torch.tensor([[[90.0, -90.0, 1.0, 2.0], [-6.0, 1.0, 80.0, 3.0], [1.0, 2.0, 3.0, 4.0]]]),
    torch.tensor([[[-10000.0, 10000.0, 5.0, -5.0]]]).expand(1, 3, 4),
]
variation_results = []
for tail in variations:
    alternate = torch.cat([a, tail], dim=1).requires_grad_()
    value = masked_loss(alternate, extended)
    value.backward()
    assert torch.equal(value, before)
    assert torch.equal(alternate.grad[:, 2:], torch.zeros_like(alternate.grad[:, 2:]))
    variation_results.append({"tail_logits": tail.tolist(), "mean": value.item(), "ignored_gradient_zero": True})

changed = extended.clone()
changed[0, 2] = 0
changed_logits = b.clone().requires_grad_()
changed_loss = masked_loss(changed_logits, changed)
changed_loss.backward()
changed_sum, changed_count = loss_sum(b, changed)
assert changed_count.item() == 3
assert abs(changed_loss.item() - expected_changed_mean) < 5e-7
assert changed_logits.grad[0, 2, 0].item() < 0
assert (changed_logits.grad[0, 2, 1:] > 0).all().item()
assert torch.equal(changed_logits.grad[0, 3:], torch.zeros_like(changed_logits.grad[0, 3:]))
assert_failure = False
try:
    assert torch.allclose(before, changed_loss)
except AssertionError:
    assert_failure = True
assert assert_failure
changed[0, 2] = IGNORE
restored = masked_loss(b, changed)
assert torch.equal(restored, before)

all_ignored_error = None
try:
    masked_loss(b, torch.full_like(extended, IGNORE))
except ValueError as error:
    all_ignored_error = {"type": type(error).__name__, "message": str(error)}
assert all_ignored_error is not None
batch_x, batch_y, valid = pad_batch([
    (torch.tensor([1, 3]), torch.tensor([3, 2])),
    (torch.tensor([1]), torch.tensor([2])),
])
assert ByteTokenizer.pad_id == 0 and IGNORE == -100
assert batch_x.tolist() == [[1, 3], [1, 0]]
assert batch_y.tolist() == [[3, 2], [2, -100]]
assert valid.tolist() == [[True, True], [True, False]]

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-6/verify_loss.py",
    "environment": {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version), "device": str(a.device),
        "logits_dtype": str(a.dtype), "labels_dtype": str(labels.dtype),
        "threads": str(torch.get_num_threads()),
    },
    "implementation": {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in ("tiny_perceptron/model.py", "tiny_perceptron/data.py")
    },
    "derivation": {
        "first_probability_formula": "exp(2)/(exp(2)+3)", "first_probability": first_probability,
        "first_loss_formula": "-ln(first_probability)", "first_loss": first_loss,
        "uniform_probability": 0.25, "uniform_loss_formula": "ln(4)", "uniform_loss": uniform_loss,
        "mean_formula": "(first_loss+ln(4))/2", "mean": expected_mean,
        "changed_mean_formula": "(first_loss+2*ln(4))/3", "changed_mean": expected_changed_mean,
        "cross_entropy_logit_derivative": "(softmax(z)[j] - 1[j=target])/effective_count",
    },
    "baseline": {
        "shape_a": list(a.shape), "shape_b": list(b.shape), "labels": labels.tolist(),
        "extended": extended.tolist(), "probabilities": probabilities.tolist(),
        "per_position_loss": per_position.tolist(), "before_sum": before_sum.item(),
        "after_sum": after_sum.item(), "before_count": before_count.item(), "after_count": after_count.item(),
        "before": before.item(), "after": after.item(), "equal": torch.equal(before, after),
        "ignored_gradient": ignored_gradient.tolist(),
    },
    "ignored_finite_logit_variations": variation_results,
    "exercise": {
        "sum": changed_sum.item(), "effective_count": changed_count.item(), "mean": changed_loss.item(),
        "added_target_gradient": changed_logits.grad[0, 2].tolist(),
        "remaining_ignored_gradient": changed_logits.grad[0, 3:].tolist(),
        "original_assert_fails": assert_failure, "restored_mean": restored.item(),
        "original_two_logit_rows_unchanged": torch.equal(changed_logits[:, :2], a),
    },
    "all_ignored": all_ignored_error,
    "padding_contract": {
        "pad_id": ByteTokenizer.pad_id, "ignore_index": IGNORE,
        "x": batch_x.tolist(), "y": batch_y.tolist(), "valid": valid.tolist(),
        "masked_loss_signature": str(inspect.signature(masked_loss)),
    },
    "result": "所有驗證斷言通過。",
}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
