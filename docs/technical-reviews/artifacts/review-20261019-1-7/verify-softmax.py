"""以獨立高精度計算與 CPU tensor 核對候選分數換算。"""

import contextlib
import io
import json
import platform
import sys
from decimal import Decimal, localcontext

import torch


torch.set_default_device("cpu")
torch.set_num_threads(1)

with localcontext() as context:
    context.prec = 50
    scores = [Decimal(1), Decimal(2), Decimal(-1)]
    exponentials = [score.exp() for score in scores]
    denominator = sum(exponentials)
    probabilities = [weight / denominator for weight in exponentials]
    shifted_scores = [score - max(scores) for score in scores]
    shifted_weights = [score.exp() for score in shifted_scores]
    exercise_scores = [Decimal(3), Decimal(2), Decimal(-1)]
    exercise_weights = [score.exp() for score in exercise_scores]
    exercise_denominator = sum(exercise_weights)
    exercise_probabilities = [weight / exercise_denominator for weight in exercise_weights]
    decimal_results = {
        "precision": context.prec,
        "raw_sum": str(sum(scores)),
        "raw_has_negative": any(score < 0 for score in scores),
        "e": str(Decimal(1).exp()),
        "exponentials": [str(value) for value in exponentials],
        "denominator": str(denominator),
        "probabilities": [str(value) for value in probabilities],
        "shifted_scores": [str(value) for value in shifted_scores],
        "shifted_weights": [str(value) for value in shifted_weights],
        "exercise_denominator": str(exercise_denominator),
        "exercise_probabilities": [str(value) for value in exercise_probabilities],
        "second_probability_decreases": exercise_probabilities[1] < probabilities[1],
    }
    assert abs(Decimal("2.718") - Decimal(1).exp()) < Decimal("0.0005")
    assert all(
        abs(value - expected) < Decimal("0.0005")
        for value, expected in zip(exponentials, map(Decimal, ["2.718", "7.389", "0.368"]))
    )
    assert abs(denominator - Decimal("10.475")) < Decimal("0.0005")
    assert all(
        abs(value - expected) < Decimal("0.00005")
        for value, expected in zip(probabilities, map(Decimal, ["0.2595", "0.7054", "0.0351"]))
    )
    assert all(
        abs(value - expected) < Decimal("0.0005")
        for value, expected in zip(shifted_weights, map(Decimal, ["0.368", "1", "0.050"]))
    )
    assert all(
        abs(value - expected) < Decimal("0.00005")
        for value, expected in zip(exercise_probabilities, map(Decimal, ["0.7214", "0.2654", "0.0132"]))
    )

z = torch.tensor([1.0, 2.0, -1.0])
original_z = z.clone()
weights = (z - z.max()).exp()
p = weights / weights.sum()
builtin = z.softmax(dim=0)
shifted_builtin = (z + 100).softmax(dim=0)
z64 = z.to(torch.float64)
p64 = z64.softmax(dim=0)
batch = torch.tensor([[1.0, 2.0, -1.0], [3.0, 2.0, -1.0]])
batch_probabilities = batch.softmax(dim=-1)
batch_independent = torch.stack([row.softmax(dim=0) for row in batch])
exercise = batch_probabilities[1]
wrong_highest = torch.tensor([1.0, 2.0, 3.0]).softmax(dim=0)

assert p.dtype == torch.float32
assert torch.allclose(p, builtin)
assert torch.allclose(p, shifted_builtin)
assert torch.equal(batch_probabilities, batch_independent)
assert torch.allclose(batch_probabilities.sum(dim=-1), torch.ones(2))
assert torch.allclose(batch.softmax(dim=1), batch_probabilities)
assert torch.max(torch.abs(p64 - torch.tensor([float(v) for v in probabilities], dtype=torch.float64))).item() < 1e-15
assert exercise[1] < p[1]
assert wrong_highest.argmax().item() == 2
assert torch.equal(z, original_z)

assert_stdout = io.StringIO()
assert_stderr = io.StringIO()
with contextlib.redirect_stdout(assert_stdout), contextlib.redirect_stderr(assert_stderr):
    assert torch.allclose(p, builtin)

item_error = None
try:
    p.item()
except RuntimeError as error:
    item_error = type(error).__name__
assert item_error == "RuntimeError"

near = torch.allclose(torch.tensor([1.0 + 5e-6]), torch.tensor([1.0]))
far = torch.allclose(torch.tensor([1.0 + 2e-5]), torch.tensor([1.0]))
assert near and not far
assert abs(p.sum().item() - 1.0) <= torch.finfo(p.dtype).eps
assert torch.isfinite(shifted_builtin).all().item()

record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-7/verify-softmax.py",
    "result": "全部斷言通過；本次只執行確定性數值與 API 核對。",
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": str(z.device),
        "dtype": str(z.dtype),
        "default_dtype": str(torch.get_default_dtype()),
        "debug": str(__debug__),
    },
    "decimal": decimal_results,
    "tensor": {
        "weights": weights.tolist(),
        "manual_probabilities": p.tolist(),
        "builtin_probabilities": builtin.tolist(),
        "manual_matches_builtin": torch.allclose(p, builtin),
        "sum": p.sum().item(),
        "sum_error": p.sum().item() - 1.0,
        "float32_epsilon": torch.finfo(p.dtype).eps,
        "float64_probabilities": p64.tolist(),
        "float64_sum": p64.sum().item(),
        "shifted_builtin": shifted_builtin.tolist(),
        "shift_max_absolute_difference": torch.max(torch.abs(builtin - shifted_builtin)).item(),
        "unshifted_exp_of_plus_100_is_finite": (z + 100).exp().isfinite().tolist(),
        "batch_shape": list(batch.shape),
        "batch_probabilities": batch_probabilities.tolist(),
        "batch_row_sums": batch_probabilities.sum(dim=-1).tolist(),
        "batch_equals_independent_rows": torch.equal(batch_probabilities, batch_independent),
        "exercise_probabilities": exercise.tolist(),
        "wrong_highest_probabilities": wrong_highest.tolist(),
        "wrong_highest_index": wrong_highest.argmax().item(),
        "item_return_type": type(p.sum().item()).__name__,
        "single_element_vector_item": torch.tensor([2.0]).item(),
        "multi_element_item_error_type": item_error,
        "allclose_near": near,
        "allclose_far": far,
        "assert_stdout": assert_stdout.getvalue(),
        "assert_stderr": assert_stderr.getvalue(),
        "input_unchanged": torch.equal(z, original_z),
        "requires_grad": z.requires_grad,
        "grad_is_none": z.grad is None,
    },
}
print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
