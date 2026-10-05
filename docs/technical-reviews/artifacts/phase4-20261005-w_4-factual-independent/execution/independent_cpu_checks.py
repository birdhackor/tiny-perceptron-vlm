"""W.4 reviewer-owned bounded CPU checks; no training or model downloads."""
import ast
import hashlib
import json
import math
import sys
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_default_device("cpu")
evidence = Path(__file__).resolve().parents[1]
root = evidence.parents[3]
assert torch.version.cuda is None and not torch.cuda.is_available()
cards = Counter(["cat", "cat", "cat", "dog"])
card_probability = {name: Fraction(count, sum(cards.values())) for name, count in cards.items()}
assert card_probability == {"cat": Fraction(3, 4), "dog": Fraction(1, 4)}
assert sum(card_probability.values()) == 1

rows = []
with localcontext() as context:
    context.prec = 60
    for text in ["0.9", "0.5", "0.1", "0.25"]:
        independent = -Decimal(text).ln()
        observed = -math.log(float(text))
        assert abs(Decimal.from_float(observed) - independent) < Decimal("1e-14")
        rounded_reference = independent.quantize(Decimal("0.001"))
        assert Decimal(str(round(observed, 3))) == rounded_reference
        assert abs(observed - float(rounded_reference)) <= 0.0005
        rows.append({"p": text, "decimal60_reference": str(independent),
                     "math_log_observed": observed, "round3": round(observed, 3)})
    inverse_e = (-Decimal(1)).exp()
    assert abs(Decimal.from_float(math.e ** -1) - inverse_e) < Decimal("1e-15")
    assert round(math.e ** -1, 3) == float(inverse_e.quantize(Decimal("0.001")))

grid = [0.000001, 0.1, 0.25, 0.5, 0.9, 1.0]
costs = [-math.log(p) for p in grid]
assert all(a > b for a, b in zip(costs, costs[1:]))
assert costs[-1] == 0.0 and all(v >= 0 for v in costs)
assert math.isclose(math.log(math.e ** 0.5), 0.5, abs_tol=1e-15)
assert math.e ** 2 == math.e * math.e
try:
    math.log(0.0)
except ValueError as error:
    zero_error = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("math.log(0) must raise ValueError")

independent_outcomes = [(a, b) for a in [False, True] for b in [False, True]]
p_first = Fraction(sum(a for a, b in independent_outcomes), len(independent_outcomes))
p_second = Fraction(sum(b for a, b in independent_outcomes), len(independent_outcomes))
p_joint = Fraction(sum(a and b for a, b in independent_outcomes), len(independent_outcomes))
assert p_first == p_second == Fraction(1, 2)
assert p_joint == p_first * p_second == Fraction(1, 4)
dependent_outcomes = [(False, False), (True, True)]
dependent_joint = Fraction(sum(a and b for a, b in dependent_outcomes), len(dependent_outcomes))
assert dependent_joint == Fraction(1, 2) != p_first * p_second
identity_checks = []
for p1, p2 in [(0.5, 0.5), (0.9, 0.1), (0.25, 0.9)]:
    joint_cost, sum_cost = -math.log(p1 * p2), -math.log(p1) - math.log(p2)
    assert math.isclose(joint_cost, sum_cost, rel_tol=0, abs_tol=1e-14)
    identity_checks.append({"p1": p1, "p2": p2, "product": p1 * p2,
                            "joint_cost": joint_cost, "sum_cost": sum_cost})
assert round(-math.log(float(p_joint)), 3) == 1.386

logits = torch.tensor([[math.log(9.0), 0.0], [0.0, math.log(9.0)]], dtype=torch.float64)
targets = torch.tensor([0, 0], dtype=torch.long)
probabilities = F.softmax(logits, dim=1)
assert torch.allclose(probabilities, torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64), atol=1e-14, rtol=0)
assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-14, rtol=0)
reference_costs = torch.tensor([-float(Decimal("0.9").ln()), -float(Decimal("0.1").ln())], dtype=torch.float64)
unreduced = F.cross_entropy(logits, targets, reduction="none")
mean_cost = F.cross_entropy(logits, targets)
sum_cost = F.cross_entropy(logits, targets, reduction="sum")
assert torch.allclose(unreduced, reference_costs, atol=1e-14, rtol=0)
assert torch.allclose(mean_cost, reference_costs.sum() / 2, atol=1e-14, rtol=0)
assert torch.allclose(sum_cost, reference_costs.sum(), atol=1e-14, rtol=0)
assert torch.allclose(F.cross_entropy(logits + 500.0, targets), mean_cost, atol=1e-13, rtol=0)
probability_input_cost = F.cross_entropy(probabilities, targets)
assert not torch.allclose(probability_input_cost, mean_cost, atol=1e-14, rtol=0)

official_source = evidence / "sources/pytorch-functional.py"
installed_source = Path(F.__file__)
assert official_source.read_bytes() == installed_source.read_bytes()
tree = ast.parse(official_source.read_text())
function_locators = {n.name: [n.lineno, n.end_lineno] for n in tree.body
                     if isinstance(n, ast.FunctionDef) and n.name in ["softmax", "cross_entropy"]}
print(json.dumps({
    "environment": {"python": sys.version, "executable": sys.executable,
                    "torch": torch.__version__, "torch_git_version": torch.version.git_version,
                    "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device": "cpu"},
    "card_probabilities": {k: str(v) for k, v in card_probability.items()}, "card_denominator": sum(cards.values()),
    "numeric_rows": rows, "e": math.e, "e_inverse": math.e ** -1,
    "log_inverse_half_exponent": math.log(math.e ** 0.5),
    "monotonic_p_grid": grid, "cost_grid": costs, "log_zero": zero_error,
    "independent_joint_probability": str(p_joint), "independent_outcome_denominator": len(independent_outcomes),
    "dependent_joint_probability": str(dependent_joint), "dependent_outcome_denominator": len(dependent_outcomes),
    "product_identity": identity_checks,
    "software": {"logits_shape": list(logits.shape), "class_axis": 1,
                 "target_shape": list(targets.shape), "probabilities": probabilities.tolist(),
                 "unreduced_costs": unreduced.tolist(), "mean_denominator": 2,
                 "mean_cost": mean_cost.item(), "sum_cost": sum_cost.item(),
                 "probability_input_cost": probability_input_cost.item(),
                 "same_constant_shift_cost": F.cross_entropy(logits + 500.0, targets).item(),
                 "parameters_updated": False},
    "official_installed_source_equality": True,
    "official_source_sha256": hashlib.sha256(official_source.read_bytes()).hexdigest(),
    "function_locators": function_locators,
    "all_assertions_passed": True
}, indent=2, ensure_ascii=False, allow_nan=False))
