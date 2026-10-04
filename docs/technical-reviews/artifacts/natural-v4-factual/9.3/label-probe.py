"""Bounded target semantics probe; does not train a model."""
import math
import sys
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
# Class0 denotes answer6; class1 denotes stale answer4 for the new3+3 prompt.
logits = torch.tensor([[3.0, -3.0]], dtype=torch.float64, requires_grad=True)
correct = F.cross_entropy(logits, torch.tensor([0]))
stale = F.cross_entropy(logits, torch.tensor([1]))
correct_expected = math.log1p(math.exp(-6.0))
stale_expected = 6.0 + correct_expected
assert abs(correct.item() - correct_expected) < 1e-12
assert abs(stale.item() - stale_expected) < 1e-12
gradient, = torch.autograd.grad(stale, logits)
assert gradient[0, 1].item() < 0 and gradient[0, 0].item() > 0
print("ENVIRONMENT", {"python": sys.version, "torch": torch.__version__, "device": "cpu", "dtype": "float64", "seed": "not applicable: deterministic explicit input"})
print("INPUT", {"shape": [1, 2], "logits": [3.0, -3.0], "classes": ["answer6", "stale_answer4"], "prompt": "3+3", "correct_target": 0, "stale_target": 1, "examples": 1, "updates": 0})
print("CORRECT_LOSS", correct.item(), "HAND_FORMULA", correct_expected)
print("STALE_LOSS", stale.item(), "HAND_FORMULA", stale_expected)
print("STALE_TARGET_GRADIENT", gradient.tolist())
print("PASS: loss follows supplied target; stale label would favor wrong class, without arithmetic checking; no optimizer or training executed")
