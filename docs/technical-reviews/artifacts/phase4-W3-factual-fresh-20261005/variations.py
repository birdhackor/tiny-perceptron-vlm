"""Bounded CPU verification of W.3; no training or network access."""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import torch

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
torch.set_num_threads(1)
scores = torch.tensor([[60.0, 70.0, 80.0], [80.0, 90.0, 100.0]])
original = scores.clone()
expected_rows = [Fraction(60 + 70 + 80, 3), Fraction(80 + 90 + 100, 3)]
expected_cols = [Fraction(60 + 80, 2), Fraction(70 + 90, 2), Fraction(80 + 100, 2)]
assert scores.mean(dim=1).tolist() == list(map(float, expected_rows))
assert scores.mean(dim=0).tolist() == list(map(float, expected_cols))
assert torch.equal(scores.mean(dim=-1), scores.mean(dim=1))
assert tuple(scores.mean(dim=1).shape) == (2,)
assert tuple(scores.mean(dim=0).shape) == (3,)
assert tuple(scores.shape) == (2, 3)
assert scores.tolist() == [[60.0, 70.0, 80.0], [80.0, 90.0, 100.0]]
assert scores[0].tolist() == [60.0, 70.0, 80.0]
assert type(scores[1, 2].item()) is float
assert scores[1, 2].item() == 100.0
assert torch.equal(scores, original)
flat = scores.reshape(6)
assert flat.tolist() == [60.0, 70.0, 80.0, 80.0, 90.0, 100.0]
assert torch.equal(flat.reshape(2, 3), scores)
failures = {}
for name, operation in {"multi_element_item": lambda: scores.item(), "reshape_eight": lambda: scores.reshape(8)}.items():
    try:
        operation()
    except RuntimeError as exc:
        failures[name] = {"type": type(exc).__name__, "message": str(exc)}
    else:
        raise AssertionError(f"Expected a RuntimeError: {name}")
changed = scores.clone()
changed[0, 2] = 110.0
assert changed.mean(dim=1).tolist() == [80.0, 90.0]
assert changed.mean(dim=0).tolist() == [70.0, 80.0, 105.0]
assert torch.equal(scores, original)
rank_examples = [torch.tensor(5.0), torch.tensor([5.0, 6.0]), scores]
assert [t.dim() for t in rank_examples] == [0, 1, 2]
three_axes = torch.arange(24).reshape(2, 4, 3)
assert tuple(three_axes[0].shape) == (4, 3)
assert three_axes[0].tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
# Move semantic labels with the data to show that dim numbers are not labels.
transposed = scores.T
assert torch.equal(transposed.mean(dim=0), scores.mean(dim=1))
assert torch.equal(transposed.mean(dim=1), scores.mean(dim=0))
print(json.dumps({
    "environment": {"python": sys.version, "torch": torch.__version__, "torch_git_revision": torch.version.git_version, "device": "cpu", "cuda_build": torch.version.cuda},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "original_row_means": scores.mean(dim=1).tolist(),
    "original_col_means": scores.mean(dim=0).tolist(),
    "denominators": {"per_student": 3, "per_quiz": 2},
    "units": "score points; arithmetic means retain the score unit",
    "changed_row_means": changed.mean(dim=1).tolist(),
    "changed_col_means": changed.mean(dim=0).tolist(),
    "flat": flat.tolist(), "shape_after_first_axis_index": list(three_axes[0].shape),
    "rank_examples": [t.dim() for t in rank_examples], "failure_contracts": failures,
    "nonmutation": True, "negative_dim_equivalent": True, "transposed_axis_semantics": True,
    "tolerance": "exact equality; all scores and stated means are exactly representable here",
    "status": "all assertions passed"
}, ensure_ascii=False, indent=2))
