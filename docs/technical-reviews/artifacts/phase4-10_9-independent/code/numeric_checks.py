"""Bounded CPU checks of the 10.9 claims; no models, training or checkpoint writes."""
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from torch.nn import functional as F

torch.set_num_threads(1)
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_default_device("cpu")
torch.manual_seed(42)
records = {}

def record(name, value):
    records[name] = value.tolist() if isinstance(value, torch.Tensor) else value

def similarity(image, text):
    return F.normalize(image, dim=-1) @ F.normalize(text, dim=-1).T

image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
text = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
original = similarity(image, text)
assert torch.equal(original, torch.eye(2))
assert original.argmax(-1).tolist() == [0, 1]
record("original", original)
record("original_argmax", original.argmax(-1))
swapped = similarity(image, text.flip(0))
assert torch.equal(swapped, torch.tensor([[0., 1.], [1., 0.]]))
assert swapped.argmax(-1).tolist() == [1, 0]
record("swapped", swapped)
record("swapped_argmax", swapped.argmax(-1))

v = torch.tensor([3.0, 4.0])
assert v.norm().item() == 5
assert torch.allclose(F.normalize(v, dim=-1), torch.tensor([0.6, 0.8]), atol=1e-7, rtol=0)
record("length_3_4", v.norm())
record("normalized_3_4", F.normalize(v, dim=-1))
directions = similarity(torch.tensor([[1., 0.]]), torch.tensor([[2., 0.], [0., 3.], [-2., 0.]]))
assert torch.equal(directions, torch.tensor([[1., 0., -1.]]))
record("same_perpendicular_opposite", directions)
scaled = similarity(image * torch.tensor([[7.], [0.25]]), text * torch.tensor([[0.125], [12.]]))
assert torch.equal(scaled, original)
record("positive_rescaling", scaled)

two_candidates = similarity(torch.tensor([[1., 1.]]), torch.tensor([[1., 0.], [0., 1.]]))
assert torch.allclose(two_candidates, torch.full((1, 2), 1 / math.sqrt(2)), atol=1e-7, rtol=0)
assert abs(two_candidates.sum().item() - math.sqrt(2)) < 1e-7
record("two_equal_candidates", two_candidates)
record("candidate_score_sum", two_candidates.sum())
record("candidate_softmax", F.softmax(two_candidates, dim=-1))
assert torch.allclose(F.softmax(two_candidates, dim=-1), torch.tensor([[0.5, 0.5]]), atol=1e-7, rtol=0)
record("tie_argmax", two_candidates.argmax(-1))

# Nonsquare candidates establish row=image, column=text independently of 2x2 symmetry.
assert directions.shape == (1, 3)
assert directions.argmax(-1).tolist() == [0]
record("row_column_shapes", {"image": [1, 2], "text": [3, 2], "text_transpose": [2, 3], "scores": list(directions.shape)})
try:
    F.normalize(torch.tensor([[1, 0], [0, 1]]), dim=-1)
except RuntimeError as error:
    record("integer_normalize_error", str(error))
else:
    raise AssertionError("Integer normalize unexpectedly succeeded")

# Equal width alone imposes no semantic pairing; this seeded random counterexample is not a benchmark.
random_scores = similarity(torch.randn(2, 2), torch.randn(2, 2))
record("independent_random_features", random_scores)
record("random_argmax", random_scores.argmax(-1))
assert random_scores.argmax(-1).tolist() != [0, 1]
assert not image.requires_grad and not text.requires_grad and not original.requires_grad
record("manual_scope", {"encoder_executed": False, "backward_called": False, "optimizer_steps": 0, "original_requires_grad": original.requires_grad})
record("zero_vector_api_boundary", F.normalize(torch.zeros(1, 2), dim=-1))

print(json.dumps({"environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "device": "cpu", "threads": torch.get_num_threads()}, "records": records}, indent=2, ensure_ascii=False, allow_nan=False))
