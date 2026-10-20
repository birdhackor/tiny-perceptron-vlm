import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from torch.nn import functional as F


def real(value):
    return value.detach().cpu().tolist()


def complex_values(value):
    return [[[float(z.real), float(z.imag)] for z in row] for row in value.tolist()]


torch.set_default_device("cpu")
image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
text = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
normal_image = F.normalize(image, dim=-1)
normal_text = F.normalize(text, dim=-1)
similarity = normal_image @ normal_text.T
swapped = F.normalize(torch.tensor([[0.0, 3.0], [2.0, 0.0]]), dim=-1)
swapped_similarity = normal_image @ swapped.T
v = torch.tensor([3.0, 4.0])
unit_v = F.normalize(v, dim=-1)
duplicates = F.normalize(torch.tensor([[1.0, 1.0], [1.0, 1.0]]), dim=-1)
ambiguous = normal_image @ duplicates.T
integer_input = torch.tensor([[0, 3], [2, 0]])
try:
    F.normalize(integer_input, dim=-1)
except RuntimeError as error:
    integer_result = {"raises": type(error).__name__, "message": str(error)}
else:
    integer_result = {"raises": None}
complex_input = torch.tensor([[3.0 + 4.0j, 0.0j]], dtype=torch.complex64)
complex_output = F.normalize(complex_input, dim=-1)
zero_and_small = torch.tensor([[0.0, 0.0], [1e-13, 0.0]], dtype=torch.float64)
layer = torch.nn.LayerNorm(2)
layer_output = layer(v)
with torch.no_grad():
    layer.weight.fill_(2.0)
    layer.bias.fill_(1.0)
layer_affine_output = layer(v)
endpoints = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]) @ image[0]
source_path = Path("docs/course-review-20261019/freeze-01/sources/10.9.md")
freeze_path = Path("docs/course-review-20261019/freeze-04.json")
results = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git": str(torch.version.git_version), "device": "cpu", "default_dtype": str(torch.get_default_dtype())},
    "frozen_input": {"source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest()},
    "original": {"image_shape": list(image.shape), "text_shape": list(text.shape), "normalized_image": real(normal_image), "normalized_text": real(normal_text), "similarity": real(similarity), "selected_columns": similarity.argmax(-1).tolist()},
    "length_example": {"length": float(torch.linalg.vector_norm(v)), "normalized": real(unit_v), "unit_length": float(torch.linalg.vector_norm(unit_v))},
    "direction_scores": real(endpoints),
    "positive_rescaling": {"normalized": real(F.normalize(7.0 * v, dim=-1)), "unchanged": bool(torch.allclose(F.normalize(7.0 * v, dim=-1), unit_v))},
    "candidate_duplication": {"normalized_candidates": real(duplicates), "scores": real(ambiguous), "row_sums": real(ambiguous.sum(-1)), "softmax_row_sums": real(ambiguous.softmax(-1).sum(-1))},
    "swapped_text": {"similarity": real(swapped_similarity), "selected_columns": swapped_similarity.argmax(-1).tolist()},
    "integer_input": {"dtype": str(integer_input.dtype), **integer_result},
    "complex_input": {"dtype": str(complex_input.dtype), "is_floating_point": complex_input.is_floating_point(), "is_complex": complex_input.is_complex(), "output_real_imag": complex_values(complex_output), "output_norms": real(torch.linalg.vector_norm(complex_output, dim=-1))},
    "normalization_boundary": {"inputs": real(zero_and_small), "outputs": real(F.normalize(zero_and_small, dim=-1)), "output_norms": real(torch.linalg.vector_norm(F.normalize(zero_and_small, dim=-1), dim=-1)), "default_eps": 1e-12},
    "layernorm_comparison": {"input": real(v), "default_output": real(layer_output), "affine_output": real(layer_affine_output), "l2_output": real(unit_v)}
}
assert torch.equal(similarity, torch.eye(2))
assert similarity.argmax(-1).tolist() == [0, 1]
assert torch.equal(swapped_similarity, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
assert swapped_similarity.argmax(-1).tolist() == [1, 0]
assert torch.allclose(unit_v, torch.tensor([0.6, 0.8]), atol=1e-7)
assert torch.equal(endpoints, torch.tensor([1.0, 0.0, -1.0]))
assert torch.allclose(ambiguous, torch.full((2, 2), 1.0 / math.sqrt(2.0)), atol=1e-7)
assert integer_result["raises"] == "RuntimeError"
assert not complex_input.is_floating_point() and complex_input.is_complex()
assert torch.allclose(torch.linalg.vector_norm(complex_output, dim=-1), torch.ones(1))
assert not torch.allclose(layer_output, unit_v)
print(json.dumps(results, ensure_ascii=False, indent=2))
