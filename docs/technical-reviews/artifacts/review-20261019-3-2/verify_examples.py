"""以獨立純量計算核對點積、正規化及練習中的變體。"""

import hashlib
import json
import math
import platform
import re
from pathlib import Path

import torch


def scalar_weights(scores):
    exponentials = [math.exp(value - max(scores)) for value in scores]
    total = math.fsum(exponentials)
    return [value / total for value in exponentials]


def verify(name, query, keys, expected_scores, rounded_weights=None):
    q = torch.tensor(query, dtype=torch.float32, device="cpu")
    k = torch.tensor(keys, dtype=torch.float32, device="cpu")
    scores = q @ k.T
    weights = scores.softmax(dim=0)
    scalar_scores = [math.fsum(a * b for a, b in zip(query, key, strict=True)) for key in keys]
    reference = scalar_weights(scalar_scores)
    assert scores.tolist() == expected_scores == scalar_scores
    assert k.T.shape == (2, 3)
    assert scores.shape == weights.shape == (3,)
    assert torch.equal(k.T, k.transpose(0, 1))
    assert torch.equal(scores, torch.stack([torch.dot(q, key) for key in k]))
    errors = [abs(observed - expected) for observed, expected in zip(weights.tolist(), reference, strict=True)]
    assert max(errors) <= 1e-7
    assert all(value > 0 for value in weights.tolist())
    assert abs(weights.sum().item() - 1.0) <= 2e-7
    if rounded_weights is not None:
        assert all(abs(a - b) <= 0.0005 for a, b in zip(reference, rounded_weights, strict=True))
    return {
        "name": name,
        "query": query,
        "keys": keys,
        "dtype": str(q.dtype),
        "device": str(q.device),
        "query_shape": list(q.shape),
        "keys_shape": list(k.shape),
        "transposed_keys_shape": list(k.T.shape),
        "scores": scores.tolist(),
        "weights_float32": weights.tolist(),
        "weights_scalar_reference": reference,
        "weights_sum_float32": weights.sum().item(),
        "max_absolute_error": max(errors),
        "highest_indices": [i for i, value in enumerate(reference) if value == max(reference)],
        "rounded_weights_in_text": rounded_weights,
    }


keys = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
results = [
    verify("原始需求", [1.0, 0.0], keys, [1.0, 0.0, -1.0], [0.665, 0.245, 0.090]),
    verify("改變需求", [0.0, 1.0], keys, [0.0, 1.0, 0.0]),
    verify("放大第一個候選", [1.0, 0.0], [[2.0, 0.0], *keys[1:]], [2.0, 0.0, -1.0]),
    verify("依序完成練習", [0.0, 1.0], [*keys[:2], [-1.0, 1.0]], [0.0, 1.0, 1.0], [0.155, 0.422, 0.422]),
]
assert results[0]["highest_indices"] == [0]
assert results[1]["highest_indices"] == [1]
assert results[3]["highest_indices"] == [1, 2]

chapter = Path("course/chapters/03.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", chapter))
selected = next(i for i, header in enumerate(headers) if header[0].startswith(b"## 3.2 "))
body = chapter[headers[selected].start():headers[selected + 1].start()]
frozen = Path("docs/course-review-20261019/freeze-01/sources/3.2.md").read_bytes()
assert body == frozen
fences = re.findall(rb"```python\n(.*?)```", body, re.S)
assert len(fences) == 1
assert fences[0] == Path(__file__).with_name("section-fence-1.py").read_bytes()
assert b"values" not in fences[0] and b"mask" not in fences[0] and b"sqrt" not in fences[0]
prior = chapter[headers[selected - 1].start():headers[selected].start()]
assert b"weights = torch.tensor([0.9, 0.1])" in prior
scaling = next(header.start() for header in headers if header[0].startswith(b"## 3.5 "))
masking = next(header.start() for header in headers if header[0].startswith(b"## 3.6 "))
next_section = next(header.start() for header in headers if header[0].startswith(b"## 3.7 "))
assert b"scores / math.sqrt(d)" in chapter[scaling:masking]
assert b"scores.masked_fill(~allowed" in chapter[masking:next_section]

output = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-2/verify_examples.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"},
    "source_sha256": hashlib.sha256(body).hexdigest(),
    "frozen_source_matches_current": True,
    "source_code_sha256": hashlib.sha256(fences[0]).hexdigest(),
    "source_checks": {"python_fences": 1, "preceding_weights_explicit": True, "has_value_readout": False, "has_scaling": False, "has_mask": False, "scaling_target_found": True, "masking_target_found": True},
    "comparison": {"score_tolerance": "精確相等", "probability_float32_absolute_tolerance": 1e-7, "probability_sum_absolute_tolerance": 2e-7, "printed_three_decimal_absolute_tolerance": 0.0005, "reference_formula": "s_j = sum_i(q_i*k_ji); p_j = exp(s_j-max(s))/sum_l(exp(s_l-max(s)))"},
    "results": results,
    "result": "全部斷言通過；原始需求、改變需求、放大候選與依序完成練習的結果均吻合。",
}
print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
