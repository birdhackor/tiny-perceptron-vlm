"""以獨立計數、精確分數與 CPU 執行核對接字表算例。"""

import hashlib
import json
import math
import platform
from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


text = "貓看狗。狗吃魚。貓看鳥。"
chars = sorted(set(text))
pairs = list(zip(text[:-1], text[1:], strict=True))
observed = Counter(pairs)
row_observations = Counter(current for current, _ in pairs)
assert chars == ["。", "吃", "狗", "看", "貓", "魚", "鳥"]
assert len(text) == 12 and len(pairs) == 11
assert observed["貓", "看"] == row_observations["貓"] == 2
ids = [chars.index(char) for char in text]
assert all(chars[index] == char for index, char in zip(ids, text, strict=True))

results = {}
for alpha in [Fraction(1), Fraction(1, 10)]:
    counts = (
        torch.ones(len(chars), len(chars))
        if alpha == 1
        else torch.full((len(chars), len(chars)), 0.1)
    )
    initial = counts.clone()
    for current, answer in zip(ids[:-1], ids[1:], strict=True):
        counts[current, answer] += 1
    totals = counts.sum(dim=1, keepdim=True)
    probabilities = counts / totals
    exact_counts = [
        [Fraction(observed[current, answer]) + alpha for answer in chars]
        for current in chars
    ]
    exact_totals = [sum(row) for row in exact_counts]
    exact_probabilities = [
        [value / total for value in row]
        for row, total in zip(exact_counts, exact_totals, strict=True)
    ]
    expected_counts = torch.tensor([[float(value) for value in row] for row in exact_counts])
    expected_probabilities = torch.tensor(
        [[float(value) for value in row] for row in exact_probabilities]
    )
    assert torch.allclose(counts, expected_counts, rtol=0, atol=1e-6)
    assert torch.allclose(probabilities, expected_probabilities, rtol=0, atol=1e-6)
    assert totals.shape == (7, 1) and counts.sum(dim=1).shape == (7,)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(7))
    assert all(sum(row) == 1 for row in exact_probabilities)
    assert sum(sum(row) for row in exact_probabilities) == 7
    cat = chars.index("貓")
    look = chars.index("看")
    for row in range(7):
        for col in range(7):
            assert math.isclose(
                probabilities[row, col].item(),
                counts[row, col].item() / totals[row, 0].item(),
                rel_tol=0,
                abs_tol=1e-6,
            )
    results[str(alpha)] = {
        "dtype": str(counts.dtype),
        "device": str(counts.device),
        "initial_shape": list(initial.shape),
        "initial_unique": initial.unique().tolist(),
        "counts": counts.tolist(),
        "totals_with_keepdim": totals.tolist(),
        "totals_shape": list(totals.shape),
        "totals_without_keepdim_shape": list(counts.sum(dim=1).shape),
        "probabilities": probabilities.tolist(),
        "row_sums": probabilities.sum(dim=1).tolist(),
        "allclose_row_sums": torch.allclose(probabilities.sum(dim=1), torch.ones(7)),
        "table_probability_sum_float": probabilities.sum().item(),
        "table_probability_sum_exact": "7",
        "cat_counts": counts[cat].tolist(),
        "cat_total_exact": str(exact_totals[cat]),
        "cat_total_float": totals[cat, 0].item(),
        "cat_look_probability_exact": str(exact_probabilities[cat][look]),
        "cat_look_probability_float": probabilities[cat, look].item(),
        "other_cat_probabilities_exact": [
            str(value) for col, value in enumerate(exact_probabilities[cat]) if col != look
        ],
        "max_absolute_probability_error": max(
            abs(probabilities[row, col].item() - float(exact_probabilities[row][col]))
            for row in range(7)
            for col in range(7)
        ),
    }

assert Fraction(3, 9) == Fraction(1, 3)
assert Fraction(21, 10) / Fraction(27, 10) == Fraction(7, 9)
assert round(float(Fraction(1, 3)), 3) == 0.333
assert round(float(Fraction(7, 9)), 3) == 0.778
assert Fraction(7, 9) > Fraction(1, 3)

w = torch.tensor(3.0, requires_grad=True)
loss = (2 * w).square()
loss.backward()
assert w.grad.item() == 24
with torch.no_grad():
    w -= 0.01 * w.grad
assert math.isclose(w.item(), 2.76, abs_tol=1e-6)

hashes = {
    path: sha256(path)
    for path in [
        "docs/course-review-20261019/freeze-01/sources/1.5.md",
        "docs/course-review-20261019/freeze-01.json",
        "course/figures/foundations_bigram_row.svg",
        "outputs/course-review-20261019/figure-renders/5f0f4445ba1fa5d97d20bdf840277b09cdba5c2cf78e7714e16bce56191db760.png",
    ]
}
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-5/verify_bigram.py",
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
    },
    "text": text,
    "chars": chars,
    "ids": ids,
    "pairs": pairs,
    "pair_counts": [{"current": current, "answer": answer, "count": count}
                    for (current, answer), count in sorted(observed.items())],
    "row_observations": dict(row_observations),
    "baseline_look_fraction_all_characters": str(Fraction(text.count("看"), len(text))),
    "baseline_look_fraction_next_positions": str(Fraction(text[1:].count("看"), len(pairs))),
    "smoothing": results,
    "autograd_demonstration": {"gradient": 24, "parameter_after_update": w.item()},
    "hashes": hashes,
    "result": "所有斷言通過；整數與精確分數完全相符，float32 誤差小於 1e-6。",
}, ensure_ascii=False, indent=2))
