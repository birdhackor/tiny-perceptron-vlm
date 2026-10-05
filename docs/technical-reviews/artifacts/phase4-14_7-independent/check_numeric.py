"""Bounded CPU checks for the paper-card example, not a model evaluation."""
import hashlib
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
sentence = "校慶週六上午開始請在操場入口集合"
assert len(sentence) == 16
cards = list(enumerate(sentence))
distance_checks = []
for train_length in (8, 12):
    seen = list(range(train_length))
    new = list(range(train_length, len(sentence)))
    max_old = max(abs(i - j) for i in seen for j in seen)
    max_extended = max(abs(i - j) for i, _ in cards for j, _ in cards)
    assert max_old == train_length - 1
    assert max_extended == 15
    distance_checks.append({"train_cards": train_length, "seen_positions": seen,
                            "new_positions": new, "old_max_distance": max_old,
                            "extended_max_distance": max_extended,
                            "distance_unit": "token-index gaps"})

ns = {"svg": "http://www.w3.org/2000/svg"}
svg = BASE / "frozen/course/figures/rewrite-14-7-position-range.svg"
tree = ET.parse(svg)
texts = ["".join(e.itertext()) for e in tree.findall(".//svg:text", ns)]
figure_chars = [t for t in texts if t in sentence and len(t) == 1]
figure_indices = [int(t) for t in texts if t.isdecimal()]
assert figure_chars == list(sentence)
assert figure_indices == list(range(16))
assert "位置 0 → 7：相隔 7 格" in texts
assert "位置 0 → 15：相隔 15 格" in texts

def rotate_pairs(vector, position):
    d = len(vector)
    result = []
    for pair in range(d // 2):
        theta = 10000 ** (-2 * pair / d)
        angle = position * theta
        c, s = math.cos(angle), math.sin(angle)
        x, y = vector[2 * pair:2 * pair + 2]
        result.extend((c * x - s * y, s * x + c * y))
    return result

def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))

q, k = [0.6, -0.2, 1.4, 0.3], [-0.1, 0.8, -0.5, 1.1]
relative_checks = []
for m, n, shift in ((0, 7, 10), (0, 15, 10), (8, 15, 10)):
    absolute_score = dot(rotate_pairs(q, m), rotate_pairs(k, n))
    relative_score = dot(q, rotate_pairs(k, n - m))
    shifted_score = dot(rotate_pairs(q, m + shift), rotate_pairs(k, n + shift))
    error = max(abs(absolute_score - relative_score), abs(absolute_score - shifted_score))
    assert error <= 1e-12
    assert all(math.isfinite(x) for x in rotate_pairs(q, n))
    relative_checks.append({"m": m, "n": n, "common_shift": shift,
                            "absolute_score": absolute_score,
                            "relative_score": relative_score,
                            "shifted_score": shifted_score, "max_error": error})

scaled = [m * 8 / 16 for m in range(16)]
assert len(scaled) == 16 and scaled[0] == 0 and scaled[-1] == 7.5
assert all(0 <= x < 8 for x in scaled)

print(json.dumps({
    "environment": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "device": "CPU, Python stdlib only"},
    "sentence": sentence, "card_count": len(cards), "distance_checks": distance_checks,
    "figure": {"chars": figure_chars, "indices": figure_indices,
               "sha256": hashlib.sha256(svg.read_bytes()).hexdigest()},
    "rotary_identity": {"dimension": 4, "frequencies": [1, 0.01],
                        "tolerance": 1e-12, "checks": relative_checks},
    "paper_pi_formula_example": {"L": 8, "L_prime": 16, "card_count": len(scaled),
                                 "coordinate_min": min(scaled), "coordinate_max": max(scaled),
                                 "coordinate_range": "[0, 8), not the seen integer set {0,...,7}"},
    "scope": "Arithmetic and rotation identities only; no training, task accuracy, or interpolation adaptation measured.",
    "assertions": "passed"
}, ensure_ascii=False, indent=2))
