"""Independent bounded standard-library CPU verification of chapter 1.4."""
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import contextlib
import io
import json
import platform
import sys

OUT = Path(__file__).resolve().parent
original = (OUT / "original-fence.py").read_bytes()
environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
               "device": "CPU; standard library only, no torch/model/training", "command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-1_4-independent/verify.py",
               "original_fence_sha256": sha256(original).hexdigest()}
print("ENVIRONMENT", json.dumps(environment, ensure_ascii=False))
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")

def execute(code, name):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, name, "exec"), namespace)
    print(name, "STDOUT", repr(stdout.getvalue()))
    return namespace

base = execute(original, "original-fence.py")
assert base["counts"] == Counter({"貓": 3, "狗": 1, "鳥": 1})
assert base["total"] == 5
assert base["probability"] == {"貓": 3 / 5, "狗": 1 / 5, "鳥": 1 / 5}
assert sum(Fraction(count, base["total"]) for count in base["counts"].values()) == 1
print("BASE", json.dumps({"counts": dict(base["counts"]), "total_tokens": base["total"],
                         "probability": base["probability"], "sum": sum(base["probability"].values())}, ensure_ascii=False))

changed = original.replace('text = "貓貓貓狗鳥"'.encode(), 'text = "貓貓貓狗狗狗鳥"'.encode())
assert changed != original
(OUT / "exercise-fence.py").write_bytes(changed)
exercise = execute(changed, "exercise-fence.py")
assert exercise["counts"]["貓"] == 3 and exercise["total"] == 7
assert exercise["probability"]["貓"] == 3 / 7
assert round(exercise["probability"]["貓"], 3) == 0.429
assert base["probability"]["貓"] > exercise["probability"]["貓"]
print("EXERCISE", json.dumps({"counts": dict(exercise["counts"]), "total_tokens": exercise["total"],
                             "p_cat_exact": str(Fraction(3, 7)), "p_cat_float": exercise["probability"]["貓"],
                             "p_cat_3_decimals": round(exercise["probability"]["貓"], 3),
                             "sum": sum(exercise["probability"].values()), "sum_error": abs(sum(exercise["probability"].values()) - 1)}, ensure_ascii=False))

for name, text in [("permuted", "鳥貓狗貓貓"), ("duplicated", "貓貓貓狗鳥" * 2)]:
    counts = Counter(text)
    p = {char: count / sum(counts.values()) for char, count in counts.items()}
    assert p == base["probability"]
    print(name.upper(), json.dumps({"text": text, "counts": dict(counts), "total": sum(counts.values()), "probability": p}, ensure_ascii=False))

greedy = max(base["probability"], key=base["probability"].get)
assert greedy == "貓"
# An exact inverse-CDF calculation, not a measured generative accuracy result.
intervals = {"貓": (Fraction(0), Fraction(3, 5)), "狗": (Fraction(3, 5), Fraction(4, 5)), "鳥": (Fraction(4, 5), Fraction(1))}
for char, (lo, hi) in intervals.items():
    assert hi - lo == Fraction(base["counts"][char], base["total"])
print("SELECTION", json.dumps({"greedy": greedy, "sample_intervals": {c: [str(a), str(b)] for c, (a, b) in intervals.items()}, "sampling_executed_by_chapter": False}, ensure_ascii=False))
print("FLOAT", json.dumps({"literal_1e_minus_6": 1e-6, "equals_one_millionth": 1e-6 == 0.000001,
                          "abs_negative_example": abs(-0.2), "decimal_example": 0.1 + 0.2,
                          "decimal_example_equals_0_3": 0.1 + 0.2 == 0.3}))
assert abs(-0.2) == 0.2 and 1e-6 == 0.000001
print("PASS: original plus exercise, permutation, duplication, exact probability denominator, greedy/sample interval, floating-point checks; no training or context API.")
