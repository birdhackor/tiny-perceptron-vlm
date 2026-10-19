"""核對凍結教材的算例與 Counter 的實際介面差異。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


root = Path(__file__).resolve().parents[4]
snapshot = root / "docs/course-review-20261019/freeze-01/sources/1.4.md"
raw = snapshot.read_bytes()
source_hash = hashlib.sha256(raw).hexdigest()
assert source_hash == "4436f03e544b3b36ee6408e1c61a60a8357530f5c2eec637791329336bd6eeb0"
current = (root / "course/chapters/01.md").read_bytes()
match = re.search(rb"(?ms)^## 1\.4 .*?(?=^## |\Z)", current)
assert match and match.group() == raw
freeze = root / "docs/course-review-20261019/freeze-01.json"
freeze_hash = hashlib.sha256(freeze.read_bytes()).hexdigest()
assert freeze_hash == "71bd28f633f6b1ae3fce6e6279239f261893b42055e70974b142d25b9a45b5c0"

code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(snapshot), "exec"), namespace)
counts = namespace["counts"]
probability = namespace["probability"]
assert counts == {"貓": 3, "狗": 1, "鳥": 1}
assert namespace["total"] == 5
assert probability == {"貓": 0.6, "狗": 0.2, "鳥": 0.2}
assert list(counts.values()) == [3, 1, 1]
assert list(counts.items()) == [("貓", 3), ("狗", 1), ("鳥", 1)]
assert isinstance(counts, dict)
assert counts["魚"] == 0
assert "魚" not in counts
assert 1e-6 == 0.000001
assert abs(-0.2) == 0.2
exact = {k: Fraction(v, 5) for k, v in counts.items()}
assert sum(exact.values()) == 1
assert all(exact[k] > 0 for k in ("狗", "鳥"))
assert max(probability, key=probability.get) == "貓"

exercise_namespace = {}
exercise_stdout = io.StringIO()
exercise_code = code.replace('text = "貓貓貓狗鳥"', 'text = "貓貓貓狗狗狗鳥"')
with contextlib.redirect_stdout(exercise_stdout):
    exec(compile(exercise_code, "exercise", "exec"), exercise_namespace)
assert exercise_namespace["counts"] == {"貓": 3, "狗": 3, "鳥": 1}
assert exercise_namespace["total"] == 7
assert exercise_namespace["probability"]["貓"] == 3 / 7
assert round(3 / 7, 3) == 0.429
assert Fraction(3, 7) < Fraction(3, 5)

dictionary_fromkeys = dict.fromkeys("貓狗")
try:
    Counter.fromkeys("貓狗")
except NotImplementedError as error:
    counter_fromkeys = {"exception": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("Counter.fromkeys 的行為與官方文件不符")
dictionary = {"貓": 3}
counter = Counter({"貓": 3})
dictionary.update({"貓": 1})
counter.update({"貓": 1})
assert dictionary["貓"] == 1
assert counter["貓"] == 4

report = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-4/verify.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU", "float_mant_dig": str(sys.float_info.mant_dig)},
    "source_sha256": source_hash,
    "freeze_sha256": freeze_hash,
    "current_matches_freeze": True,
    "original": {"stdout": captured.getvalue(), "counts": dict(counts), "total": namespace["total"], "probability": probability, "sum": sum(probability.values()), "exact_rationals": {k: str(v) for k, v in exact.items()}, "values": list(counts.values()), "items": list(counts.items())},
    "exercise": {"stdout": exercise_stdout.getvalue(), "counts": dict(exercise_namespace["counts"]), "total": exercise_namespace["total"], "cat_probability": exercise_namespace["probability"]["貓"], "rounded_three_places": round(3 / 7, 3)},
    "floating_point": {"cat_float_ratio": str(Fraction.from_float(probability["貓"])), "cat_exact_ratio": "3/5", "cat_representation_error": str(Fraction.from_float(probability["貓"]) - Fraction(3, 5)), "sum_error": abs(sum(probability.values()) - 1), "tolerance": 1e-6, "absolute_value_example": abs(-0.2)},
    "selection": {"greedy": max(probability, key=probability.get), "sampling_intervals": {"貓": "[0, 3/5)", "狗": "[3/5, 4/5)", "鳥": "[4/5, 1)"}, "interpretation": "對均勻的 U∈[0,1) 依上述區間選字；區間長度等於各字機率，這是解析核對，未宣稱已進行抽樣實驗。"},
    "counter_interface": {"is_dict": isinstance(counts, dict), "missing_subscript": counts["魚"], "dict_fromkeys": dictionary_fromkeys, "Counter_fromkeys": counter_fromkeys, "dict_update_cat": dictionary["貓"], "Counter_update_cat": counter["貓"]},
    "result": "全部核對斷言通過；Counter.fromkeys 與 update 的實際差異否定所有字典寫法皆可照用的概括。"
}
print(json.dumps(report, ensure_ascii=False, indent=2))
