"""Independent, finite checks for 8.1; no model inference or training."""
import contextlib
import copy
import io
import json
import runpy
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
expected = (
    "4 True False\n"
    "4，就像兩雙筷子共有四根。 True True\n"
    "5，數字如星光流動。 False False\n"
)
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    namespace = runpy.run_path(str(BASE / "execution/original/fence-1.py"))
assert capture.getvalue() == expected
examples = namespace["examples"]
assert all(type(r[k]) is bool for r in examples for k in ("正確", "貼切比喻"))
assert [(r["正確"], r["貼切比喻"]) for r in examples] == [
    (True, False), (True, True), (False, False)
]
print("original stdout matches exactly; 3 manually labelled records")

# Change only the text: a print loop cannot infer a new semantic label.
variant = copy.deepcopy(examples)
variant[1]["回答"] = "4，像一片深邃的海。"
assert variant[1]["貼切比喻"] is True
print("text-only mutation (label unchanged):", variant[1])
# The exercise explicitly calls for a human to change the second label.
variant[1]["貼切比喻"] = False
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    for row in variant:
        print(row["回答"], row["正確"], row["貼切比喻"])
assert capture.getvalue().splitlines()[1] == "4，像一片深邃的海。 True False"
print("exercise after manual relabelling:")
print(capture.getvalue(), end="")

# Literal quantities, not a semantic analogy detector: two pairs each contain 2.
groups = [[1, 1], [1, 1]]
assert [len(g) for g in groups] == [2, 2]
assert sum(map(len, groups)) == 2 + 2 == 4
assert 2 + 2 != 5
print("exact arithmetic: two groups of two = 4; 5 is false; tolerance = 0")

# RFC 8259 object example and two representational boundaries.
value = json.loads('{"answer": 4}')
assert value == {"answer": 4} and type(value["answer"]) is int
assert json.loads("4") == 4  # JSON can also be a scalar, not only field/value objects.
try:
    json.loads('答案是：{"answer": 4}')
except json.JSONDecodeError:
    print("JSON: object maps answer to integer 4; scalar 4 is valid; prose prefix rejected")
else:
    raise AssertionError("prose-prefixed reply unexpectedly parsed")
print("scope: printing supplied labels and parsing supplied JSON; no learning or evaluation of a model")
