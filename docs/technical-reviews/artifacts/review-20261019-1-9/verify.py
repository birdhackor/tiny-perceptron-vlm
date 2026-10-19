"""用精確分數與原始 Python 程式核對平方代價的算例。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "docs/course-review-20261019/freeze-01/sources/1.9.md"


def loss(w):
    return (w - 3) ** 2


def rational(value):
    return str(Fraction(value))


def central(w, h):
    return (loss(w + h) - loss(w - h)) / (2 * h)


raw = FROZEN.read_bytes()
code = re.search(rb"```python\n(.*?)```", raw, re.S).group(1).decode()
rows = []
for text in ("1", "1.1", "1.01"):
    w = Fraction(text)
    delta = w - 1
    value = loss(w)
    expected = {"1": "4", "1.1": "3.61", "1.01": "3.9601"}[text]
    assert value == Fraction(expected)
    rows.append({"w": text, "loss": rational(value), "change": rational(value - 4),
                 "quotient": rational((value - 4) / delta) if delta else None})

executions = []
for initial in (1.0, 5.0, 3.0):
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "w" for target in node.targets
        ):
            node.value = ast.Constant(value=initial)
    ast.fix_missing_locations(tree)
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, str(FROZEN), "exec"), namespace)
    gradient = namespace["gradient"]
    expected = 2 * (initial - 3)
    assert abs(gradient - expected) < 2e-12
    assert namespace["w"] == initial
    executions.append({"initial_w": initial, "final_w": namespace["w"],
                       "h": namespace["h"], "left": namespace["left"],
                       "right": namespace["right"], "gradient": gradient,
                       "analytic": expected, "absolute_error": abs(gradient - expected),
                       "stdout": stdout.getvalue()})

h = Fraction("0.001")
points = []
for w in map(Fraction, (1, 5, 3)):
    left, right = loss(w - h), loss(w + h)
    value = central(w, h)
    assert value == 2 * (w - 3)
    points.append({"w": rational(w), "h": rational(h),
                   "left_w": rational(w - h), "right_w": rational(w + h),
                   "left_loss": rational(left), "right_loss": rational(right),
                   "loss_difference": rational(right - left),
                   "parameter_difference": rational(2 * h), "central": rational(value)})

step = Fraction("0.001")
actual_change = loss(1 + step) - loss(Fraction(1))
linear_change = -4 * step
assert actual_change == Fraction("-0.003999")
assert actual_change - linear_change == step**2
assert loss(Fraction(7)) == 16 > loss(Fraction(1))

float_steps = []
for size in (1e-1, 1e-3, 1e-8, 1e-12, 1e-16, 1e-20):
    approximate = central(1.0, size)
    float_steps.append({"h": size, "central": approximate,
                        "absolute_error": abs(approximate + 4)})

large_step_example = {"function": "f(x)=x^3", "x": 0,
                      "h": 1, "central": 1, "derivative": 0}
assert (1**3 - (-1)**3) / 2 == 1

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-1-9/verify.py",
    "environment": {"python": sys.version, "implementation": platform.python_implementation(),
                    "platform": platform.platform(), "device": "CPU；Python float 與 Fraction",
                    "float_mantissa_bits": str(sys.float_info.mant_dig)},
    "frozen_source_sha256": hashlib.sha256(raw).hexdigest(),
    "fence_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "exact_table": rows, "original_code_and_exercises": executions,
    "exact_central_differences": points,
    "local_change": {"delta_w": rational(step), "linear_delta_loss": rational(linear_change),
                     "exact_delta_loss": rational(actual_change),
                     "remainder": rational(actual_change - linear_change)},
    "long_jump": {"old_w": 1, "old_loss": 4, "new_w": 7, "new_loss": 16},
    "finite_difference_float_steps": float_steps,
    "general_large_step_counterexample": large_step_example,
    "derivations": [
        "L(w+d)-L(w)=2(w-3)d+d²；d 非零時商為 2(w-3)+d，d 趨近零得到 L'(w)=2(w-3)。",
        "L(w+h)-L(w-h)=4(w-3)h；h 非零時中心差分精確等於 2(w-3)。這是平方函式的特性，不能外推到一般函式。",
        "w=1 與 w=5 的代價同為 4，導數分別為 -4 與 +4，故代價值本身不決定調整方向。",
        "w=1 的增量代價為 -4d+d²；0<d<4 時下降，但 d=6 時代價從 4 增為 16。",
        "一般函式 f(x)=x³ 在 x=0 的中心差分為 h²，導數為 0；步長過大可以失去局部斜率的代表性。"
    ],
    "result": "全部精確數值、原始輸出、練習與參數未更新的檢查均通過。"
}
print(json.dumps(result, ensure_ascii=False, indent=2))
