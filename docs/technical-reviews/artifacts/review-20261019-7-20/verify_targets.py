import ast
import contextlib
import hashlib
import io
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path


base = Path(__file__).resolve().parent
bundle = json.loads((base / "section-execution.json").read_text())
code = bundle["files"]["fence-1.py"]["content"]
expected_base = [
    ("〈B〉", "雨", "停"),
    ("〈B〉／雨", "停", "了"),
    ("〈B〉／雨／停", "了", "〈E〉"),
    ("〈B〉／雨／停／了", "〈E〉", "（沒有目標）"),
]
expected_exercise = [
    ("〈B〉", "雨", "又"),
    ("〈B〉／雨", "又", "停"),
    ("〈B〉／雨／又", "停", "了"),
    ("〈B〉／雨／又／停", "了", "〈E〉"),
    ("〈B〉／雨／又／停／了", "〈E〉", "（沒有目標）"),
]
cases = []
for name, tokens, expected in [
    ("原始序列", ["〈B〉", "雨", "停", "了", "〈E〉"], expected_base),
    ("練習序列", ["〈B〉", "雨", "又", "停", "了", "〈E〉"], expected_exercise),
]:
    tree = ast.parse(code)
    tree.body[0].value = ast.parse(repr(tokens), mode="eval").body
    ast.fix_missing_locations(tree)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(tree, "frozen-target-example", "exec"), {})
    expected_text = "".join(
        f"位置{t} 讀到 {read}｜距離1：{near}｜距離2：{far}\n"
        for t, (read, near, far) in enumerate(expected)
    )
    assert captured.getvalue() == expected_text, "目標與人工列出的答案不符"
    cases.append({"name": name, "tokens": tokens, "stdout": captured.getvalue()})
assert cases[0]["stdout"] == bundle["files"]["stdout.txt"]["content"], "原始程式輸出不一致"

allowed = [[j <= t for j in range(4)] for t in range(4)]
assert allowed[1] == [True, True, False, False], "雨的位置只能讀到開始標記與雨"

joint = {("A", "X"): Fraction(2, 5), ("B", "Y"): Fraction(7, 20), ("C", "Y"): Fraction(1, 4)}
assert sum(joint.values()) == 1, "機率總和必須為一"
near_marginal = {x: sum(p for (a, _), p in joint.items() if a == x) for x in "ABC"}
far_marginal = {y: sum(p for (_, b), p in joint.items() if b == y) for y in "XY"}
pair = (max(near_marginal, key=near_marginal.get), max(far_marginal, key=far_marginal.get))
assert pair not in joint, "邊際最高分組合應呈現不受聯合相容性約束的例子"

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-20/verify_targets.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；純 Python 與精確分數"},
    "source_sha256": bundle["result"]["source_sha256"],
    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "cases": cases,
    "causal_allowed": allowed,
    "marginal_counterexample": {
        "joint": {"".join(k): str(v) for k, v in joint.items()},
        "near_marginal": {k: str(v) for k, v in near_marginal.items()},
        "far_marginal": {k: str(v) for k, v in far_marginal.items()},
        "independent_modes": pair,
        "joint_probability_of_modes": "0",
        "meaning": "手工機率例子只證明平行邊際預測可能不相容，不是模型實測。",
    },
    "result": "所有精確比較通過",
}, ensure_ascii=False, indent=2))
