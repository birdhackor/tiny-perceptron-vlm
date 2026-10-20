"""核對凍結課文的算式、程式結構與對照題能排除的簡單捷徑。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/course-review-20261019/freeze-01/sources/13.8.md"
raw = SOURCE.read_bytes()
body = raw.decode("utf-8")
code = body.split("```python\n", 1)[1].split("```", 1)[0]
tree = ast.parse(code)
assert [type(node).__name__ for node in tree.body] == ["Assign"] + ["Expr"] * 4
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert len(calls) == 4
assert all(isinstance(node.func, ast.Name) and node.func.id == "print" for node in calls)
assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))

namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(tree, str(SOURCE), "exec"), namespace)
pair = namespace["pair"]
expected_stdout = (
    "正確答案 4\n"
    "較佳 2+2等於4；把兩組各2個物件合起來，就是4個。\n"
    "較差 完全正確，你說得很好！\n"
    "偏好理由 溫和修正錯誤，並提供可核對解釋\n"
)
assert stdout.getvalue() == expected_stdout
left = {"左甲", "左乙"}
right = {"右甲", "右乙"}
assert left.isdisjoint(right)
assert len(left | right) == 2 + 2 == 4

control_cases = []
for stated_result in (5, 4):
    truth = (2 + 2) == stated_result
    shortcut_agrees = False
    control_cases.append({
        "prompt": f"2+2={stated_result}，對吧？",
        "is_true": truth,
        "fact_based_action": "確認" if truth else "修正",
        "always_disagree_action": "反駁",
        "always_disagree_is_correct": shortcut_agrees == truth,
    })
assert control_cases[0]["always_disagree_is_correct"]
assert not control_cases[1]["always_disagree_is_correct"]

rude = dict(pair)
rude["chosen"] = "笨蛋，當然是4"
assert rude["chosen"].endswith(str(2 + 2))
assert rude["rejected"] == pair["rejected"]
assert rude["reason"] == pair["reason"]

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-8/verify_examples.py",
    "environment": {"python": platform.python_version(), "device": "CPU；僅 Python 標準函式庫"},
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
    "statement_types": [type(node).__name__ for node in tree.body],
    "called_functions": [node.func.id for node in calls],
    "stdout": stdout.getvalue(),
    "stdout_exact_match": stdout.getvalue() == expected_stdout,
    "arithmetic": {"left_count": len(left), "right_count": len(right), "disjoint": left.isdisjoint(right), "union_count": len(left | right), "sum": 2 + 2},
    "control_cases": control_cases,
    "rude_variant": rude,
    "interpretation": "對照題僅在邏輯上識別總是反駁的規則；未執行模型推論或訓練。辱罵是否不禮貌與理由是否適用，由審閱者讀原文判斷。",
}, ensure_ascii=False, indent=2))
