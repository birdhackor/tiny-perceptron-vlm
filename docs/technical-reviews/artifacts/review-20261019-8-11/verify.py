"""核對回答身分、顯示位置與手寫規則的確定性行為。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FROZEN = ROOT / "docs/course-review-20261019/freeze-01/sources/8.11.md"
EXPECTED_HASH = "7fb03bd772f6ccb6948827689b1e544e54093e5fef2d0fa8f9b1c42e47565d54"
raw = FROZEN.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_HASH
chapter = (ROOT / "course/chapters/08.md").read_bytes()
section = re.search(rb"(?ms)^## 8\.11 .*?(?=^## |\Z)", chapter).group()
assert section == raw
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
tree = ast.parse(code)
assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))
calls = sorted({ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)})
assert calls == ["choose_first", "print"]
judge = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
assert judge.name == "choose_first"
assert len(judge.body) == 1
assert ast.unparse(judge.body[0]) == "return order[0]"


def execute(program):
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(program, str(FROZEN), "exec"), namespace)
    return namespace, output.getvalue()


original, original_stdout = execute(tree)
expected_original = "原順序選 correct 2+2=4\n交換後選 wrong 2+2=5\n內容身分一致 False\n"
assert original_stdout == expected_original
assert original["answers"] == {"correct": "2+2=4", "wrong": "2+2=5"}
assert original["original_order"] == ["correct", "wrong"]
assert original["swapped_order"] == ["wrong", "correct"]
assert original["original_order"] is not original["swapped_order"]
assert 2 + 2 == 4
assert 2 + 2 != 5

orders = [original["original_order"], original["swapped_order"]]
labels = [dict(zip(("A", "B"), order, strict=True)) for order in orders]
assert [mapping["A"] for mapping in labels] == ["correct", "wrong"]
assert [next(key for key, value in mapping.items() if value == "correct") for mapping in labels] == ["A", "B"]
assert [original["choose_first"](order) for order in orders] == ["correct", "wrong"]

ideal_tree = ast.parse(code)
ideal_judge = next(node for node in ideal_tree.body if isinstance(node, ast.FunctionDef))
ideal_judge.body = [ast.Return(value=ast.Constant(value="correct"))]
ast.fix_missing_locations(ideal_tree)
ideal, ideal_stdout = execute(ideal_tree)
expected_ideal = "原順序選 correct 2+2=4\n交換後選 correct 2+2=4\n內容身分一致 True\n"
assert ideal_stdout == expected_ideal
assert [ideal["choose_first"](order) for order in orders] == ["correct", "correct"]

manifest = ROOT / "docs/course-review-20261019/freeze-04.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert manifest_hash == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
assert not re.search(r"!\[|<img|\.svg", raw.decode())
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-11/verify.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU；僅標準函式庫"},
    "source_sha256": EXPECTED_HASH,
    "current_equals_frozen": True,
    "freeze_manifest_sha256": manifest_hash,
    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "arithmetic": {"2+2": 2 + 2, "equals_4": 2 + 2 == 4, "equals_5": 2 + 2 == 5},
    "original_stdout": original_stdout,
    "exercise_stdout": ideal_stdout,
    "original_order": original["original_order"],
    "swapped_order": original["swapped_order"],
    "labels_to_identity": labels,
    "same_letter_A_identities": [mapping["A"] for mapping in labels],
    "correct_identity_display_letters": [next(key for key, value in mapping.items() if value == "correct") for mapping in labels],
    "all_calls": calls,
    "judge_body": ast.unparse(judge.body[0]),
    "model_calls": 0,
    "figure_references": 0,
    "result": "全部斷言通過；原例與練習的輸出逐字相符。",
    "scope": "兩個固定回答與兩種排列的手寫規則。未呼叫真實模型，不能推論其位置偏差頻率，也不能以順序一致推出答案正確。",
}, ensure_ascii=False, indent=2))
