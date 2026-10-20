import ast
import contextlib
import copy
import hashlib
import io
import itertools
import json
import platform
import re
from fractions import Fraction
from pathlib import Path


root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-01/sources/7.25.md"
raw = source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == "8b7bc5f4b4f8a4b73a7ab8e64d230761447fab9bdc8d4895d928f99e56d728d0", "凍結原文不符"
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S).group(1)
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(code, str(source), "exec"), namespace)
logs = namespace["logs"]
summary = {}
for name, row in logs.items():
    count = len(row["ok"])
    assert count == len(row["work"]) == 4, "資料長度不符"
    assert set(row["ok"]) <= {0, 1}, "正確標籤必須是二元值"
    summary[name] = {
        "correct": sum(row["ok"]),
        "questions": count,
        "accuracy": str(Fraction(sum(row["ok"]), count)),
        "total_work": sum(row["work"]),
        "mean_work": str(Fraction(sum(row["work"]), count)),
    }
assert [summary[n]["accuracy"] for n in logs] == ["1/2", "1", "3/4"], "正確率不符"
assert [summary[n]["mean_work"] for n in logs] == ["1", "3", "5"], "工作平均不符"

names = tuple(logs)
oracle_choices = []
for i in range(4):
    correct = [name for name in names if logs[name]["ok"][i]]
    oracle_choices.append(min(correct, key=lambda name: logs[name]["work"][i]) if correct else "low")
assert oracle_choices == ["low", "low", "medium", "medium"], "理想選擇不符"
assert namespace["oracle_ok"] == [1, 1, 1, 1], "理想正確標籤不符"
assert namespace["oracle_work"] == [1, 1, 3, 3], "理想工作量不符"

policies = []
for choices in itertools.product(names, repeat=4):
    correct = sum(logs[name]["ok"][i] for i, name in enumerate(choices))
    work = sum(logs[name]["work"][i] for i, name in enumerate(choices))
    for i, name in enumerate(choices):
        assert logs[name]["ok"][i] <= namespace["oracle_ok"][i], "選擇器超出固定候選上界"
    policies.append({"choices": list(choices), "correct": correct, "total_work": work})
perfect = [p for p in policies if p["correct"] == 4]
minimum_perfect_work = min(p["total_work"] for p in perfect)
assert len(policies) == 81 and minimum_perfect_work == 8, "枚舉結果不符"

all_wrong = copy.deepcopy(logs)
for row in all_wrong.values():
    row["ok"][-1] = 0
tree = ast.parse(code)
assert isinstance(tree.body[0], ast.Assign), "預期先設定人工紀錄"
tree.body[0].value = ast.parse(repr(all_wrong), mode="eval").body
ast.fix_missing_locations(tree)
wrong_namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(tree, str(source), "exec"), wrong_namespace)
wrong_scores, wrong_work = wrong_namespace["oracle_ok"], wrong_namespace["oracle_work"]
assert wrong_namespace["right"] == [], "最後一題應無正確候選"
assert wrong_scores == [1, 1, 1, 0] and wrong_work == [1, 1, 3, 1], "全錯分支不符"
assert bool([]) is False and bool([0]) is True, "清單真值不符"
feasible_fixed = [name for name in names if Fraction(sum(logs[name]["work"]), 4) <= 3]
assert feasible_fixed == ["low", "medium"], "平均預算判斷不符"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-7-25/verify_budget_table.py",
    "environment": {"python": platform.python_version(), "device": "CPU，純 Python 整數與有理數計算"},
    "source_sha256": digest,
    "original_stdout": captured.getvalue(),
    "summary": summary,
    "oracle": {"choices": oracle_choices, "correct": 4, "questions": 4, "total_work": 8, "mean_work": "2"},
    "enumeration": {"policy_count": len(policies), "minimum_work_for_four_correct": minimum_perfect_work},
    "all_wrong_branch": {"correct": wrong_scores, "work": wrong_work},
    "truth_values": {"empty_list": bool([]), "nonempty_list_with_zero": bool([0])},
    "mean_budget_three": {"feasible_fixed": feasible_fixed, "best_fixed": "medium"},
    "scope": "固定四題、三檔位與已記錄的答案；不包含路由計算、搜尋成本或新抽樣。",
    "result": "所有有理數、分支與枚舉核對通過。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
