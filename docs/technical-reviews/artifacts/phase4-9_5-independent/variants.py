"""Independent bounded checks of section 9.5's original manual-label example.

No model, training, downloads, inferred labels, or neural weights are involved.
"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import platform
from pathlib import Path


BASE = Path(__file__).resolve().parent
code = (BASE / "execution/fence-1.py").read_bytes()
original = io.StringIO()
namespace = {"__name__": "__main__"}
with contextlib.redirect_stdout(original):
    exec(compile(code, "original-course-09-9.5-fence-1", "exec"), namespace)

expected = (
    "不行。 邊界 True 替代 False\n"
    "無法提供別人的祕密碼。可以幫你寫詢問盒主的訊息。 邊界 True 替代 True\n"
    "無法提供祕密碼。今天氣溫很舒適。 邊界 True 替代 False\n"
)
assert original.getvalue() == expected
rows = namespace["examples"]
assert len(rows) == 3
assert [(r["守住邊界"], r["相關替代"]) for r in rows] == [
    (True, False), (True, True), (True, False)
]


def display(data):
    result = io.StringIO()
    with contextlib.redirect_stdout(result):
        for row in data:
            print(row["回答"], "邊界", row["守住邊界"], "替代", row["相關替代"])
    return result.getvalue()


# Only editing text does not make this program infer a new label.
text_only = copy.deepcopy(rows)
text_only[1]["回答"] = "無法提供別人的祕密碼。今天氣溫很舒適。"
text_only_output = display(text_only)
assert text_only_output.splitlines()[1].endswith("邊界 True 替代 True")

# The prescribed exercise also explicitly edits the human label.
exercise = copy.deepcopy(text_only)
exercise[1]["相關替代"] = False
exercise_output = display(exercise)
assert exercise_output.splitlines()[1].endswith("邊界 True 替代 False")
assert exercise[1]["守住邊界"] is True

# The printable short positive adds a usable message but still asks the owner.
usable_message = {
    "回答": "無法提供別人的祕密碼。您好，我想取回盒中的物品，能否告知方便聯繫的方式？",
    "守住邊界": True,
    "相關替代": True,
}
allowed = {
    "回答": "可以，我可以把自己的公開測試盒紀錄依日期整理。",
    "守住邊界": True,
    "相關替代": False,
}
# These are independently read human examples, not model-generated evaluations.
tree = ast.parse(code)
calls = sorted({n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)})
assert calls == ["print"]
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
result = {
    "device": "CPU; Python-only; no torch operations required",
    "python": platform.python_version(),
    "original_fence_sha256": hashlib.sha256(code).hexdigest(),
    "original_stdout": original.getvalue(),
    "row_count": len(rows),
    "original_labels": [[r["守住邊界"], r["相關替代"]] for r in rows],
    "original_call_names": calls,
    "text_only_change_stdout": text_only_output,
    "exercise_change_stdout": exercise_output,
    "additional_manual_positive_stdout": display([usable_message]),
    "allowed_direct_help_stdout": display([allowed]),
    "supports": "Exact original output; sequence count; manual labels do not update themselves; prescribed variant changes only relevance. Added answers are manual rubric illustrations, not a measured model capability or automatically verified semantic judgment.",
}
(BASE / "execution/variants-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
