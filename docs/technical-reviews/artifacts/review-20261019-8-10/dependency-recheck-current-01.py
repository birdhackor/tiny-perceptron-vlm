"""核對依賴課文差異、明確規則驗證器與確定性重試的算例。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
CHAPTER = ROOT / "course/chapters/0C.md"
APPLICATIONS = ROOT / "scripts/course_experiments/applications.py"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_8-10/technical-8-10-original-01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*arguments):
    command = ["git", *arguments]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    return {"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}


def first_code(raw):
    section = re.search(r"^## [^\n]*最後答對代表過程正確嗎？\n(.*?)(?=^## |\Z)", raw, re.M | re.S).group(1)
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


original_raw = ORIGINAL.read_bytes()
assert digest(original_raw) == "15f0b0d0b969ac9c68f16b6bdfb8ad264d47063ee685cb044a23180079eca4b4"
original_report = json.loads(original_raw)
old = git("show", "5ef56bc756f3:course/chapters/0C.md")
updated = git("show", "3af7f13ef959:course/chapters/0C.md")
diff = git("diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/0C.md")
current = CHAPTER.read_bytes()
assert digest(old["stdout"].encode()) == "04f2fe71edbf5f788d3e27d1a3325710ff53e46429d9295c7555f65efbd2cd59"
assert updated["stdout"].encode() == current
old_code = first_code(old["stdout"])
current_code = first_code(current.decode())
assert old_code == current_code

modified_tree = ast.parse(current_code)
for node in modified_tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "steps" for target in node.targets):
        node.value = ast.parse("[(2, 2, 4), (4, 3, 7), (7, -3, 4)]", mode="eval").body
ast.fix_missing_locations(modified_tree)
namespace = {}
cycle_stdout = io.StringIO()
with contextlib.redirect_stdout(cycle_stdout):
    exec(compile(modified_tree, str(CHAPTER), "exec"), namespace)
assert namespace["valid"] == [True, True, True]
assert namespace["linked"] == [True, True]
assert namespace["used"] == [2, 2, 3, -3]
assert sorted(namespace["used"]) != sorted(namespace["question"])
assert namespace["steps"][-1][2] == namespace["truth"]

application_text = APPLICATIONS.read_text()
application_tree = ast.parse(application_text)
function = next(node for node in application_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_verify_reasoning")
function_code = ast.get_source_segment(application_text, function)
verifier_namespace = {"re": re}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(APPLICATIONS), "exec"), verifier_namespace)
row = {"a": 2, "b": 2, "c": 3, "truth": 7}
cases = {}
for name, text in (("valid", "2+2=4;4+3=7;answer=7"), ("different_operands_same_total", "1+3=4;4+3=7;answer=7")):
    cases[name] = {"generated": text, "result": verifier_namespace["_verify_reasoning"]({"generated": text, "invalid_special_tokens": []}, row, "steps")}
assert cases["valid"]["result"]["fully_verified"]
changed = cases["different_operands_same_total"]["result"]
assert changed["equations_valid"] and changed["linked"] and changed["final_correct"]
assert not changed["task_operands_valid"] and not changed["fully_verified"]

scores = torch.tensor([0.2, 0.6, 0.1])
chosen = [int(scores.argmax(-1).item()) for _ in range(8)]
assert chosen == [1] * 8
coverage = [{"p": p, "k": k, "coverage": 1 - (1 - p) ** k} for p in (0, 1) for k in (1, 2, 4, 8)]
assert all(value["coverage"] == value["p"] for value in coverage)

integrity = []
for artifact in original_report["artifacts"]:
    observed = digest((ROOT / artifact["path"]).read_bytes())
    assert observed == artifact["sha256"]
    integrity.append({"path": artifact["path"], "expected_sha256": artifact["sha256"], "observed_sha256": observed})
sample_function = next(node for node in application_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_sample")
sample_code = ast.get_source_segment(application_text, sample_function)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-8-10/dependency-recheck-current-01.py",
    "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "git_receipts": {"old_source": old, "updated_source": updated, "requested_diff": diff},
    "current_sources": {"course/chapters/0C.md": digest(current), "scripts/course_experiments/applications.py": digest(APPLICATIONS.read_bytes())},
    "verifier_first_code": {"old_sha256": digest(old_code.encode()), "current_sha256": digest(current_code.encode()), "identical": True},
    "cycle_exercise": {"steps": namespace["steps"], "valid": namespace["valid"], "linked": namespace["linked"], "used": namespace["used"], "task_operands_valid": False, "final_correct": True, "stdout": cycle_stdout.getvalue()},
    "experiment_verifier": {"function": "_verify_reasoning", "code": function_code, "code_sha256": digest(function_code.encode()), "row": row, "cases": cases, "scope": "原碼按 (a,b,c) 的位置精確比較，不只比較多重集合；明示數字各用一次是必要條件，仍受本文列出的步驟與順序約束。"},
    "greedy": {"source_function": "_sample", "source_code": sample_code, "scores": scores.tolist(), "chosen": chosen, "coverage": coverage, "derivation": "固定題目、固定模型與確定性解碼時，整份回答固定；正確性指示變數是常數 0 或 1，所以重試 k 次的 1-(1-p)^k 仍等於 p。這不把一般抽樣 p 與 greedy 混用，也不聲稱不同輸入或非確定性後端一定輸出相同。"},
    "original_artifacts_integrity": integrity,
    "result": "全部斷言通過；既有 artifact 雜湊不變。未訓練模型，未執行遠端服務。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
