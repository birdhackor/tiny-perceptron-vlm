import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

source_path = Path("docs/course-review-20261019/freeze-01/sources/9.1.md")
raw = source_path.read_bytes()
fences = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(fences) == 1, "原始 Python 程式區塊數量不符預期"
code = fences[0]
tree = ast.parse(code)
assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)), "範例含有匯入"
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert all(isinstance(node.func, ast.Name) and node.func.id == "print" for node in calls), "範例含有列印以外的函式呼叫"


def evaluate(program):
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(program, str(source_path), "exec"), namespace)
    return namespace, stdout.getvalue()


original, original_output = evaluate(code)
assert original_output == "True True True\n較安全回答可直接當安全正例 False\n", "原始輸出不符預期"
key = f"回答{original['pair']['相對較安全']}安全"
assert key == "回答0安全", "組合後的字典鍵不符預期"
assert original["pair"][key] is False, "原始安全標籤不符預期"
assert original["pair"]["回答1安全"] is False, "另一個安全標籤不符預期"
assert code.count('"回答0安全": False') == 1, "要修改的字典欄位不唯一"
modified_code = code.replace('"回答0安全": False', '"回答0安全": True')
modified, modified_output = evaluate(modified_code)
assert modified_output == "True True True\n較安全回答可直接當安全正例 True\n", "修改後輸出不符預期"
assert modified["record"] == original["record"], "修改影響到另一筆獨立資料"
assert "有用" not in modified["pair"] and "誠實" not in modified["pair"], "偏好對出現未宣告的評估欄位"
assert modified["pair"]["回答1安全"] is False, "修改影響到另一回答的標籤"

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-9-1/verify_example.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "cpu", "torch": "本次驗證未使用"},
    "source_path": str(source_path),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "original_stdout": original_output,
    "original_record": original["record"],
    "original_pair": original["pair"],
    "selected_key": key,
    "selected_safety": original["pair"][key],
    "modified_stdout": modified_output,
    "modified_record": modified["record"],
    "modified_pair": modified["pair"],
    "record_unchanged": modified["record"] == original["record"],
    "pair_has_usefulness_or_honesty_labels": "有用" in modified["pair"] or "誠實" in modified["pair"],
    "imports": [],
    "called_functions": [node.func.id for node in calls],
    "result": "全部斷言通過；僅核對字典、字串與輸出行為，未評估模型能力。"
}
print(json.dumps(result, ensure_ascii=False, indent=2))
