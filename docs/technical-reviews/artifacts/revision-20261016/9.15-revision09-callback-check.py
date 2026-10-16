"""原技術審閱者對 freeze09 的知情複查；不寫教材或既有報告。"""

import ast
import contextlib
import difflib
import hashlib
import io
import json
import platform
import re
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.reading_time import figure_hashes, lesson_slices


def sha(content):
    return hashlib.sha256(content).hexdigest()


def code(body):
    return re.search(r"```python\n(.*?)```", body, re.S).group(1)


def execute(text):
    namespace = {}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(text, namespace)
    return captured.getvalue(), namespace


def page(freeze):
    data = json.loads((ROOT / freeze).read_text())
    item = next(item for item in data["pages"] if item["page_id"] == "9.15")
    return item, (ROOT / item["snapshot"]).read_text()


original_path = ROOT / "docs/course-revision-20261016/reviews/technical-9.15-original.json"
original_bytes = original_path.read_bytes()
old_page, old_body = page("docs/course-revision-20261016/freeze-08.json")
new_page, new_body = page("docs/course-revision-20261016/freeze-09.json")
current_source = ROOT / "course/chapters/09.md"
current_body = lesson_slices(current_source.read_text())["9.15"]
assert sha(old_body.encode()) == old_page["source_sha256"]
assert sha(new_body.encode()) == new_page["source_sha256"]
assert new_body == current_body
old_code, new_code = code(old_body), code(new_body)
prose_equal = old_body.replace(old_code, "<PYTHON BLOCK>") == new_body.replace(new_code, "<PYTHON BLOCK>")
ast_equal = ast.dump(ast.parse(old_code), include_attributes=False) == ast.dump(
    ast.parse(new_code), include_attributes=False
)
assert prose_equal and ast_equal
old_figures = figure_hashes(ROOT, current_source, old_body)
new_figures = figure_hashes(ROOT, current_source, new_body)
assert old_figures == new_figures == new_page["figures_sha256"] == {}
old_stdout, _ = execute(old_code)
new_stdout, _ = execute(new_code)
assert old_stdout == new_stdout == "最大機率 0.6 集中度 0.4\n"

results = []
for name, probabilities, expected in [
    ("正文三選一", [0.6, 0.2, 0.2], Fraction(2, 5)),
    ("相同最大值二選一", [0.6, 0.4], Fraction(1, 5)),
    ("均勻基準", [1 / 3, 1 / 3, 1 / 3], Fraction(0)),
    ("全部集中", [1.0, 0.0, 0.0], Fraction(1)),
    ("練習重排", [0.2, 0.6, 0.2], Fraction(2, 5)),
    ("集中在可對或可錯的候選", [0.99, 0.005, 0.005], Fraction(197, 200)),
]:
    substituted_code = "p = " + repr(probabilities) + "\n" + new_code.split("\n", 1)[1]
    stdout, values = execute(substituted_code)
    observed = values["confidence"]
    assert abs(observed - float(expected)) <= 1e-12
    results.append({"name": name, "p": probabilities, "expected_exact": str(expected),
                    "observed": observed, "argmax_index": probabilities.index(max(probabilities)),
                    "stdout": stdout})
try:
    execute("p = [1.0]\n" + new_code.split("\n", 1)[1])
except ZeroDivisionError:
    singleton = "ZeroDivisionError"
else:
    raise AssertionError("n=1應除零")

assert original_bytes == original_path.read_bytes()
result = {
    "reviewer_task": "/root/tech_20261016_9_15",
    "initial_review_context": "fresh",
    "callback_context": "informed; same original reviewer, not a new blind review",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/9.15-revision09-callback-check.py",
    "environment": {"python": platform.python_version(), "executable": sys.executable,
                    "platform": platform.platform(), "device": "CPU; Python scalar arithmetic only"},
    "old_snapshot": old_page["snapshot"], "new_snapshot": new_page["snapshot"],
    "old_source_sha256": sha(old_body.encode()), "new_source_sha256": sha(new_body.encode()),
    "current_matches_freeze09": new_body == current_body,
    "original_report_sha256_before_and_after": sha(original_bytes),
    "non_python_prose_identical": prose_equal, "python_ast_identical": ast_equal,
    "old_figure_sha256": old_figures, "new_figure_sha256": new_figures,
    "actual_diff": "".join(difflib.unified_diff(old_body.splitlines(keepends=True),
                    new_body.splitlines(keepends=True), fromfile="freeze08/9.15.md", tofile="freeze09/9.15.md")),
    "old_code": old_code, "new_code": new_code,
    "old_stdout": old_stdout, "new_stdout": new_stdout,
    "numeric_cases_using_new_code_expression": results,
    "singleton_using_new_code_expression": singleton,
    "derivation_rechecked": [
        "n>1時1/n<=p_max<=1，分母1-1/n>0；均勻c=0，全部集中c=1。",
        "n=3,p_max=3/5：(3/5-1/3)/(2/3)=2/5；n=2：(3/5-1/2)/(1/2)=1/5。",
        "重排保留p_max,n所以c不變而argmax換位；公式不含真值y，不提供答對頻率。",
        "n=1分母0；單一候選需另訂介面行為，並非宣稱官方既有邊界契約。"
    ],
    "claim_rechecks": {
        "c1": "已重讀新版公式與解說，與freeze08和原一手來源定義相同；端點推導重核仍成立。",
        "c2": "已親自執行新版Python原碼與六組替換輸入；相同公式、數值、重排及單候選邊界均成立。",
        "c3": "已重讀新版校準與澄清段；公式仍不含真值，不能由集中度推答對頻率，原留出資料限制不變。",
        "c4": "已重讀新版Noul/Score段，非Python正文逐字未變；原官方來源的型別定義及自有介面預告範圍仍適用。"
    },
    "result": "全部比較與算例通過；本次僅小數前導零與Python空白格式變更。",
    "scope": "本審閱者只執行此短Python核對；未執行12個Notebook kernels，也未把作者kernel-execution09稱為自己執行。保留原一手來源核對，不重新下載；未改教材、訓練、提交或派人。"
}
target = ROOT / "docs/technical-reviews/artifacts/revision-20261016/9.15-revision09-callback-execution.json"
with target.open("x") as stream:
    stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
