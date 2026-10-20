"""核對凍結範例、查詢日期與新增地點的預期結果。"""

import ast
import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[4]
source = root / "docs/course-review-20261019/freeze-01/sources/A.10.md"
raw = source.read_bytes()
expected_hash = "6e6be582d451415dc04c2fb54a6d2074b8fe75861e834f2ae4abfcc4389fa3da"
if hashlib.sha256(raw).hexdigest() != expected_hash:
    raise AssertionError("凍結原文的 SHA-256 不符")
blocks = re.findall(r"```python\n(.*?)\n```", raw.decode(), re.S)
if len(blocks) != 1:
    raise AssertionError("原文必須恰有一個 Python 範例")
code = blocks[0]
tree = ast.parse(code)
imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
if imports:
    raise AssertionError("範例不應匯入模型或資料載入程式")
namespace = {}
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code, str(source), "exec"), namespace)

expected_stdout = (
    "目前（10月9日提問） {'時間': '16:00', '地點': 'M17', '格式': '兩個條列'}\n"
    "截至10月2日 {'時間': '15:00', '地點': 'M17', '格式': '一小段文字'}\n"
    "晚餐地點 對話中沒有，需要補問\n"
)
if stdout.getvalue() != expected_stdout:
    raise AssertionError("標準輸出與逐步計算的預期值不同")

original = {"時間": "15:00", "地點": "M17", "格式": "一小段文字"}
changed_time = {"時間": "16:00", "地點": "M17", "格式": "一小段文字"}
current = {"時間": "16:00", "地點": "M17", "格式": "兩個條列"}
expected = {0: {}, 1: original, 2: original, 3: changed_time, 4: current, 9: current}
observations = {}
for day, wanted in expected.items():
    actual = namespace["plan_as_of"](day)
    if actual != wanted:
        raise AssertionError(f"日期 {day} 的有效安排不符：{actual!r}")
    observations[str(day)] = actual

first = namespace["plan_as_of"](9)
second = namespace["plan_as_of"](9)
if first is second:
    raise AssertionError("不同查詢應從新的字典開始")
first["地點"] = "測試修改"
if namespace["plan_as_of"](9) != current:
    raise AssertionError("修改查詢結果不應污染原始更新")

fallback = namespace["plan_as_of"](9).get("晚餐", "對話中沒有，需要補問")
if fallback != "對話中沒有，需要補問":
    raise AssertionError("沒有晚餐安排時應回傳指定的補問文字")
dictionary = {"時間": "15:00", "地點": "M17"}
update_return = dictionary.update({"時間": "16:00"})
if dictionary != {"時間": "16:00", "地點": "M17"} or update_return is not None:
    raise AssertionError("dict.update 的覆寫與欄位保留行為不符")
if dictionary.get("時間", "缺少") != "16:00" or "晚餐" in dictionary:
    raise AssertionError("dict.get 的存在鍵與缺少鍵行為不符")

namespace["updates"].append((5, {"地點": "M18"}))
exercise_current = namespace["plan_as_of"](9)
exercise_past = namespace["plan_as_of"](2)
if exercise_current != {"時間": "16:00", "地點": "M18", "格式": "兩個條列"}:
    raise AssertionError("新增地點後的目前安排不符")
if exercise_past != original or namespace["plan_as_of"](4) != current:
    raise AssertionError("新增地點不應改動更新日期以前的安排")

record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-A-10/verify_updates.py",
    "environment": {"python": sys.version, "platform": platform.platform(), "device": "CPU，純 Python，未載入模型"},
    "source_sha256": expected_hash,
    "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
    "result": "全部核對符合預期；字串、字典、日期邊界及更新前後狀態皆精確相等",
    "stdout": stdout.getvalue(),
    "date_queries": observations,
    "independent_query_dictionaries": True,
    "missing_dinner_fallback": fallback,
    "dictionary_update_result": dictionary,
    "dictionary_update_return": update_return,
    "exercise_current": exercise_current,
    "exercise_as_of_day_2": exercise_past,
    "import_count": len(imports),
    "scope": "只驗證已按日期排列、單一活動、明確替換欄位的有限範例；未呼叫或測量語言模型。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
