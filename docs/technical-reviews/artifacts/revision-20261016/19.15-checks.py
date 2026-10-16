"""19.15 獨立技術核對；只跑透明算例及必要前文短示意，沒有訓練。"""
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
sys.path.insert(0, str(ROOT))
from scripts.reading_time import lesson_slices

import torch

checks = []

def overwrite(initial, events):
    state = initial
    for event in events:
        state = event
    return state

forward = overwrite(0, [5, 1])
reverse = overwrite(0, [1, 5])
ablated = overwrite(0, [5])
assert (forward, reverse, ablated) == (1, 5, 5)
assert sorted([5, 1]) == sorted([1, 5])
checks.append({
    "name": "19.15 覆寫、倒序與刪除末事件",
    "input": {"initial": 0, "events": [5, 1]},
    "result": {"original": forward, "reversed": reverse, "delete_last": ablated},
    "scope": "手工世界真值。初始值任意，因第一個set即覆寫。相同數字多重集合可有不同答案；未呼叫模型。",
})
chap14 = lesson_slices((ROOT / "course/chapters/14.md").read_text())
for lesson in ["14.14", "14.15"]:
    body = chap14[lesson]
    code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, f"{lesson}-necessary-prerequisite", "exec"), {})
    result = output.getvalue()
    if lesson == "14.14":
        assert result == "本次終止 5 末端loss 0.6 逐輪平均 0.8400000000000001\n"
    else:
        assert result == "1 0.9\n2 0.1\n3 0.9\n4 0.1\n5 0.9\n6 0.1\n"
    checks.append({
        "name": lesson + " 必要前文手工示意",
        "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "stdout": result,
        "scope": "人工loss查表／變號反例，均無神經模型訓練；不能當B產品成績。",
    })
assert Fraction(90, 100) == Fraction(9, 10)
assert Fraction(85, 100) == Fraction(17, 20)
checks.append({
    "name": "預定門檻含義",
    "result": "狀態追蹤correct/N >= 9/10；算術correct/M >= 17/20。N、M為未來各自最後測試題數，沒有用假題數充當實測。",
})
print(json.dumps({
    "reviewer_task": "/root/tech_20261016_19_15",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.15-checks.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "platform": platform.platform()},
    "scope": "19.15與其明示第14章必要機制前置；未訓練B/C、未測GPU、未檢查網站版面。",
    "checks": checks,
    "result": "all assertions passed",
}, ensure_ascii=False, indent=2))
