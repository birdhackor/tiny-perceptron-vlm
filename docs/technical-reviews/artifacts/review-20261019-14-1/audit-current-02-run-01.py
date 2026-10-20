"""限定比較命令來自指定練習，保留先前過廣枚舉的執行收據。"""

import hashlib
import json
from pathlib import Path

source_path = Path(__file__).with_name("audit-current-02.py")
source = source_path.read_text()
old = "for line in current.decode().splitlines():"
new = 'for line in section(current, "T.6").decode().splitlines():'
assert source.count(old) == 1
source = source.replace(old, new, 1)
old_command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-02.py"
new_command = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-02-run-01.py"
assert source.count(old_command) == 1
source = source.replace(old_command, new_command, 1)
receipt = {
    "initial_command": old_command,
    "initial_exit_code": 1,
    "initial_stderr": 'Traceback (most recent call last):\n  File "/Users/yclin/project/tiny-perceptron-vlm/docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-02.py", line 203, in <module>\n    main()\n    ~~~~^^\n  File "/Users/yclin/project/tiny-perceptron-vlm/docs/technical-reviews/artifacts/review-20261019-14-1/audit-current-02.py", line 90, in main\n    assert sorted(c["task"] for c in parsed_commands) == ["audio", "joint", "vision"]\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nAssertionError\n',
    "cause": "初次掃描整份training.md，額外納入入口區的一條vision通路檢查命令；不是教材或模型失敗。",
    "extra_command": ".venv/bin/python scripts/train.py --task vision --device cpu",
    "adjustment": "只枚舉〈讓圖片與聲音先有可用的基礎特徵〉的三條訓練命令；其餘驗證步驟不變。",
    "source_program_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
}
insertion = "    result[\"preparation_scope_adjustment\"] = " + repr(receipt) + "\n"
needle = '    with OUTPUT.open("x", encoding="utf-8") as target:'
assert source.count(needle) == 1
source = source.replace(needle, insertion + needle, 1)
exec(compile(source, str(source_path), "exec"), {"__file__": str(source_path), "__name__": "__main__"})
