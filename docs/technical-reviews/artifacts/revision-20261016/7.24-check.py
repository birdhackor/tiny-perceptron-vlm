"""7.24 獨立技術核對；不建模型、不訓練、不更新教材。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import torch

from scripts.reading_time import lesson_slices
from tiny_perceptron.data import ByteTokenizer, IGNORE, render_chat


root = Path(__file__).resolve().parents[4]
source = root / "course/chapters/07.md"
body = lesson_slices(source.read_bytes().decode("utf-8"))["7.24"]
freeze = json.loads((root / "docs/course-revision-20261016/freeze-08.json").read_text())
frozen = next(item for item in freeze["pages"] if item["page_id"] == "7.24")
digest = hashlib.sha256(body.encode()).hexdigest()
assert digest == frozen["source_sha256"]
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 1
namespace = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec(compile(blocks[0], "course/chapters/07.md#7.24", "exec"), namespace)
records = namespace["records"]
expected_output = (
    "<low> 先有2件，新增3件，再移除1件，剩幾件？ -> 4\n"
    "<high> 先有2件，新增3件，再移除1件，剩幾件？ -> 2+3=5；5-1=4。答案4。\n"
)
assert captured.getvalue() == expected_output
assert len(records) == 2 and records[0]["question"] == records[1]["question"]

tok = ByteTokenizer()
serialized = []
for row in records:
    prompt = f"<{row['effort']}> {row['question']}"
    x, y = render_chat([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["answer"]},
    ])
    targets = y[y != IGNORE].tolist()
    expected_targets = tok.encode(row["answer"]) + [tok.eos_id]
    assert targets == expected_targets
    assert targets[-1] == tok.eos_id
    assert prompt in tok.decode(x.tolist())
    serialized.append({
        "effort": row["effort"],
        "input_contains_effort": True,
        "input_length": x.numel(),
        "input_dtype": str(x.dtype),
        "answer_utf8_bytes": len(row["answer"].encode("utf-8")),
        "supervised_tokens_including_eos": len(targets),
        "nonanswer_labels_ignored": True,
        "last_supervised_token": targets[-1],
        "supervised_answer_roundtrip": tok.decode(targets),
    })
assert serialized[1]["supervised_tokens_including_eos"] > serialized[0]["supervised_tokens_including_eos"]
assert 2 + 3 == 5 and 5 - 1 == 4 and 2 + 3 - 1 == 4
assert 2 + 3 != 6 and 6 - 1 != 4

result = {
    "reviewer_task": "/root/tech_20261016_7_24",
    "command": "uv run python docs/technical-reviews/artifacts/revision-20261016/7.24-check.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "platform": platform.platform()},
    "source_sha256": digest,
    "freeze08_matches": True,
    "original_code_stdout": captured.getvalue(),
    "original_code_result": "Exact output matched; original block contains only records, iteration, and printing. No model or optimizer is constructed.",
    "serialization_check": "Additional independent interface check using the explicitly linked 7.11 render_chat, with effort serialized as ordinary user-content bytes. This is not a trained low/high assistant.",
    "serialization": serialized,
    "arithmetic": {"correct": ["2+3=5", "5-1=4", "2+3-1=4"], "deliberately_wrong": ["2+3=6", "6-1=4"], "both_wrong_equations_rejected": True},
    "optimizer_steps": 0,
    "scope": "CPU data formatting, mask/EOS, token accounting and arithmetic only; no training, no GPU measurement, no quality comparison.",
}
out = root / "docs/technical-reviews/artifacts/revision-20261016/7.24-execution.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
