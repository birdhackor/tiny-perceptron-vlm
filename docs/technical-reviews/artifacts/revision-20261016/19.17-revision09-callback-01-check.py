"""核對新版快照、兩盒交換算例與所連家族分配短例；不執行模型。"""
import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path

from scripts.reading_time import lesson_slices

root = Path(__file__).resolve().parents[4]
freeze_path = root / "docs/course-revision-20261016/freeze-09.json"
freeze = json.loads(freeze_path.read_text())
records = {item["page_id"]: item for item in freeze["pages"]}
contexts = {}
for lesson in ("19.17", "19.16", "19.14", "19.13", "19.15"):
    record = records[lesson]
    raw = (root / record["snapshot"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == record["source_sha256"]
    contexts[lesson] = {"snapshot": record["snapshot"], "sha256": digest}
source = root / "course/chapters/19.md"
live = lesson_slices(source.read_bytes().decode("utf-8"))["19.17"]
assert live.encode() == (root / records["19.17"]["snapshot"]).read_bytes()


def swap(a, b):
    return b, a


complete = {"A": 2, "B": 6}
a_after, b_after = swap(complete["A"], complete["B"])
assert (a_after, b_after) == (6, 2)
completions = []
for a_before in (2, 5):
    a_end, b_end = swap(a_before, 6)
    completions.append({"A_initial": a_before, "B_initial": 6, "A_final": a_end, "B_final": b_end})
assert len({item["B_final"] for item in completions}) == 2

family_body = (root / records["19.14"]["snapshot"]).read_text()
family_code = re.search(r"```python\n(.*?)```", family_body, re.S)[1]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    exec(compile(family_code, "freeze09-19.14-linked-snippet", "exec"), {})
family_lines = buffer.getvalue().strip().splitlines()
assert len(family_lines) == 3
assert len({line.rsplit(" ", 1)[1] for line in family_lines}) == 1

original_receipts = json.loads(
    (root / "docs/technical-reviews/artifacts/revision-20261016/19.17-primary-receipts.json").read_text()
)
primary_hash_checks = []
for receipt in original_receipts["sources"]:
    path = Path(receipt["local_fulltext"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == receipt["original_sha256"]
    primary_hash_checks.append({"id": receipt["id"], "sha256": actual, "same_as_initial_inspection": True})

result = {
    "reviewer_task": "/root/tech_20261016_19_17",
    "review_mode": "informed_callback",
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/19.17-revision09-callback-01-check.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "CPU; stdlib only; no model execution"},
    "snapshot_checks": contexts,
    "live_19.17_matches_freeze09": True,
    "complete_swap": {"input": complete, "operation": "simultaneous swap", "final": {"A": a_after, "B": b_after}, "query": "B", "answer": b_after},
    "missing_A_indistinguishable_completions": completions,
    "linked_19.14_exact_snippet_stdout": family_lines,
    "primary_hash_checks": primary_hash_checks,
    "result": "all deterministic checks passed",
    "limitations": ["新增兩盒例的獨立整數算例，不是B/C或服務實作驗證。", "家族短例只證明相同family_id穩定分配，不證明正式生成器已合併等價家族或無資料洩漏。", "沒有模型訓練、推論、校準、品質／延遲或GPU效果測試。"],
}
output = root / "docs/technical-reviews/artifacts/revision-20261016/19.17-revision09-callback-01-execution.json"
with output.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
