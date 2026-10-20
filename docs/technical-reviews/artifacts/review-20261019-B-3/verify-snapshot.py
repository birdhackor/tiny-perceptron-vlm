"""原始判斷存檔後，逐筆比對課文連結的輸入重建紀錄。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.applications import _prompt_ids
from tiny_perceptron.data import ByteTokenizer


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


base = "docs/technical-reviews/artifacts/review-20261019-B-3"
original_path = "docs/course-review-20261019/reviews/rv1019_technical_B-3/technical-B-3-original-01.json"
original_sha = digest(original_path)
assert original_sha == "b92a870a2f83cb3676c85038fc284a38a842ad74136715522a6cd573a6e02b60"
original = json.loads((ROOT / original_path).read_bytes())
assert original["verdict"] == "revise"
assert [c["id"] for c in original["claims"] if c["status"] != "verified"] == ["c20"]

facts_path = base + "/facts-execution.json"
facts = json.loads((ROOT / facts_path).read_bytes())
independent = {r["original_json_pointer"]: r for r in facts["independent_snapshots"]}
source_path = "docs/course-experiments/results/tools.json"
source = json.loads((ROOT / source_path).read_bytes())
snapshot_path = "docs/course-repair-20261008/evidence/tool-input-snapshot-reconstruction.json"
snapshot = json.loads((ROOT / snapshot_path).read_bytes())
assert snapshot["original_file"] == source_path
assert snapshot["original_sha256"] == digest(source_path)

tok, seen, checked = ByteTokenizer(), set(), []
for row in snapshot["records"]:
    pointer = row["original_json_pointer"]
    assert pointer not in seen
    seen.add(pointer)
    value = source
    for part in pointer.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    ids = _prompt_ids(row["messages_at_generation"], tok)
    assert ids == value["input_ids"] == row["actual_input_ids"]
    assert len(ids) == value["input_tokens"] == row["actual_input_tokens"]
    assert row["messages_at_generation"] == independent[pointer]["messages_at_generation"]
    assert row["original_metadata_differs"] == (row["messages_at_generation"] != value["messages"])
    assert row["all_ids_verified_equal"] is True
    checked.append({"original_json_pointer": pointer, "input_tokens": len(ids),
                    "matches_original_input_ids": True, "matches_independent_reconstruction": True})
assert seen == set(independent)
assert len(checked) == snapshot["records_verified"] == 56
assert sum(r["original_metadata_differs"] for r in snapshot["records"]) == snapshot["metadata_mismatches"] == 36
print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + base + "/verify-snapshot.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "source_receipts": {name: digest(name) for name in [original_path, facts_path, source_path, snapshot_path]},
    "read_order": "先存下完整原始判斷與獨立重建，之後才讀課文所連的舊重建檔；未讀其他歷史審閱或處置。",
    "records_checked": len(checked),
    "metadata_mismatches_checked": 36,
    "records": checked,
    "result": "所有 assertion 通過；課文連結檔 56/56 筆與原始輸入 ID 及獨立重建逐筆相同。",
}, ensure_ascii=False, indent=2))
