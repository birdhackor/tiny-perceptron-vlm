"""5.12 的短 CPU 核對；只算片段與既有資料切分，不建立或訓練模型。"""

import contextlib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import _deduplicate_text
from tiny_perceptron.data import fingerprint


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, identifier):
    match = re.search(r"^## " + re.escape(identifier) + r" .*?(?=^## |\Z)", raw.decode(), re.M | re.S)
    if match is None:
        raise ValueError(identifier)
    return match[0].encode()


def execute(code):
    namespace = {}
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, "frozen-5.12-code", "exec"), namespace)
    return namespace, stream.getvalue()


freeze_path = "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(item for item in freeze["pages"] if item["page_id"] == "5.12")
raw = (ROOT / page["snapshot"]).read_bytes()
assert digest(raw) == page["source_sha256"]
assert section((ROOT / page["source"]).read_bytes(), "5.12") == raw
assert page["figures_sha256"] == {}
blocks = re.findall(r"```python\n(.*?)```", raw.decode(), re.S)
assert len(blocks) == 1
namespace, base_stdout = execute(blocks[0])
grams = namespace["grams"]
left, right = "紅色圓形在左邊", "紅色圓形在右邊"
assert len(left) == len(right) == 7
assert namespace["common"] == {"紅色", "色圓", "圓形", "形在"}
assert len(namespace["a"]) == len(namespace["b"]) == 6
assert len(namespace["all_parts"]) == 8 and namespace["score"] == 0.5
common_before = namespace["common"].copy()
sorted(namespace["common"])
assert namespace["common"] == common_before
assert fingerprint(left) != fingerprint(right)

exercise_code = blocks[0].replace('grams("紅色圓形在左邊")', 'grams("紅色圓形在左邊", n=3)')
exercise_code = exercise_code.replace('grams("紅色圓形在右邊")', 'grams("紅色圓形在右邊", n=3)')
exercise, exercise_stdout = execute(exercise_code)
assert len(exercise["a"]) == len(exercise["b"]) == 5
assert exercise["common"] == {"紅色圓", "色圓形", "圓形在"}
assert len(exercise["all_parts"]) == 7 and exercise["score"] == 3 / 7
assert abs(exercise["score"] - 0.429) < 0.0005

edge_cases = []
for x, y in [("", ""), ("左", "右")]:
    a, b = grams(x), grams(y)
    try:
        len(a & b) / len(a | b)
    except ZeroDivisionError:
        edge_cases.append({"texts": [x, y], "grams": [sorted(a), sorted(b)], "result": "ZeroDivisionError"})
    else:
        raise AssertionError("empty-union division should fail")
assert grams("甲甲甲") == {"甲甲"}

limitations = []
for x, y in [("小貓睡著了", "幼貓正在熟睡"), ("甲乙甲", "乙甲乙")]:
    a, b = grams(x), grams(y)
    limitations.append({"texts": [x, y], "grams": [sorted(a), sorted(b)], "jaccard": len(a & b) / len(a | b)})
assert limitations[0]["jaccard"] == 0 and limitations[1]["jaccard"] == 1

synthetic = [
    {"text": left, "answer": "左", "family": "red-circle"},
    {"text": right, "answer": "右", "family": "red-circle"},
    {"text": "藍色方形在左邊", "answer": "左", "family": "blue-square"},
    {"text": "綠色三角在右邊", "answer": "右", "family": "green-triangle"},
    {"text": "黃色方形在左邊", "answer": "左", "family": "yellow-square"},
]
synthetic_parts = split_records(synthetic, seed=42)
family_locations = {}
for split, rows in synthetic_parts.items():
    for row in rows:
        family_locations.setdefault(row["family"], set()).add(split)
assert all(len(splits) == 1 for splits in family_locations.values())
assert {row["answer"] for rows in synthetic_parts.values() for row in rows if row["family"] == "red-circle"} == {"左", "右"}

result_path = "docs/course-experiments/results/real_text.json"
recorded = json.loads((ROOT / result_path).read_bytes())
split_replay = {}
for name, filename in [("tinystories", "tinystories-train-512.jsonl"), ("chinese-poetry", "chinese-classical-train-365.jsonl")]:
    source_path = ROOT / "data/training/text-initial" / filename
    asset = next(item for item in recorded["assets"] if item["id"] == name)
    file_info = next(item for item in asset["files"] if item["path"] == "text-initial/" + filename)
    source_bytes = source_path.read_bytes()
    assert len(source_bytes) == file_info["bytes"] and digest(source_bytes) == file_info["sha256"]
    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    deduplicated = _deduplicate_text(rows)
    parts = split_records(deduplicated, seed=recorded["seed"])
    expected = recorded["results"]["runs"][name]
    assert len(rows) == expected["source_records"]
    assert len(deduplicated) == expected["deduplicated_records"]
    assert expected["split_warning"] == "original source split is not our test; near-duplicate clustering was not performed"
    split_info = {}
    normalized_by_split = {}
    for split, selected in parts.items():
        serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
        sha = digest(serialized)
        families = {row["family"] for row in selected}
        assert sha == expected["data"][split]["sha256"]
        assert len(selected) == expected["data"][split]["records"]
        assert len(families) == expected["data"][split]["families"]
        normalized_by_split[split] = {" ".join(row["text"].split()) for row in selected}
        assert all(row["family"] == fingerprint(row["text"]) for row in selected)
        split_info[split] = {"records": len(selected), "families": len(families), "sha256": sha, "matches_recorded": True}
    intersections = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        intersections[a + "/" + b] = len(normalized_by_split[a] & normalized_by_split[b])
    assert set(intersections.values()) == {0}
    split_replay[name] = {
        "source_path": str(source_path),
        "source_bytes": len(source_bytes),
        "source_sha256": digest(source_bytes),
        "source_matches_published_asset_receipt": True,
        "source_records": len(rows),
        "deduplicated_records": len(deduplicated),
        "split_warning": expected["split_warning"],
        "splits": split_info,
        "normalized_full_content_intersections": intersections,
    }

prerequisites = []
for file, identifier, page_id in [
    ("course/chapters/05.md", "5.11", "5.11"),
    ("course/first-steps.md", "W.2", "first-steps"),
    ("course/chapters/01.md", "1.2", "1.2"),
    ("course/chapters/02.md", "2.5", "2.5"),
    ("course/training.md", "T.4", "training"),
]:
    current = section((ROOT / file).read_bytes(), identifier)
    frozen_page = next(item for item in freeze["pages"] if item["page_id"] == page_id)
    frozen = (ROOT / frozen_page["snapshot"]).read_bytes()
    if frozen_page["selector"] == "whole-file":
        frozen = section(frozen, identifier)
    prerequisites.append({"path": file, "selector": "section:" + identifier, "current_slice_sha256": digest(current), "freeze_slice_sha256": digest(frozen), "current_matches_freeze_slice": current == frozen})

code_receipts = {}
for path in ["tiny_perceptron/data.py", "scripts/course_experiments/common.py", "scripts/course_experiments/text.py", "scripts/course_experiments/run.py"]:
    current = digest((ROOT / path).read_bytes())
    code_receipts[path] = {"current_sha256": current, "recorded_run_sha256": recorded["code_sha256"][path], "whole_file_equal_to_recorded_run": current == recorded["code_sha256"][path]}

output = {
    "created_at": datetime.now(UTC).isoformat(),
    "reviewer_task": "/root/omni_technical_coordinator/technical_5_12",
    "command": "uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-technical_5_12-v1/verify-v1.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "freeze": {"path": freeze_path, "sha256": digest((ROOT / freeze_path).read_bytes()), "source_snapshot": page["snapshot"], "source_sha256": digest(raw), "current_exact_slice_matches": True, "figure_sha256": {}},
    "embedded_code_sha256": digest(blocks[0].encode()),
    "base_stdout": base_stdout,
    "base": {"text_lengths": [len(left), len(right)], "starts": list(range(6)), "sets": [sorted(namespace["a"]), sorted(namespace["b"])], "intersection": sorted(namespace["common"]), "union": sorted(namespace["all_parts"]), "score": namespace["score"], "fingerprints": [fingerprint(left), fingerprint(right)], "sorted_did_not_mutate_common": True},
    "exercise_code_sha256": digest(exercise_code.encode()),
    "exercise_stdout": exercise_stdout,
    "exercise": {"starts": list(range(5)), "sets": [sorted(exercise["a"]), sorted(exercise["b"])], "intersection": sorted(exercise["common"]), "union": sorted(exercise["all_parts"]), "score": exercise["score"], "round_3dp": round(exercise["score"], 3)},
    "edge_cases": edge_cases,
    "repeated_gram_example": {"text": "甲甲甲", "grams": sorted(grams("甲甲甲"))},
    "limitations_witnesses": limitations,
    "family_demo": {"splits": synthetic_parts, "family_locations": {key: sorted(value) for key, value in family_locations.items()}, "both_answers_preserved": True},
    "existing_result": {"path": result_path, "sha256": digest((ROOT / result_path).read_bytes()), "revision": recorded["revision"], "original_device": recorded["device"], "original_seed": recorded["seed"], "original_evidence_status": recorded["evidence_status"]},
    "real_text_split_replay": split_replay,
    "current_prerequisites_read": prerequisites,
    "code_receipts": code_receipts,
    "scope": "Exact frozen example and n=3 exercise; CPU-only deterministic existing-data replay. No model constructed, no training, no quality or speed rerun, and no claim that near-duplicate families are separated.",
    "result": "all assertions passed",
}
target = Path(__file__).with_name("execution-v1.json")
with target.open("x", encoding="utf-8") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({"output": str(target.relative_to(ROOT)), "sha256": digest(target.read_bytes()), "result": output["result"], "base_stdout": base_stdout, "exercise_stdout": exercise_stdout, "split_counts": {name: {split: info["records"] for split, info in run["splits"].items()} for name, run in split_replay.items()}}, ensure_ascii=False))
