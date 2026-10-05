"""Re-score original saved samples with selected historical pure functions.

Only data generation and metrics are compiled from the revision named by the
raw result. No module top-level imports, checkpoints, generation or fit run.
"""
import ast
import copy
import hashlib
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
revision = "ae7bbbf95537d228a44810041d2a9e978360d369"
path = BASE / "inputs/current/docs/course-experiments/results/style.json"
raw = path.read_bytes()
document = json.loads(raw)
assert document["revision"] == revision
result = document["results"]
namespace = {"json": json, "hashlib": hashlib, "random": random}


def load_selected(filename, names):
    source = (BASE / "sources" / filename).read_bytes()
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    assert all(not n.decorator_list for n in nodes)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    print("historical pure functions", filename, sorted(names), hashlib.sha256(source).hexdigest())


load_selected("historical-common.py", ["split_records"])
load_selected("historical-text.py", ["arithmetic_records"])
load_selected("historical-behavior.py", ["_conversation", "_style_record", "_style_metrics"])
split_records = namespace["split_records"]
style_record = namespace["_style_record"]
conversation = namespace["_conversation"]
score = namespace["_style_metrics"]
arithmetic = split_records(namespace["arithmetic_records"](), seed=document["seed"])
conditional = {
    split: [style_record(row, style, True) for row in rows for style in ("concise", "vivid", "json")]
    for split, rows in arithmetic.items()
}
# Exact short date-data construction from historical run_style, lines 124-139.
dates = []
for day in range(1, 25):
    date = f"2026-10-{day:02d}"
    dates.append(conversation(f"task=date;date={date};confirm", "已確認" + date + "。", "date-" + date, style="clarification"))
    dates.append(conversation(f"task=date;id={day};date=?;confirm", "請提供日期。", "date-" + date, style="clarification"))
date_parts = split_records(dates, seed=document["seed"])
conditional = {split: rows + date_parts[split] for split, rows in conditional.items()}


def verify_split_hashes(parts, manifest, title):
    for split, rows in parts.items():
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        assert digest == manifest[split]["sha256"]
        assert len(rows) == manifest[split]["records"]
        print(title, split, "records", len(rows), "original JSONL SHA256 matched", digest)
    families = [{r["family"] for r in rows} for rows in parts.values()]
    assert all(not left & right for i, left in enumerate(families) for right in families[i + 1:])


verify_split_hashes(arithmetic, result["arithmetic_data"], "arithmetic")
verify_split_hashes(conditional, result["data"], "conditional")
summary = {"original_result_sha256": hashlib.sha256(raw).hexdigest(), "revision": revision, "no_training_or_generation": True, "after": {}}
for split in ("validation", "test"):
    saved = result["after"][split]
    samples, rows = saved["samples"], conditional[split]
    assert len(samples) == len(rows) == saved["records"]
    for row, sample in zip(rows, samples, strict=True):
        assert sample["messages"] == row["messages"][:-1]
        assert sample["expected"] == row["messages"][-1]["content"]
        ids = sample["generated_ids"]
        assert all(type(x) is int and 0 <= x < 264 for x in ids)
        answer_ids = ids[:ids.index(2)] if 2 in ids else ids
        assert all(x >= 8 for x in answer_ids), "hidden control token in saved answer"
        assert bytes(x - 8 for x in answer_ids).decode("utf-8") == sample["generated"]
        assert sample["eos"] == (2 in ids)
        assert sample["exact"] == (answer_ids == [x + 8 for x in sample["expected"].encode("utf-8")])
    recalculated = score(copy.deepcopy(saved), rows)
    assert recalculated["rubric"] == saved["rubric"]
    for original, rescored in zip(samples, recalculated["samples"], strict=True):
        for key in ("content_correct", "style_correct"):
            if key in original:
                assert rescored[key] == original[key]
    assert saved["eos_rate"] == sum(s["eos"] for s in samples) / len(samples)
    summary["after"][split] = recalculated["rubric"]
    print("re-scored", split, json.dumps(recalculated["rubric"], ensure_ascii=False))
    if split == "test":
        for row, sample in zip(rows, samples, strict=True):
            if "a" in row:
                print(sample["messages"][0]["content"], "=>", sample["generated"], "expected", row["a"] + row["b"], "content", sample["content_correct"], "style", sample["style_correct"])
        for style in ("concise", "vivid", "json"):
            m = recalculated["rubric"][style]
            assert m["records"] == m["style_correct"] == 7
            assert m["content_correct"] == 0

for style, probe in result["prompt_only_comparison_same_weights"].items():
    rows = [style_record(r, style, True) for r in arithmetic["test"]]
    assert len(probe["samples"]) == len(rows) == 7
    recorded = [s for s in result["after"]["test"]["samples"] if s.get("style") == style]
    assert [s["generated"] for s in probe["samples"]] == [s["generated"] for s in recorded]
    print("saved same-weight probe agrees", style, "7 samples")

for style in ("concise", "vivid"):
    parts = {split: [style_record(row, style, False) for row in rows] for split, rows in arithmetic.items()}
    run = result["default_style_runs"][style]
    verify_split_hashes(parts, run["data"], "default-" + style)
    for split in ("validation", "test"):
        assert score(copy.deepcopy(run["after"][split]), parts[split])["rubric"] == run["after"][split]["rubric"]

# A bounded failure boundary of the actual narrow historical checker.
row = style_record({"a": 2, "b": 2, "family": "2+2"}, "vivid", True)
probe = {"samples": [{"generated": "5，像把兩組積木合在一起再數。", "exact": False}]}
boundary = score(probe, [row])["samples"][0]
assert boundary["style_correct"] is True and boundary["content_correct"] is False
print("boundary: fixed metaphor can pass style while arithmetic fails")
print("all comparisons exact; no float score tolerance used; denominator = records per split per style")
print("original GPU metadata is provenance only:", document["device"], document["gpu"], document["torch_version"])
print("no checkpoint loaded; no training, token generation, or GPU run")
(BASE / "execution/saved-style-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
