"""Bounded CPU review: select hypothetical scores and inspect data splits; no model updates."""

import contextlib
import hashlib
import io
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.course_experiments.text import _deduplicate_text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


raw = (ROOT / "course/chapters/05.md").read_text()
start = raw.index("## 5.10 ")
end = raw.index("\n## ", start + 1)
body = raw[start : end + 1]
code = re.search(r"```python\n(.*?)```", body, re.S)[1]
cases = []
for label, inserted, expected_lr, expected_score in [
    ("original", None, 0.01, 0.8),
    ("exercise_0.6", 0.6, 0.01, 0.8),
    ("exercise_0.9", 0.9, 0.1, 0.9),
]:
    example = code
    if inserted is not None:
        example = code.replace(
            ']\nselected', f'    {{"lr": 0.1, "validation": {inserted}}},\n]\nselected'
        )
    scope = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(example, f"5.10/{label}", "exec"), scope)
    selected = scope["selected"]
    assert selected == {"lr": expected_lr, "validation": expected_score}
    assert "test仍封存，尚未計算最終成績" in stdout.getvalue()
    assert "test" not in scope
    cases.append({"case": label, "selected": selected, "stdout": stdout.getvalue(), "test_computed": False})

historical_path = ROOT / "docs/course-experiments/results/real_text.json"
historical = json.loads(historical_path.read_text())
data_base = ROOT / "data/training/text-initial"
audits = {}
for identifier, filename, expected_counts in [
    ("tinystories", "tinystories-train-512.jsonl", [409, 51, 52]),
    ("chinese-poetry", "chinese-classical-train-365.jsonl", [292, 36, 37]),
]:
    path = data_base / filename
    original = read_rows(path)
    parts = split_records(_deduplicate_text(original), seed=42)
    run = historical["results"]["runs"][identifier]
    expected_asset = next(a for a in historical["assets"] if a["id"] == identifier)
    expected_file = next(f for f in expected_asset["files"] if f["path"].endswith(filename))
    assert sha(path) == expected_file["sha256"]
    family_sets = {split: {r["family"] for r in rows} for split, rows in parts.items()}
    overlaps = {
        f"{a}/{b}": len(family_sets[a] & family_sets[b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }
    assert not any(overlaps.values())
    split_results = {}
    for index, (split, rows) in enumerate(parts.items()):
        encoded = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        assert len(rows) == expected_counts[index]
        assert digest == run["data"][split]["sha256"]
        examples = text_examples(rows, mode="text", max_length=128)
        assert all(len(x) <= 128 and len(x) == len(y) for x, y in examples)
        split_results[split] = {
            "records": len(rows), "families": len(family_sets[split]),
            "jsonl_sha256": digest, "matches_historical_split": True,
            "windows": len(examples),
            "effective_targets": sum(len(y) for x, y in examples),
            "matches_source_bytes_plus_bos_eos_targets":
                sum(len(y) for x, y in examples) == sum(len(r["text"].encode()) + 1 for r in rows),
        }
        assert split_results[split]["matches_source_bytes_plus_bos_eos_targets"]
    assert records_sha256(parts["train"]) == run["training"]["records_sha256"]
    complete_check = {}
    if identifier == "tinystories":
        source = data_base / "tinystories-train-prefix-complete.txt"
        stories = [x.strip() for x in source.read_text().split("<|endoftext|>") if x.strip()]
        assert [r["text"] for r in original] == stories
        assert all(r["source_split"] == "train" for r in original)
        complete_check = {"whole_prefix_stories_match": True, "prefix_stories": len(stories),
                          "prefix_sha256": sha(source), "source_splits": ["train"]}
    else:
        source = data_base / "tang300-source.json"
        poems = json.loads(source.read_text())
        rendered = ["\n".join([p["title"], p["author"], *p["paragraphs"]]) for p in poems]
        unique = list(dict.fromkeys(rendered))
        assert [r["text"] for r in original] == unique
        assert all(r["source_split"] == "unpartitioned" for r in original)
        complete_check = {"whole_poem_renderings_match": True, "source_records": len(poems),
                          "unique_complete_poems": len(unique), "upstream_sha256": sha(source),
                          "source_splits": ["unpartitioned"]}
    audits[identifier] = {
        "asset_path": str(path.relative_to(ROOT)), "asset_sha256": sha(path),
        "matches_historical_asset": True, "completeness": complete_check,
        "splits": split_results, "family_overlap_counts": overlaps,
        "training_records_hash_matches_historical": True,
        "historical_training_steps": run["training"]["steps"],
        "historical_effective_training_tokens": run["training"]["effective_tokens"],
        "preserved_warning": run["split_warning"],
        "preserved_scope": run["scope"],
    }

code_matches = {
    filename: sha(ROOT / filename) == historical["code_sha256"][filename]
    for filename in ["scripts/course_experiments/text.py", "scripts/course_experiments/common.py"]
}
assert all(code_matches.values())
print(json.dumps({
    "reviewer_task": "/root/repair_tech_5_10",
    "scope": "Actual current CPU selection code and data-only split reconstruction; historical training NOT rerun",
    "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
                    "platform": platform.platform()},
    "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
    "selection_cases": cases,
    "historical_result_path": str(historical_path.relative_to(ROOT)),
    "historical_result_sha256": sha(historical_path),
    "historical_environment": {k: historical[k] for k in ["revision", "seed", "device", "torch_version", "python_version", "step_scale"]},
    "historical_code_matches_current": code_matches,
    "data_only_audits": audits,
    "model_updates_this_review": 0,
}, ensure_ascii=False, indent=2))
