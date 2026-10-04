"""Independently count the exact recorded UTF-8 prompt and generated IDs."""
import json
import platform
from pathlib import Path

root = Path(__file__).resolve().parents[5]
record = json.loads((root / "docs/course-experiments/results/modern.json").read_text())
sample = record["results"]["variants"]["rope"]["heldout"]["test"]["samples"][0]
assert sample["prompt"] == "Once upon a time, there "
assert sample["generated"] == "was a loked there was a bough a "
count = len(sample["prompt"].encode("utf-8"))
assert count == 24
assert len(sample["generated_ids"]) == 32
print(json.dumps({
    "environment": {"python": platform.python_version(), "device": "cpu"},
    "prompt": sample["prompt"], "prompt_utf8_bytes": count,
    "generated": sample["generated"], "generated_ids": len(sample["generated_ids"]),
    "all_assertions_passed": True,
}, ensure_ascii=False, indent=2))
