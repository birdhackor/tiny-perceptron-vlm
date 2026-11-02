"""Portable recount from the reviewer's compact, already extracted gold/argmax rows."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

proof = Path(__file__).resolve().parent
source = proof / "compact-b5-rows-v1.json"
raw = source.read_bytes()
rows = json.loads(raw)
result = {"created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
          "device": "CPU; string comparison and rational arithmetic only",
          "input": source.name, "input_sha256": hashlib.sha256(raw).hexdigest(), "results": {},
          "scope": "Independent portable recount of retained gold/argmax vectors; argmax extraction from original logits is recorded in inspection-recomputation-v1.json."}
for task, row in rows["tasks"].items():
    assert len(row["gold"]) == len(row["argmax_depth5"])
    correct = sum(a == b for a, b in zip(row["gold"], row["argmax_depth5"], strict=True))
    count = len(row["gold"])
    threshold = 0.90 if task == "boxes" else 0.85
    result["results"][task] = {"correct": correct, "count": count, "accuracy": correct / count,
                                "families": row["families"], "threshold": threshold,
                                "pass": correct / count >= threshold}
assert result["results"]["boxes"]["pass"] and not result["results"]["sum"]["pass"]
with (proof / "compact-recount-v1.json").open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
