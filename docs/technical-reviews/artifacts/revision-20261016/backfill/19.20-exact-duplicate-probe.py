"""Find one exact four-split observation witness, including candidate semantics."""
import hashlib
import json
import platform
from pathlib import Path

root = Path(__file__).resolve().parents[5]
source = root / "outputs/lesson-products/data-v2/c.jsonl"
seen = {}
with source.open() as stream:
    for line in stream:
        row = json.loads(line)
        if row["state"] == "卡片進度;申請編號=?" and row["question"] == "info" and sorted(row["candidates"]) == ["no", "yes"] and row["gold"] == "no" and row["split"] in ("train", "calibration", "policy", "test"):
            seen.setdefault(row["split"], {k: row[k] for k in ("id", "family_id", "state", "question", "candidates", "gold")})
        if len(seen) == 4:
            break
assert len(seen) == 4
out = {
    "command": "python docs/technical-reviews/artifacts/revision-20261016/backfill/19.20-exact-duplicate-probe.py",
    "environment": {"python": platform.python_version(), "device": "CPU; stdlib JSON scan"},
    "source": "outputs/lesson-products/data-v2/c.jsonl",
    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "witnesses": seen,
    "result": "Identical state/question/candidate meanings/label found in all four core splits with different family IDs. No inference.",
}
Path(__file__).with_name("19.20-exact-duplicate-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(out["result"])
