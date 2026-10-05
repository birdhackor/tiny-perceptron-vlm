"""Fetch narrowly selected original authorities with TLS verification; no datasets or weights."""
import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCES = BASE / "sources"
records = []

def fetch(url, target):
    request = urllib.request.Request(url, headers={"User-Agent": "independent-factual-review-13.2"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        record = {"url": url, "final_url": response.url, "status": response.status,
                  "accessed_at": datetime.now(UTC).isoformat(), "path": str(target),
                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    target.write_bytes(raw)
    records.append(record)
    print(json.dumps(record, ensure_ascii=False))
    return raw

try:
    revision = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
    card_url = f"https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/resolve/{revision}/README.md"
    card = fetch(card_url, SOURCES / "ultrafeedback-binarized-pinned-README.md")
    local = Path("data/training/behavior-initial/ultrafeedback-dpo/source-README.md").read_bytes()
    print(json.dumps({"pinned_card_matches_local_original": card == local}))
    api_url = "https://api.github.com/repos/eric-mitchell/direct-preference-optimization/commits/main"
    metadata = fetch(api_url, SOURCES / "dpo-author-repository-head.json")
    commit = json.loads(metadata)["sha"]
    for name in ["trainers.py", "preference_datasets.py"]:
        url = f"https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/{commit}/{name}"
        fetch(url, SOURCES / ("author-dpo-" + name))
    print(json.dumps({"dpo_author_repository_commit": commit}))
finally:
    (BASE / "authority-fetch.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
