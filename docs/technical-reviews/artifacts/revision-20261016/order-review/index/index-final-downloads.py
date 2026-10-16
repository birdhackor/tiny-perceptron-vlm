"""Inspect strict preview downloads via HTTP; never run Notebook kernels."""

import hashlib
import json
import platform
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "docs/technical-reviews/artifacts/revision-20261016/order-review/index"
URL = "http://127.0.0.1:8898/"


def fetch(path):
    with urllib.request.urlopen(URL + path, timeout=15) as response:
        return response.read()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(item):
    raw = fetch(item["notebook"])
    return {"id": item["id"], "source_sha256": digest((ROOT / item["notebook"]).read_bytes()), "download_sha256": digest(raw)}


def main():
    index = json.loads((ROOT / "course/lesson-index.json").read_text())
    with ThreadPoolExecutor(max_workers=6) as workers:
        rows = list(workers.map(check, index))
    info = json.loads(fetch("build-info.json"))
    page = fetch("index.html")
    changed = next(row for row in rows if row["id"] == "19.12")
    result = {
        "reviewer_task": "/root/tech_20261016_index", "current_context": "informed", "recorded_at": datetime.now(UTC).isoformat(),
        "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/index/index-final-downloads.py",
        "environment": {"python": platform.python_version(), "torch": version("torch"), "device": "CPU; local HTTP file reads only", "platform": platform.platform()},
        "checks": {
            "preview_url": URL, "author_preview_build_info": info,
            "preview_home_sha256": digest(page),
            "navigation_unique_lesson_count": len(set(re.findall(rb'href="(?:\./)?([A-Z\d]+\.\d+)\.html"', page))),
            "http_downloads_checked": len(rows),
            "byte_identical_to_current_notebooks": sum(row["source_sha256"] == row["download_sha256"] for row in rows),
            "failures": [row for row in rows if row["source_sha256"] != row["download_sha256"]],
            "rows_digest": digest(json.dumps(rows, sort_keys=True).encode()),
            "19.12": changed,
            "19.12_download_contains_19.13_transition": "接著在[19.13]" in fetch("notebooks/19/19.12.ipynb").decode(),
        },
        "limits": "This reviewer only fetched and hashed author-built preview files. No site build, Notebook kernel, model training, formal tests, HF download, or reading-time validation. Earlier 8897 mismatch retained in index-execution.json; no earlier artifact was overwritten.",
    }
    (BASE / "index-final-downloads.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
