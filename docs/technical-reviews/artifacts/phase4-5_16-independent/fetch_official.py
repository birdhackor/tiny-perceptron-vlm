"""Fetch only pinned official API/source text; no data or model artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

BASE = Path(__file__).resolve().parent
sources = BASE / "sources"
sources.mkdir(exist_ok=True)
urls = {
    "pytorch-v2.14.1-loss.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/loss.py",
    "pytorch-v2.14.1-functional.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py",
    "python-v3.13.5-random.py": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Lib/random.py",
}
receipts = []
for name, url in urls.items():
    with urllib.request.urlopen(url, timeout=25) as response:
        raw, status, final_url = response.read(), response.status, response.url
    (sources / name).write_bytes(raw)
    receipts.append({"path": str((sources / name).relative_to(BASE)), "url": url, "final_url": final_url, "http_status": status, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "accessed_on": "2026-10-05"})
    print(name, status, len(raw), hashlib.sha256(raw).hexdigest())
(sources / "retrieval.json").write_text(json.dumps(receipts, indent=2) + "\n")
print("python", sys.version)
print("python_executable", sys.executable)
