"""Fetch bounded original documentation bytes, without model/data downloads."""
from pathlib import Path
from datetime import datetime, UTC
import hashlib
import json
import urllib.request

BASE = Path(__file__).resolve().parents[1]
RECORDS = [
    ("qwen3-vl-2b-instruct-readme.md", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/89644892e4d85e24eaac8bacfd4f463576704203/README.md", "Qwen model repository revision 89644892e4d85e24eaac8bacfd4f463576704203"),
    ("cpython-3.13.5-stdtypes.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst", "CPython v3.13.5"),
]
results = []
for filename, url, version in RECORDS:
    request = urllib.request.Request(url, headers={"User-Agent": "Independent factual review original-document reader"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Original documentation exceeds the bounded retrieval budget")
        final_url = response.url
        status = response.status
    path = BASE / "sources" / filename
    path.write_bytes(raw)
    results.append({"filename": filename, "url": url, "final_url": final_url, "version": version, "status": status, "accessed_at": datetime.now(UTC).isoformat(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    print(json.dumps(results[-1], ensure_ascii=False))
(BASE / "sources" / "primary-fetch.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
