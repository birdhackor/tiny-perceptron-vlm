"""Fetch only original papers/official documentation; no data or model downloads."""
import hashlib
import json
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parent / "sources"
OUT.mkdir(exist_ok=True)
URLS = {
    "tiny-episodic-memory-v1.pdf": "https://arxiv.org/pdf/1902.10486v1",
    "python-stdtypes-v3.13.5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst",
    "python-functions-v3.13.5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst",
    "python-expressions-v3.13.5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/expressions.rst",
}
records = []
for filename, url in URLS.items():
    with urllib.request.urlopen(url, timeout=25) as response:
        raw = response.read()
        final_url = response.url
    path = OUT / filename
    path.write_bytes(raw)
    record = {"url": url, "final_url": final_url, "path": filename,
              "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    records.append(record)
    print(json.dumps(record))
(OUT / "fetch-receipt.json").write_text(json.dumps(records, indent=2) + "\n")
