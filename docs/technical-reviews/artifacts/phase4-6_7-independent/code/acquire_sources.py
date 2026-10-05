"""Fetch authoritative, version-pinned text only; no datasets or models."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parents[1]
SOURCES = {
    "python-standardtypes.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst",
    "python-codecs.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/codecs.rst",
    "python-utf8.py": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Lib/encodings/utf_8.py",
    "tokenizers-byte-level.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.1/tokenizers/src/pre_tokenizers/byte_level.rs",
    "tokenizers-bpe-trainer.rs": "https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.1/tokenizers/src/models/bpe/trainer.rs",
    "unicode-uax29.html": "https://www.unicode.org/reports/tr29/tr29-47.html",
    "rfc3629.txt": "https://www.rfc-editor.org/rfc/rfc3629.txt",
}

def fetch(item):
    name, url = item
    path = BASE / "sources" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=35) as response:
        raw = response.read()
        record = {"file": str(path.relative_to(BASE)), "url": url,
                  "final_url": response.geturl(), "http_status": response.status,
                  "accessed_at": datetime.now(timezone.utc).isoformat(),
                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    path.write_bytes(raw)
    return record

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(fetch, SOURCES.items()))
(BASE / "sources/provenance.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
