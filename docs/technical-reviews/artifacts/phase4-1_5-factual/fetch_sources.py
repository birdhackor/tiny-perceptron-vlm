"""Fetch authoritative originals for an independent factual review; no model/data download."""
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import UTC, datetime
import hashlib
import json

root = Path(__file__).resolve().parent
sources = {
    "slp3-ngram.pdf": "https://web.stanford.edu/~jurafsky/slp3/3.pdf",
    "torch-2.9.0-_torch_docs.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_torch_docs.py",
    "torch-2.9.0-broadcasting.rst": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/notes/broadcasting.rst",
    "python-3.13-stdtypes.html": "https://docs.python.org/3.13/library/stdtypes.html",
}
receipt = []
for filename, url in sources.items():
    with urlopen(Request(url, headers={"User-Agent": "factual-review/1.5"}), timeout=40) as response:
        raw = response.read()
        destination = root / filename
        destination.write_bytes(raw)
        receipt.append({
            "url": url, "final_url": response.url, "http_status": response.status,
            "accessed_at": datetime.now(UTC).isoformat(), "path": str(destination.relative_to(root)),
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "content_type": response.headers.get("Content-Type"),
        })
        print(filename, response.status, len(raw), receipt[-1]["sha256"])
    (root / "source-downloads.json").write_text(json.dumps(receipt, indent=2) + "\n")
