"""Fetch original official sources without changing TLS verification."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

A = Path(__file__).resolve().parent
O = A / "official"
O.mkdir(exist_ok=True)
base = "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/"
sources = {
    "functions.rst": base + "library/functions.rst",
    "stdtypes.rst": base + "library/stdtypes.rst",
    "exceptions.rst": base + "library/exceptions.rst",
    "simple_stmts.rst": base + "reference/simple_stmts.rst",
    "unicode.rst": base + "howto/unicode.rst",
    "bengio2003.pdf": "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
}
receipts = []
for name, url in sources.items():
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    raw = response.content
    (O / name).write_bytes(raw)
    receipts.append({"path": name, "url": url, "resolved_url": response.url,
                     "accessed_at": datetime.now(timezone.utc).isoformat(), "status": response.status_code,
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "TLS_verification": True})
(A / "source-fetch-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
