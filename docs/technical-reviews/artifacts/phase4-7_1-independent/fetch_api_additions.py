"""Fetch API primary-source additions; preserve failures as well as successful reads."""
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent / "sources"
items = [
    ("pytorch-v2.11.0-_tensor_docs.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/_tensor_docs.py"),
    ("pytorch-v2.11.0-IndexingUtils.h", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/IndexingUtils.h"),
    ("pytorch-v2.14.1-functional.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/functional.py"),
]
records = []
for name, url in items:
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
            status = response.status
        (HERE / name).write_bytes(raw)
        records.append({"path": name, "url": url, "http_status": status, "sha256": hashlib.sha256(raw).hexdigest(), "accessed_on": "2026-10-05", "tls_verification": "default validation enabled"})
    except Exception as error:
        records.append({"path": name, "url": url, "error": str(error)})
(HERE / "api-additions-provenance.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
