"""Fetch original PyTorch source at the installed git revision; no summaries."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parents[1]
REV = "5c4886908584029761b579af026dcfb627c84070"
PATHS = [
    "docs/source/notes/autograd.md",
    "torch/nn/modules/module.py",
    "torch/nn/modules/linear.py",
    "torch/nn/modules/dropout.py",
    "torch/autograd/grad_mode.py",
    "torch/nn/parameter.py",
    "torch/nn/functional.py",
    "torch/_tensor.py",
    "torch/_torch_docs.py",
    "torch/_tensor_docs.py",
]
out = BASE / "sources"
out.mkdir(parents=True, exist_ok=True)
receipts = []
for path in PATHS:
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{REV}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
        receipt = {
            "url": url,
            "resolved_url": response.url,
            "http_status": response.status,
            "revision": REV,
            "accessed_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "path": "sources/" + path.replace("/", "--"),
        }
    (BASE / receipt["path"]).write_bytes(raw)
    receipts.append(receipt)
    print(json.dumps(receipt, ensure_ascii=False))
(out / "fetch-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
