"""Official source docstrings matching the installed PyTorch commit."""
import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

BASE = Path(__file__).resolve().parent
commit = "5c4886908584029761b579af026dcfb627c84070"
receipts = []
for name, upstream in [("loss-installed.py", "torch/nn/modules/loss.py"), ("functional-installed.py", "torch/nn/functional.py"), ("torch-docs-installed.py", "torch/_torch_docs.py"), ("rnn-installed.py", "torch/nn/utils/rnn.py"), ("tensor-docs-installed.py", "torch/_tensor_docs.py")]:
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{upstream}"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
        receipt = {"url": url, "response_url": response.url, "status": response.status,
                   "accessed_on": datetime.now(UTC).date().isoformat(),
                   "path": "sources/" + name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    (BASE / "sources" / name).write_bytes(raw)
    receipts.append(receipt)
(BASE / "exact-source-acquisition.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
