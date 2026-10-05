"""Fetch only authoritative PyTorch source at the installed immutable commit."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from hashlib import sha256
from pathlib import Path
import json
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
COMMIT = "5c4886908584029761b579af026dcfb627c84070"
PATHS = {
    "official-parameter.py": "torch/nn/parameter.py",
    "official-tensor.py": "torch/_tensor.py",
    "official-tensor-docs.py": "torch/_tensor_docs.py",
    "official-torch-docs.py": "torch/_torch_docs.py",
    "official-optimizer.py": "torch/optim/optimizer.py",
    "official-sgd.py": "torch/optim/sgd.py",
    "official-autograd.md": "docs/source/notes/autograd.md",
    "official-amp-examples.md": "docs/source/notes/amp_examples.md",
}

def fetch(item):
    name, source_path = item
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/{source_path}"
    with urlopen(url, timeout=30) as response:
        raw = response.read()
        receipt = {"filename": name, "url": url, "effective_url": response.url,
                   "http_status": response.status, "version": COMMIT,
                   "accessed_at_utc": datetime.now(UTC).isoformat(),
                   "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    (ROOT / name).write_bytes(raw)
    return receipt

with ThreadPoolExecutor(max_workers=4) as pool:
    receipts = list(pool.map(fetch, PATHS.items()))
(ROOT / "source-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
