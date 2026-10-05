"""Acquire immutable primary sources only; no prior review conclusions."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = "5c4886908584029761b579af026dcfb627c84070"
SOURCES = {
    "vaswani-v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "torch-linear.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/nn/modules/linear.py",
    "torch-tensor-docs.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_tensor_docs.py",
    "torch-docs.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_torch_docs.py",
    "torch-random.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/random.py",
    "transformers-cache-explanation.md": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/cache_explanation.md",
}

def acquire(item):
    name, url = item
    with urllib.request.urlopen(url, timeout=40) as response:
        raw = response.read()
        final_url = response.url
    path = ROOT / "sources" / name
    path.write_bytes(raw)
    return {"name": name, "url": url, "final_url": final_url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "accessed_on": "2026-10-05"}

with ThreadPoolExecutor(max_workers=6) as pool:
    receipts = list(pool.map(acquire, SOURCES.items()))
(ROOT / "source-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
