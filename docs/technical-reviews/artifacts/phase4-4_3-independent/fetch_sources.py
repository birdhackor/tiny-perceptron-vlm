"""Retrieve original authoritative material; no report or reviewer notes are inputs."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "5c4886908584029761b579af026dcfb627c84070"
URLS = {
    "ba-2016-layer-normalization-v1.pdf": "https://arxiv.org/pdf/1607.06450v1",
    "torch-normalization-installed-commit.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/nn/modules/normalization.py",
    "torch-docs-installed-commit.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_torch_docs.py",
    "torch-functional-installed-commit.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/nn/functional.py",
}

def fetch(item):
    name, url = item
    try:
        with urllib.request.urlopen(url, timeout=40) as response:
            data = response.read()
            receipt = {"path": str(ROOT / name), "url": url, "final_url": response.url,
                       "status": response.status, "accessed_on": "2026-10-05",
                       "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        (ROOT / name).write_bytes(data)
        return receipt
    except Exception as error:
        return {"url": url, "path": name, "error": repr(error)}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    receipts = list(pool.map(fetch, URLS.items()))
(ROOT / "source-retrieval.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
assert all("error" not in receipt for receipt in receipts)
