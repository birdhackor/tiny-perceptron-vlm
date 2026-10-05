"""Fetch only original authoritative sources; preserve URL/hash/failures."""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = "5c4886908584029761b579af026dcfb627c84070"
DEST = Path("/tmp/phase4-14_4-independent-authorities")
DEST.mkdir(exist_ok=True)
URLS = {
    "gelu-v5.pdf": "https://arxiv.org/pdf/1606.08415v5",
    "attention-v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "functional.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/nn/functional.py",
    "torch_docs.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_torch_docs.py",
    "tensor_docs.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_tensor_docs.py",
    "Activation.cpp": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/aten/src/ATen/native/cpu/Activation.cpp",
    "common_pitfalls.rst": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/common_pitfalls.rst",
}

def fetch(item):
    name, url = item
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
            final_url = response.url
        (DEST / name).write_bytes(raw)
        return {"name": name, "url": url, "final_url": final_url,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "status": "downloaded", "accessed_on": "2026-10-05"}
    except Exception as error:
        return {"name": name, "url": url, "status": "failed",
                "error": f"{type(error).__name__}: {error}"}

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, URLS.items()))
    print(json.dumps(records, ensure_ascii=False, indent=2))
