"""Fetch original official sources for independent 1.8 review; no review verdicts."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCES = {
    "torch-functional-installed-commit.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py",
    "torch-loss-installed-commit.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/modules/loss.py",
    "torch-lossnll-installed-commit.cpp": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp",
    "torch-tensor-docs-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor_docs.py",
    "torch-loss-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/loss.py",
    "torch-functional-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py",
    "torch-docs-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_torch_docs.py",
    "torch-lossnll-v2.9.0.cpp": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/LossNLL.cpp",
    "scipy-entropy-v1.16.2.py": "https://raw.githubusercontent.com/scipy/scipy/v1.16.2/scipy/stats/_entropy.py",
    "hf-perplexity-v4.57.1.md": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/perplexity.md",
}


def fetch(pair):
    name, url = pair
    result = {"path": name, "url": url, "accessed_on": "2026-10-05"}
    try:
        with urllib.request.urlopen(url, timeout=25) as response:
            raw = response.read()
            result.update(status=response.status, resolved_url=response.url)
        (HERE / name).write_bytes(raw)
        result.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    receipts = list(pool.map(fetch, SOURCES.items()))
(HERE / "source-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
for receipt in receipts:
    print(json.dumps(receipt))
