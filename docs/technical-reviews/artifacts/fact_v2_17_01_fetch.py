"""Fetch original, version-bound documentation for the independent 17.1 review."""

import hashlib
import json
import platform
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COMMIT = torch.version.git_version
BASE = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}"
URLS = {
    "tensor_docs": f"{BASE}/torch/_tensor_docs.py",
    "torch_docs": f"{BASE}/torch/_torch_docs.py",
    "dtypes": f"{BASE}/docs/source/tensor_attributes.md",
    "quantization": f"{BASE}/docs/source/quantization.md",
    "serialization": f"{BASE}/docs/source/notes/serialization.md",
    "autograd": f"{BASE}/docs/source/notes/autograd.md",
    "adam": f"{BASE}/torch/optim/adam.py",
    "cache": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/cache_explanation.md",
    "jacob_paper": "https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf",
    "private_fp32": (
        "https://huggingface.co/birdhackor/tiny-perceptron-checkpoints/resolve/"
        "d62f91233f17070c9c50e8d339a852f4ebabfd5e/"
        "course/course-v1/quantization/gha-37059971741-1/fp32.pt"
    ),
}


def fetch(item):
    key, url = item
    request = urllib.request.Request(url, headers={"User-Agent": "17.1-independent-review"})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read()
            status = response.status
        suffix = ".pdf" if key == "jacob_paper" else ".pt" if key == "private_fp32" else ".txt"
        path = OUT / f"fact_v2_17_01_{key}{suffix}"
        path.write_bytes(data)
        return {
            "id": key,
            "url": url,
            "status": status,
            "bytes": len(data),
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    except urllib.error.HTTPError as error:
        return {"id": key, "url": url, "status": error.code, "reason": str(error.reason)}
    except (OSError, ValueError) as error:
        return {"id": key, "url": url, "error": str(error)}


def main():
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(fetch, URLS.items()))
    output = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_17_01_fetch.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "pytorch_git_version": COMMIT,
        "accessed_on": "2026-10-04",
        "results": results,
    }
    path = OUT / "fact_v2_17_01_fetch.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
