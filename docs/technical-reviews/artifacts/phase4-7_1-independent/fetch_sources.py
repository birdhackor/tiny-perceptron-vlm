"""Fetch primary sources with default verified HTTPS; preserve bytes and provenance."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
TARGET = HERE / "sources"
TARGET.mkdir(exist_ok=True)
SOURCES = [
    ("transformers-v4.57.1-chat_templating.md", "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/chat_templating.md"),
    ("pytorch-v2.11.0-functional.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/functional.py"),
    ("pytorch-v2.11.0-TensorAdvancedIndexing.cpp", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/TensorAdvancedIndexing.cpp"),
    ("pytorch-v2.11.0-_tensor.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/_tensor.py"),
    ("python-v3.13.5-stdtypes.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst"),
    ("instructgpt-arxiv-2203.02155v1-abs.html", "https://arxiv.org/abs/2203.02155v1"),
    ("instructgpt-arxiv-2203.02155v1.pdf", "https://arxiv.org/pdf/2203.02155v1"),
]

def fetch(item):
    name, url = item
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Independent factual review; source inspection"})
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read()
            status, final_url, content_type = response.status, response.url, response.headers.get("Content-Type")
        (TARGET / name).write_bytes(raw)
        return {"name": name, "url": url, "resolved_url": final_url, "http_status": status, "content_type": content_type, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "accessed_on": "2026-10-05", "tls_verification": "Python urllib default CA/TLS validation enabled"}
    except Exception as e:
        return {"name": name, "url": url, "error_type": type(e).__name__, "error": str(e)}

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(fetch, SOURCES))
(TARGET / "fetch-provenance.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
