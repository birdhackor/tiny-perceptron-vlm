"""Fetch immutable official originals for an independent review; no model or dataset downloads."""
import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

DEST = Path(__file__).resolve().parent / "sources"
SOURCES = {
    "torch-functional-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py",
    "torch-docs-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_torch_docs.py",
    "tensor-docs-v2.9.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor_docs.py",
    "hf-logits-process-v4.57.1.py": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/logits_process.py",
    "hf-generation-strategies-v4.57.1.md": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/generation_strategies.md",
    "holtzman-1904.09751v2.pdf": "https://arxiv.org/pdf/1904.09751v2",
}

receipts = []
for name, url in SOURCES.items():
    with urllib.request.urlopen(url, timeout=35) as response:
        raw = response.read()
        receipt = {
            "file": name,
            "url": url,
            "final_url": response.url,
            "accessed_at": datetime.now(UTC).isoformat(),
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (DEST / name).write_bytes(raw)
    receipts.append(receipt)
    print(name, receipt["http_status"], receipt["bytes"], receipt["sha256"])
(DEST / "acquisition.json").write_text(json.dumps(receipts, indent=2) + "\n")
