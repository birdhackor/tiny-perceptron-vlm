"""Acquire official pinned source bytes; no model or dataset download."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "sources"
ENTRIES = {
    "pytorch-2.9.0-torch-docs.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_torch_docs.py",
    "pytorch-2.9.0-tensor-docs.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor_docs.py",
    "pytorch-2.9.0-tensor-view.rst": "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/tensor_view.rst",
    "transformers-4.56.2-modeling-outputs.py": "https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/modeling_outputs.py",
    "transformers-4.56.2-generation-utils.py": "https://raw.githubusercontent.com/huggingface/transformers/v4.56.2/src/transformers/generation/utils.py",
}
receipts = []
for name, url in ENTRIES.items():
    request = urllib.request.Request(url, headers={"User-Agent": "section-7.8-independent-factual-review"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        receipt = {
            "name": name, "url": url, "resolved_url": response.url,
            "http_status": response.status, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "accessed_on": "2026-10-05", "tls_verification": "Python default HTTPS context; certificate verification enabled",
        }
    (DEST / name).write_bytes(raw)
    receipts.append(receipt)
    print(json.dumps(receipt))
(DEST / "download-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
