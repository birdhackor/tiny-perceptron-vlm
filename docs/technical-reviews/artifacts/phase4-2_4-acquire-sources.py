"""Fetch pinned original PyTorch source; no derived reviewer text is used."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/technical-reviews/artifacts"
BASE = "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/"
FILES = {
    "linear": "torch/nn/modules/linear.py",
    "activation": "torch/nn/modules/activation.py",
    "functional": "torch/nn/functional.py",
    "tensor": "torch/_tensor.py",
    "tensor-docs": "torch/_tensor_docs.py",
    "torch-docs": "torch/_torch_docs.py",
    "grad-mode": "torch/autograd/grad_mode.py",
}
receipts = []
for name, upstream_path in FILES.items():
    url = BASE + upstream_path
    request = urllib.request.Request(url, headers={"User-Agent": "independent-course-factual-review/2.4"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        receipt = {
            "url": url,
            "final_url": response.geturl(),
            "http_status": response.status,
            "version": "PyTorch v2.8.0 release tag",
            "accessed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        }
    target = OUT / f"phase4-2_4-pytorch-{name}.py"
    target.write_bytes(raw)
    receipt.update(path=target.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    receipts.append(receipt)
    print(f"{name}: HTTP {receipt['http_status']}, {len(raw)} bytes, SHA256 {receipt['sha256']}")
(OUT / "phase4-2_4-source-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
