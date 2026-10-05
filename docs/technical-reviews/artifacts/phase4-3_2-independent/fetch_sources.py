"""Retrieve version-pinned official PyTorch source for this independent review."""
from pathlib import Path
from datetime import datetime, UTC
from urllib.request import urlopen, Request
import hashlib
import json

target = Path(__file__).resolve().parent / "sources"
commit = "5c4886908584029761b579af026dcfb627c84070"
entries = []
for filename, path in [
    ("torch-docs-installed-commit.py", "torch/_torch_docs.py"),
    ("tensor-docs-installed-commit.py", "torch/_tensor_docs.py"),
    ("functional-installed-commit.py", "torch/nn/functional.py"),
]:
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{path}"
    with urlopen(Request(url, headers={"User-Agent": "independent-technical-review/3.2"}), timeout=30) as response:
        raw = response.read()
        entry = {
            "file": filename, "url": url, "response_url": response.url,
            "status": response.status, "version": commit,
            "accessed_on": datetime.now(UTC).date().isoformat(),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (target / filename).write_bytes(raw)
    entries.append(entry)
    print(json.dumps(entry))

paper = target / "attention-v7.pdf"
assert hashlib.sha256(paper.read_bytes()).hexdigest() == "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
entries.append({
    "file": paper.name,
    "url": "https://arxiv.org/pdf/1706.03762v7",
    "version": "arXiv:1706.03762v7, 2023-08-02",
    "accessed_on": datetime.now(UTC).date().isoformat(),
    "retrieval": "Reused immutable original PDF; same-day source-retrieval.json from phase4-1_3-factual reports verified HTTPS 200. Independently SHA-256 verified and personally read section 3.2 and 3.2.1.",
    "receipt_path": "docs/technical-reviews/artifacts/phase4-1_3-factual/source-retrieval.json",
    "sha256": hashlib.sha256(paper.read_bytes()).hexdigest(), "bytes": paper.stat().st_size,
})
(target / "receipts.json").write_text(json.dumps(entries, indent=2) + "\n")
