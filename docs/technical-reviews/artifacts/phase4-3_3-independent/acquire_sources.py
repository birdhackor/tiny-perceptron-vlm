"""Acquire original source bytes and preserve provenance; no review decisions."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OUT.joinpath("sources").mkdir(exist_ok=True)
receipts = []
shared = ROOT / "docs/technical-reviews/artifacts/phase4-2_1-independent/sources"
name = "vaswani-2017-v7.pdf"
raw = (shared / name).read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
(OUT / "sources" / name).write_bytes(raw)
receipts.append({
    "name": name, "url": "https://arxiv.org/pdf/1706.03762v7",
    "version": "arXiv:1706.03762v7", "accessed_on": "2026-10-05",
    "reused_raw_snapshot": str((shared / name).relative_to(ROOT)),
    "original_receipt": str((shared / "receipts.json").relative_to(ROOT)),
    "original_receipt_sha256": hashlib.sha256((shared / "receipts.json").read_bytes()).hexdigest(),
    "bytes": len(raw), "sha256": digest,
})
commit = "5c4886908584029761b579af026dcfb627c84070"
paths = {
    "linear.py": "torch/nn/modules/linear.py",
    "tensor-docs.py": "torch/_tensor_docs.py",
    "torch-docs.py": "torch/_torch_docs.py",
    "functional.py": "torch/nn/functional.py",
}
for name, path in paths.items():
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
        receipt = {
            "name": name, "url": url, "final_url": response.url,
            "http_status": response.status, "version": commit,
            "accessed_at": datetime.now(UTC).isoformat(),
            "content_type": response.headers.get("Content-Type"),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }
    (OUT / "sources" / name).write_bytes(raw)
    receipts.append(receipt)
    (OUT / "source-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
    print(name, receipt["http_status"], receipt["bytes"], receipt["sha256"])
print("paper", digest)
