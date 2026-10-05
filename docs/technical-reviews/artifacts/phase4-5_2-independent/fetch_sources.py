"""Fetch only immutable official PyTorch text sources for section 5.2 review."""
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMMIT = "5c4886908584029761b579af026dcfb627c84070"
FILES = {
    "loss": "torch/nn/modules/loss.py",
    "functional": "torch/nn/functional.py",
    "tensor": "torch/_tensor.py",
    "autograd": "docs/source/notes/autograd.rst",
    "sgd": "torch/optim/sgd.py",
    "dataloader": "torch/utils/data/dataloader.py",
}
rows = []
for name, upstream in FILES.items():
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/{upstream}"
    row = {"id": name, "url": url, "version": f"PyTorch commit {COMMIT} (installed 2.14.1+cpu)", "accessed_on": "2026-10-05"}
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
            row.update(status=response.status, final_url=response.url)
        target = BASE / "sources" / f"pytorch-{name}.txt"
        target.write_bytes(raw)
        row.update(path=str(target.relative_to(BASE)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    except (urllib.error.URLError, TimeoutError) as error:
        row.update(error=f"{type(error).__name__}: {error}")
    rows.append(row)
(BASE / "sources" / "fetch-receipt.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(rows, ensure_ascii=False, indent=2))
