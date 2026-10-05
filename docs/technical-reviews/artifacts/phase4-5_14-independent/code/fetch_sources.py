"""Fetch only original official documents/source; no data, models or weights."""
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCES = BASE / "sources"
URLS = {
    "smith-clr-v6.pdf": "https://arxiv.org/pdf/1506.01186v6",
    "pytorch-v2.8.0-sgd.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/sgd.py",
    "pytorch-v2.8.0-adamw.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/adamw.py",
    "pytorch-v2.8.0-adam.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/adam.py",
    "pytorch-v2.8.0-optimizer.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/optim/optimizer.py",
    "pytorch-v2.8.0-randomness.rst": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/notes/randomness.rst",
    "pytorch-v2.8.0-loss.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/loss.py",
}
receipt = {"accessed_at": datetime.now(UTC).isoformat(), "python": sys.version, "sources": []}
for name, url in URLS.items():
    request = urllib.request.Request(url, headers={"User-Agent": "Independent educational factual review"})
    with urllib.request.urlopen(request, timeout=25) as response:
        data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("source exceeds bounded source size")
        final_url = response.url
    path = SOURCES / name
    path.write_bytes(data)
    record = {"path": str(path.relative_to(BASE)), "url": url, "final_url": final_url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    receipt["sources"].append(record)
    print(json.dumps(record))
    (BASE / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
subprocess.run(["pdftotext", "-layout", str(SOURCES / "smith-clr-v6.pdf"), str(SOURCES / "smith-clr-v6.txt")], check=True)
