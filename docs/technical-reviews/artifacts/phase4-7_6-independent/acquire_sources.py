"""Acquire original official documentation/source only; no models or data."""
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "sources"
OUT.mkdir(exist_ok=True)
requests = [
    ("pytorch-crossentropy-2.14.html", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html"),
    ("pytorch-crossentropy-2.11.html", "https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html"),
    ("pytorch-cat-2.11.html", "https://docs.pytorch.org/docs/2.11/generated/torch.cat.html"),
    ("pytorch-allclose-2.11.html", "https://docs.pytorch.org/docs/2.11/generated/torch.allclose.html"),
    ("pytorch-padsequence-2.11.html", "https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.rnn.pad_sequence.html"),
    ("pytorch-autograd-2.11.html", "https://docs.pytorch.org/docs/2.11/notes/autograd.html"),
    ("LossNLL-installed.cpp", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/LossNLL.cpp"),
    ("LossNLL-v2.11.0.cpp", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/LossNLL.cpp"),
    ("functional-v2.11.0.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/functional.py"),
]
receipts = []
for name, url in requests:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
            receipt = {"url": url, "response_url": response.url, "status": response.status,
                       "accessed_on": datetime.now(UTC).date().isoformat(),
                       "path": str((OUT / name).relative_to(BASE)),
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        (OUT / name).write_bytes(raw)
    except (urllib.error.URLError, TimeoutError) as error:
        receipt = {"url": url, "error": str(error), "accessed_on": datetime.now(UTC).date().isoformat()}
    receipts.append(receipt)
(BASE / "source-acquisition.json").write_text(json.dumps(receipts, indent=2) + "\n")
print(json.dumps(receipts, indent=2))
