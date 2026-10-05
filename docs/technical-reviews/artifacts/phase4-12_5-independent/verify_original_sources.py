"""Recheck immutable official bytes; no model or dataset acquisition."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

out = Path(__file__).resolve().parent
base = "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/"
results = []
for name, remote in [("pytorch-installed-commit-functional.py", "functional.py"),
                     ("pytorch-installed-commit-torch-docs.py", "_torch_docs.py")]:
    url = base + remote
    with urllib.request.urlopen(url, timeout=15) as response:
        raw = response.read()
        record = {"url":url, "final_url":response.url, "status":response.status,
                  "path": "sources/" + name, "sha256":hashlib.sha256(raw).hexdigest(),
                  "python":sys.version, "version":"PyTorch git commit 5c4886908584029761b579af026dcfb627c84070"}
    assert raw == (out / "sources" / name).read_bytes()
    record["equals_previously_fetched_snapshot"] = True
    results.append(record)
print(json.dumps(results, ensure_ascii=False, indent=2))
