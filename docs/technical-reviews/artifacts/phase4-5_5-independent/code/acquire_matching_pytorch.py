"""Acquire the exact upstream commit named by the installed PyTorch CPU wheel."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

BASE = Path(__file__).resolve().parents[1]
COMMIT = json.loads((BASE / "results/installed-environment.json").read_text())["torch_git_commit"]
MODULES = ["torch/optim/adam.py", "torch/optim/optimizer.py", "torch/nn/parameter.py",
           "torch/_torch_docs.py", "torch/_tensor.py"]

def acquire(module):
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/{module}"
    path = "sources/pytorch-matching-" + module.replace("/", "--")
    record = {"requested_url": url, "path": path, "commit": COMMIT,
              "accessed_at": datetime.now(timezone.utc).isoformat(), "tls_verification": "default verified HTTPS"}
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
            record.update(status=response.status, final_url=response.geturl(), bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest())
        (BASE / path).write_bytes(data)
        if module in ("torch/optim/adam.py", "torch/optim/optimizer.py", "torch/nn/parameter.py"):
            installed = BASE / "sources" / ("installed-" + module.rsplit("/", 1)[-1])
            record["matches_installed_bytes"] = data == installed.read_bytes()
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
    return record

with ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(acquire, MODULES))
(BASE / "results/matching-source-acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
