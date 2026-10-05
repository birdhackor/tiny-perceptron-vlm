"""Acquire only primary documents/source; preserve HTTPS, bytes, hashes and failures."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

BASE = Path(__file__).resolve().parents[1]
SOURCES = [
    ("adamw-1711.05101v3.pdf", "https://arxiv.org/pdf/1711.05101v3"),
    ("pytorch-adamw-installed-commit.py", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/optim/adamw.py"),
    ("pytorch-adam-v2.5.1.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/optim/adam.py"),
    ("pytorch-adamw-v2.5.1.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/optim/adamw.py"),
    ("pytorch-optimizer-v2.5.1.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/optim/optimizer.py"),
    ("pytorch-parameter-v2.5.1.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/nn/parameter.py"),
    ("pytorch-torch-docs-v2.5.1.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/_torch_docs.py"),
    ("bert-master-commit.json", "https://api.github.com/repos/google-research/bert/commits/master"),
]

def acquire(item):
    name, url = item
    record = {"path": "sources/" + name, "requested_url": url,
              "accessed_at": datetime.now(timezone.utc).isoformat(),
              "tls_verification": "Python default verified HTTPS; no bypass"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Independent technical factual review"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            record.update(final_url=response.geturl(), status=response.status,
                          content_type=response.headers.get("Content-Type"), bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest())
        (BASE / record["path"]).write_bytes(data)
    except Exception as error:
        record["error"] = type(error).__name__ + ": " + str(error)
    return record

with ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(acquire, SOURCES))
commit_record = next(r for r in records if r["path"].endswith("bert-master-commit.json"))
if "error" not in commit_record:
    commit = json.loads((BASE / commit_record["path"]).read_text())["sha"]
    records.append(acquire(("bert-optimization.py", f"https://raw.githubusercontent.com/google-research/bert/{commit}/optimization.py")))
(BASE / "results/source-acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
