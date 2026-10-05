"""Fetch immutable official source files only; no model/data downloads."""
import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

ART = Path(__file__).resolve().parent
DEST = ART / "sources"
DEST.mkdir(exist_ok=True)


def get(url, name):
    request = urllib.request.Request(url, headers={"User-Agent": "section-4.6-factual-audit"})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read()
        info = {"url": url, "final_url": response.url, "http_status": response.status,
                "accessed_on": str(date.today()), "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    (DEST / name).write_bytes(raw)
    return info


receipts = []
commit = "5c4886908584029761b579af026dcfb627c84070"
for path, name in [
    ("torch/nn/modules/linear.py", "pytorch-linear.py"),
    ("torch/nn/modules/sparse.py", "pytorch-sparse.py"),
    ("torch/nn/modules/loss.py", "pytorch-loss.py"),
    ("torch/nn/functional.py", "pytorch-functional.py"),
    ("torch/_torch_docs.py", "pytorch-torch-docs.py"),
]:
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/{path}"
    try:
        item = get(url, name)
        item["version"] = commit
        receipts.append(item)
        print(name, item["bytes"], item["sha256"])
    except Exception as error:
        receipts.append({"url": url, "path": name, "error": repr(error)})
        print(name, "ERROR", repr(error))

try:
    meta = get("https://api.github.com/repos/openai/gpt-2/commits/master", "gpt2-master-commit.json")
    receipts.append(meta)
    commit = json.loads((DEST / "gpt2-master-commit.json").read_text())["sha"]
    for path, name in [("src/model.py", "gpt2-model.py"), ("src/sample.py", "gpt2-sample.py")]:
        item = get(f"https://raw.githubusercontent.com/openai/gpt-2/{commit}/{path}", name)
        item["version"] = commit
        receipts.append(item)
        print(name, item["bytes"], item["sha256"])
except Exception as error:
    receipts.append({"source": "openai/gpt-2", "error": repr(error)})
    print("gpt2 ERROR", repr(error))
try:
    item = get("https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/gpt2/modeling_gpt2.py", "transformers-gpt2-modeling.py")
    item["version"] = "transformers v4.57.1"
    receipts.append(item)
    print(item["path"], item["bytes"], item["sha256"])
except Exception as error:
    receipts.append({"source": "huggingface/transformers v4.57.1", "error": repr(error)})
    print("transformers GPT2 ERROR", repr(error))
(DEST / "fetch-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
