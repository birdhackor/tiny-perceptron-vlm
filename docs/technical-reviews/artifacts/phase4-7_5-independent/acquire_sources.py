"""Acquire only original authority documents; no model/data downloads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from pathlib import Path
import hashlib
import json
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = HERE / "sources"
OUT.mkdir(exist_ok=True)
URLS = {
    "attention-paper-v7.pdf": "https://arxiv.org/pdf/1706.03762v7",
    "cross-entropy.html": "https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html",
    "autograd.html": "https://docs.pytorch.org/docs/2.9/notes/autograd.html",
    "retain-grad.html": "https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.retain_grad.html",
    "is-leaf.html": "https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.is_leaf.html",
    "detach.html": "https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.detach.html",
    "embedding.html": "https://docs.pytorch.org/docs/2.9/generated/torch.nn.Embedding.html",
    "norm.html": "https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.norm.html",
    "vector-norm.html": "https://docs.pytorch.org/docs/2.9/generated/torch.linalg.vector_norm.html",
    "sdpa.html": "https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html",
    "installed-functional.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py",
    "torch-norm.html": "https://docs.pytorch.org/docs/2.9/generated/torch.norm.html",
    "backward.html": "https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.backward.html",
    "item.html": "https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.item.html",
    "manual-seed.html": "https://docs.pytorch.org/docs/2.9/generated/torch.manual_seed.html",
    "installed-tensor.py": "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_tensor.py",
}

previous = json.loads((OUT / "download-provenance.json").read_text()) if (OUT / "download-provenance.json").exists() else []
cached = {r["file"]: r for r in previous if r.get("status") == "downloaded"}

def fetch(item):
    name, url = item
    if name in cached and (OUT / name).exists() and hashlib.sha256((OUT / name).read_bytes()).hexdigest() == cached[name]["sha256"]:
        return cached[name]
    record = {"file": name, "requested_url": url, "accessed_at_utc": datetime.now(UTC).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "technical-review-source-inspection/1.0"})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(12 * 1024 * 1024 + 1)
            assert len(raw) <= 12 * 1024 * 1024
            record.update(final_url=response.url, http_status=response.status)
        (OUT / name).write_bytes(raw)
        record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), status="downloaded")
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    return record

with ThreadPoolExecutor(max_workers=11) as pool:
    records = list(pool.map(fetch, URLS.items()))
(OUT / "download-provenance.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(records, ensure_ascii=False, indent=2))
