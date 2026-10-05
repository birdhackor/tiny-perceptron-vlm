"""Bounded HTTPS retrieval of authoritative sources; no model/data downloads."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from pathlib import Path

BASE = Path(__file__).resolve().parent
ITEMS = {
    "torch-loss.html": "https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html",
    "torch-embedding.html": "https://docs.pytorch.org/docs/2.11/generated/torch.nn.Embedding.html",
    "torch-autograd.html": "https://docs.pytorch.org/docs/2.11/notes/autograd.html",
    "torch-item.html": "https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.item.html",
    "torch-sum.html": "https://docs.pytorch.org/docs/2.11/generated/torch.sum.html",
    "python-expressions.html": "https://docs.python.org/3.13/reference/expressions.html",
    "python-builtins.html": "https://docs.python.org/3.13/library/functions.html",
    "python-stdtypes.html": "https://docs.python.org/3.13/library/stdtypes.html",
    "trl-sft.html": "https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer",
    "transformers-gpt2-v4.57.1.py": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/gpt2/modeling_gpt2.py",
}

def fetch(item):
    name, url = item
    record = {"file": name, "requested_url": url, "started_utc": datetime.now(UTC).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Independent factual review 7.3"})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise RuntimeError("source exceeds bound")
            record.update(final_url=response.url, status=response.status, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (BASE / name).write_bytes(raw)
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
    return record

with ThreadPoolExecutor(max_workers=5) as executor:
    records = list(executor.map(fetch, ITEMS.items()))
(BASE / "retrieval.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
for record in records:
    print(json.dumps(record, ensure_ascii=False))
