"""Retrieve exact original official pages; no summaries or local review reports."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "authorities"
DEST.mkdir(exist_ok=True)
SOURCES = {
    "tensor-attributes": "https://docs.pytorch.org/docs/2.8/tensor_attributes.html",
    "tensor-to": "https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.to.html",
    "tensor-numel": "https://docs.pytorch.org/docs/2.8/generated/torch.numel.html",
    "tensor-element-size": "https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.element_size.html",
    "quantization": "https://docs.pytorch.org/docs/2.8/quantization.html",
    "serialization": "https://docs.pytorch.org/docs/2.8/notes/serialization.html",
    "memory-anatomy": "https://huggingface.co/docs/transformers/v4.57.1/model_memory_anatomy",
    "cache-explanation": "https://huggingface.co/docs/transformers/v4.57.1/cache_explanation",
}

def fetch(item):
    key, url = item
    try:
        with urlopen(Request(url, headers={"User-Agent": "Independent factual verification"}), timeout=30) as response:
            raw = response.read()
            resolved = response.url
        target = DEST / (key + ".html")
        target.write_bytes(raw)
        return {"id": key, "url": url, "resolved_url": resolved, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "path": str(target.relative_to(ROOT)), "retrieved_at": datetime.now(timezone.utc).isoformat(), "status": "retrieved"}
    except Exception as error:
        return {"id": key, "url": url, "status": "failed", "error": str(error)}

results = list(ThreadPoolExecutor(max_workers=4).map(fetch, SOURCES.items()))
(DEST / "retrieval.json").write_text(json.dumps(results, indent=2) + "\n")
for result in results:
    print(json.dumps(result))
