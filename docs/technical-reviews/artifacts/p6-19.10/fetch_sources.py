"""Fetch primary sources only; never run training or evaluation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "moe-paper.pdf": "https://arxiv.org/pdf/1701.06538v1",
    "distillation-paper.pdf": "https://arxiv.org/pdf/1503.02531v1",
    "quantization-paper.pdf": "https://arxiv.org/pdf/1712.05877v1",
    "safetensors-spec.html": "https://huggingface.co/docs/safetensors/index",
    "safetensors-sharing.html": "https://huggingface.co/docs/safetensors/torch_shared_tensors",
    "torch-tensors.html": "https://docs.pytorch.org/docs/2.14/tensors.html",
    "torch-dtypes.html": "https://docs.pytorch.org/docs/2.14/tensor_attributes.html",
    "torch-element-size.html": "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.element_size.html",
    "sequence-distillation-paper.pdf": "https://arxiv.org/pdf/1606.07947v1",
    "safetensors-readme.md": "https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/README.md",
}


def fetch(item):
    name, url = item
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    out = ROOT / "external" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(response.content)
    return {
        "path": str(out.relative_to(ROOT)), "url": url,
        "resolved_url": response.url, "status": response.status_code,
        "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(),
        "accessed_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(fetch, SOURCES.items()))
    (ROOT / "external-fetch.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
