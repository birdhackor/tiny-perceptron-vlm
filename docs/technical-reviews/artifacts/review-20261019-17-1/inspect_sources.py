"""用 curl 讀官方原文，只輸出定位附近的小段摘錄與全文雜湊。"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(text)


COMMIT = "5c4886908584029761b579af026dcfb627c84070"
requests = [
    {
        "id": "torch_tensor_docs",
        "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_tensor_docs.py",
        "version": f"PyTorch 2.14.1 / {COMMIT}",
        "needles": [
            ("If the ``self`` Tensor already", 360),
            ("converting floating point value to integer type", 310),
            ("Returns the size in bytes of an individual element.", 160),
            ('r"""\nnumel() -> int', 140),
        ],
    },
    {
        "id": "torch_dtypes",
        "url": "https://docs.pytorch.org/docs/2.14/tensor_attributes.html",
        "version": "PyTorch 2.14 documentation",
        "html": True,
        "needles": [("32-bit floating point", 180), ("16-bit floating point", 200), ("8-bit integer (signed)", 160)],
    },
    {
        "id": "nvidia_quantization",
        "url": "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html",
        "version": "TensorRT 11.4.0 documentation, as declared by the fetched page",
        "html": True,
        "needles": [("technique that reduces model size", 380), ("Quantization Approach", 420)],
    },
    {
        "id": "gemmlowp_quantization",
        "url": "https://raw.githubusercontent.com/google/gemmlowp/master/doc/quantization.md",
        "version": "Google gemmlowp master / actual response bound by SHA-256",
        "needles": [("real_value = scale * (quantized_value - zero_point)", 580)],
    },
    {
        "id": "nvidia_float",
        "url": "https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html",
        "version": "Train With Mixed Precision / live document; version bound by fetched SHA-256",
        "html": True,
        "needles": [("5 exponent bits", 450)],
    },
    {
        "id": "torch_memory",
        "url": "https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html",
        "version": "PyTorch official tutorial / live document; version bound by fetched SHA-256",
        "html": True,
        "needles": [("Activations that are saved", 450)],
    },
    {
        "id": "hf_cache",
        "url": "https://huggingface.co/docs/transformers/en/cache_explanation",
        "version": "Transformers official cache explanation / actual response bound by SHA-256",
        "html": True,
        "needles": [("key-value", 550), ("memory", 300)],
    },
    {
        "id": "torch_linear",
        "url": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/nn/modules/linear.py",
        "version": f"PyTorch 2.14.1 / {COMMIT}",
        "needles": [("Applies an affine linear transformation", 150)],
    },
]


def fetch(item):
    command = ["curl", "-fLsS", "--max-time", "30", item["url"]]
    completed = subprocess.run(command, capture_output=True, check=True)
    raw = completed.stdout
    content = raw.decode("utf-8")
    if item.get("html"):
        parser = TextParser()
        parser.feed(content)
        content = " ".join(" ".join(parser.parts).split())
    excerpts = []
    for needle, length in item["needles"]:
        index = content.find(needle)
        excerpts.append({"locator": needle, "found": index >= 0, "excerpt": content[max(0, index - 40):index + length] if index >= 0 else None})
    return {
        "id": item["id"],
        "url": item["url"],
        "version": item["version"],
        "accessed_on": datetime.now(timezone.utc).date().isoformat(),
        "command": command,
        "exit_code": completed.returncode,
        "full_response_sha256": hashlib.sha256(raw).hexdigest(),
        "full_response_bytes": len(raw),
        "excerpts": excerpts,
    }


with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(fetch, requests))
print(json.dumps({"sources": results}, ensure_ascii=False, indent=2))
