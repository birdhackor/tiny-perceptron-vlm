"""以 curl 讀取原始來源，只保存收據與短摘錄。"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


pages = [
    (
        "sdpa",
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html",
        "PyTorch 2.14 文件；本機執行版本為 2.14.1",
        ["Computes scaled dot product", "This function always applies", "There are currently", "query (", "Parameters:"],
        "True indicates values to participate in attention.",
    ),
    (
        "allclose",
        "https://docs.pytorch.org/docs/2.14/generated/torch.allclose.html",
        "PyTorch 2.14 文件",
        ["This function checks", "Parameters:"],
        "absolute tolerance.",
    ),
    (
        "dtype",
        "https://docs.pytorch.org/docs/2.14/tensor_attributes.html",
        "PyTorch 2.14 文件",
        ["32-bit floating point", "16-bit floating point"],
        "32-bit floating point",
    ),
    (
        "grad",
        "https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html",
        "PyTorch 2.14 文件",
        ["Compute and return", "grad_outputs", "vector-Jacobian"],
        "Compute and return the sum of gradients of outputs with respect to the inputs.",
    ),
    (
        "logits",
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
        "PyTorch 2.14 文件",
        ["unnormalized logits"],
        "unnormalized logits",
    ),
    (
        "eval",
        "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html",
        "PyTorch 2.14 文件，Module.eval",
        ["This has an effect only on certain modules"],
        "This has an effect only on certain modules.",
    ),
    (
        "cuda",
        "https://developer.nvidia.com/cuda-zone",
        "NVIDIA CUDA Zone，讀取當日頁面",
        ["What Is CUDA?"],
        "CUDA is NVIDIA's platform for accelerated computing",
    ),
    (
        "flash",
        "https://arxiv.org/abs/2205.14135",
        "arXiv:2205.14135v2；摘要與版本紀錄",
        ["Abstract:", "Submission history"],
        "IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes",
    ),
]

receipts = []
for name, url, version, needles, excerpt in pages:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "40", url]
    result = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(result.stdout.decode("utf-8"))
    plain = " ".join(" ".join(parser.parts).split())
    if excerpt not in plain:
        raise ValueError(f"來源短摘錄未出現在原文：{name}")
    receipt = {
        "id": name,
        "url": url,
        "version": version,
        "accessed_at": datetime.now(timezone.utc).isoformat(),
        "command_argv": command,
        "returncode": result.returncode,
        "original_response_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "original_response_bytes": len(result.stdout),
        "excerpt": excerpt,
        "located_terms": {},
    }
    for needle in needles:
        at = plain.find(needle)
        receipt["located_terms"][needle] = at >= 0
        if at >= 0:
            print(name, needle, plain[max(0, at - 120):at + (4100 if name == "sdpa" else 1700)])
    receipts.append(receipt)

print("SOURCE_RECEIPTS_JSON=" + json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-8/inspect_external.py",
    "environment": {"python": platform.python_version(), "transport": "curl HTTPS"},
    "sources": receipts,
}, ensure_ascii=False))
