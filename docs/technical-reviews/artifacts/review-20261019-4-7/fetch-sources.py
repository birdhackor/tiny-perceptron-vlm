"""用 curl 讀原始來源，只保存短摘錄、定位與讀取收據。"""

import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

TORCH_COMMIT = "5c4886908584029761b579af026dcfb627c84070"
base = f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/"
specs = [
    {
        "id": "transformer", "url": "https://arxiv.org/html/1706.03762v7",
        "version": "arXiv:1706.03762v7 HTML", "html": True,
        "excerpt": "This masking, combined with fact that the output embeddings are offset by one position",
        "locator": "Model Architecture; Encoder and Decoder Stacks; Attention; Applications of Attention in our Model",
        "terms": ["the decoder then generates", "output embeddings are offset", "queries simultaneously", "including that position"],
    },
    {
        "id": "torch_tensor", "url": base + "torch/_tensor.py", "version": TORCH_COMMIT,
        "excerpt": "This function accumulates gradients in the leaves",
        "locator": "Tensor.backward", "terms": ["def backward("],
    },
    {
        "id": "torch_tensor_docs", "url": base + "torch/_tensor_docs.py", "version": TORCH_COMMIT,
        "excerpt": "Returns the tensor as a (nested) list.",
        "locator": "Tensor.tolist documentation", "terms": ["Returns the tensor as a (nested) list."],
    },
    {
        "id": "torch_loss", "url": base + "torch/nn/functional.py", "version": TORCH_COMMIT,
        "excerpt": "Specifies a target value that is ignored and does not contribute to the input gradient.",
        "locator": "cross_entropy parameters ignore_index and reduction; torch._C._nn.cross_entropy_loss dispatch",
        "terms": ["def cross_entropy(", "return torch._C._nn.cross_entropy_loss("],
    },
    {
        "id": "torch_embedding", "url": base + "torch/nn/modules/sparse.py", "version": TORCH_COMMIT,
        "excerpt": "IntTensor or LongTensor of arbitrary shape containing the indices to extract",
        "locator": "Embedding Shape and forward", "terms": ["IntTensor or LongTensor", "def forward(self, input: Tensor)"],
    },
]
receipts = []
for spec in specs:
    command = ["curl", "-fLsS", "--max-time", "30", spec["url"]]
    raw = subprocess.check_output(command)
    text = raw.decode()
    if spec.get("html"):
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    normalized = re.sub(r"\s+", " ", text)
    assert spec["excerpt"] in normalized
    assert len(spec["excerpt"].split()) <= 25
    for term in spec["terms"]:
        assert term in text or term in normalized, (spec["id"], term)
    receipts.append({
        "id": spec["id"], "command_argv": command, "url": spec["url"], "version": spec["version"],
        "accessed_at": datetime.now(UTC).isoformat(), "retrieved_bytes": len(raw),
        "retrieved_sha256": hashlib.sha256(raw).hexdigest(), "locator": spec["locator"],
        "short_excerpt": spec["excerpt"], "expected_locators_present": spec["terms"],
        "storage_policy": "原文僅在記憶體讀取；不保存第三方全文。",
    })
output = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-4-7/fetch-sources.py", "receipts": receipts}
with (Path(__file__).parent / "original-source-receipts.json").open("x") as stream:
    stream.write(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(receipts, ensure_ascii=False, indent=2))
