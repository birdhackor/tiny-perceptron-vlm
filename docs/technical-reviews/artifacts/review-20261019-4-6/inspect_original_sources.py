"""以 curl 讀取官方原文，只保存支持主張的小段摘錄與來源收據。"""

import ast
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def fetch(url):
    raw = subprocess.run(["curl", "-fsSL", url], check=True, capture_output=True).stdout
    return raw, raw.decode("utf-8")


def class_doc(text, name, limit):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == name)
    return ast.get_docstring(node)[:limit]


base = "https://raw.githubusercontent.com/pytorch/pytorch/v2.14.1/torch/nn/modules/"
requests = [
    ("linear", base + "linear.py", "Linear", 1550),
    ("container", base + "container.py", "ModuleList", 650),
    ("sparse", base + "sparse.py", "Embedding", 2350),
    ("loss", base + "loss.py", "CrossEntropyLoss", 3300),
    ("normalization", base + "normalization.py", "LayerNorm", 1550),
]
receipts = []
for name, url, symbol, limit in requests:
    raw, text = fetch(url)
    receipts.append({
        "id": name,
        "url": url,
        "version": "PyTorch v2.14.1",
        "accessed_on": datetime.now(UTC).date().isoformat(),
        "full_source_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": symbol + ".__doc__",
        "excerpt": class_doc(text, symbol, limit),
    })
url = "https://raw.githubusercontent.com/openai/gpt-2/master/src/model.py"
raw, text = fetch(url)
lines = text.splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith("def model("))
block = next(i for i, line in enumerate(lines) if line.startswith("def block("))
receipts.append({
    "id": "gpt2",
    "url": url,
    "version": "master content identified by full_source_sha256",
    "accessed_on": datetime.now(UTC).date().isoformat(),
    "full_source_sha256": hashlib.sha256(raw).hexdigest(),
    "locator": "model: embedding, position, blocks, ln_f and logits; block: attention and mlp",
    "excerpt": "\n".join(lines[start:]),
    "block_excerpt": "\n".join(lines[block:block + 9]),
})
target = Path(__file__).with_name("original-source-excerpts.json")
with target.open("x", encoding="utf-8") as stream:
    json.dump(receipts, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(receipts, ensure_ascii=False, indent=2))
