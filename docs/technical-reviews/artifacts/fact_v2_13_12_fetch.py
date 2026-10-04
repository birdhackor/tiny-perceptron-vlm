"""Retrieve small official source snapshots at pinned commits without environment changes."""

import hashlib
import json
import platform
from pathlib import Path
from urllib.request import urlopen

import torch

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_13_12_"
TORCH_COMMIT = str(torch.version.git_version)
OPENAI_COMMIT = "cbfd210bb8b08f6bc5c26878c10984b90f516c66"
requests = [
    ("torch_tensor_official.txt", f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/torch/_tensor.py",
     Path(torch.__file__).parent / "_tensor.py"),
    ("torch_docs_official.txt", f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/torch/_torch_docs.py",
     Path(torch.__file__).parent / "_torch_docs.py"),
    ("torch_functional_official.txt", f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH_COMMIT}/torch/nn/functional.py",
     Path(torch.__file__).parent / "nn/functional.py"),
    ("openai_official.txt", f"https://raw.githubusercontent.com/openai/lm-human-preferences/{OPENAI_COMMIT}/lm_human_preferences/train_policy.py",
     ROOT / "outputs/posttrain-design/ppo-reference/openai-lm-human-preferences-core.py"),
]
results = []
for name, url, comparison in requests:
    with urlopen(url, timeout=30) as response:
        payload = response.read()
        status = response.status
    target = ART / (PREFIX + name)
    target.write_bytes(payload)
    assert payload == comparison.read_bytes(), f"Original snapshot differs from pinned official source: {name}"
    results.append({"url": url, "status": status, "bytes": len(payload),
                    "path": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest(),
                    "comparison": str(comparison), "byte_identical": True})
result = {
    "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_12_fetch.py",
    "result": "Four pinned official GitHub sources retrieved with HTTPS verification and are byte-identical to the inspected installed/candidate originals.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "requests": results,
}
(ART / (PREFIX + "source_fetch.json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
