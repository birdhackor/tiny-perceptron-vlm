"""Fetch immutable PyTorch original sources; no models or data downloads."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent
REV = "5c4886908584029761b579af026dcfb627c84070"
PATHS = [
    "docs/source/amp.md", "docs/source/notes/autograd.md",
    "docs/source/notes/cuda.md", "docs/source/tensor_attributes.md",
    "docs/source/type_info.md", "torch/amp/grad_scaler.py",
    "torch/amp/autocast_mode.py", "c10/util/Half.h", "c10/util/BFloat16.h",
    "torch/_torch_docs.py", "torch/_tensor_docs.py", "torch/optim/adamw.py",
    "torch/optim/adam.py",
]
manifest = []
for name in PATHS:
    url = f"https://raw.githubusercontent.com/pytorch/pytorch/{REV}/{name}"
    target = BASE / "primary-sources" / name.replace("/", "--")
    target.parent.mkdir(exist_ok=True)
    try:
        with urlopen(url, timeout=20) as response:
            raw = response.read()
        target.write_bytes(raw)
        manifest.append({"url": url, "version": REV, "path": str(target.relative_to(BASE)),
                         "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                         "accessed_on": "2026-10-05"})
        print(name, len(raw), manifest[-1]["sha256"])
    except Exception as error:
        manifest.append({"url": url, "error": str(error), "accessed_on": "2026-10-05"})
        print(name, type(error).__name__, str(error))
(BASE / "primary-source-retrieval.json").write_text(json.dumps(manifest, indent=2)+"\n")
