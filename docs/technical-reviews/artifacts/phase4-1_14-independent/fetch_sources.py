"""Read-only HTTPS retrieval of original upstream materials for 1.14."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parent / "sources"
OUT.mkdir(exist_ok=True)
URLS = {
    "torch-docs-v2.11.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/_torch_docs.py",
    "torch-tensor-docs-v2.11.0.py": "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/_tensor_docs.py",
    "torch-generator-v2.11.0.rst": "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/docs/source/generated/torch.Generator.rst",
    "torch-randomness-v2.11.0.rst": "https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/docs/source/notes/randomness.rst",
    "python-list-v3.13.5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/tutorial/datastructures.rst",
    "python-types-v3.13.5.rst": "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst",
    "transformers-generation-v4.57.1.py": "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/utils.py",
    "bengio03a.pdf": "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
}
receipts = []
for filename, url in URLS.items():
    entry = {"path": "sources/" + filename, "url": url, "accessed_on": datetime.date.today().isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "phase4 factual review/1.14"})
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read()
            entry.update(status=response.status, final_url=response.url, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (OUT / filename).write_bytes(raw)
    except Exception as exc:
        entry.update(error=type(exc).__name__ + ": " + str(exc))
    receipts.append(entry)
    print(json.dumps(entry))
(OUT / "receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
