"""Fetch only original research / official source files; no model or training assets."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib, json, subprocess, sys, time

BASE = Path(__file__).resolve().parent
dest = BASE / "sources"
dest.mkdir(exist_ok=True)
sources = [
    ("transformer-v7.pdf", "https://arxiv.org/pdf/1706.03762v7"),
    ("torch-v2.9.0-tensor.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor.py"),
    ("torch-v2.9.0-functional.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py"),
    ("torch-v2.9.0-sparse.py", "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/sparse.py"),
]
receipts = []
for name, url in sources:
    started = time.perf_counter()
    with urlopen(Request(url, headers={"User-Agent": "independent-technical-review/4.7"}), timeout=30) as response:
        raw = response.read()
        receipt = {"url": url, "final_url": response.url, "status": response.status}
    (dest / name).write_bytes(raw)
    receipt.update(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), seconds=time.perf_counter()-started)
    receipts.append(receipt)
    print(json.dumps(receipt))
command = ["pdftotext", "-layout", str(dest / "transformer-v7.pdf"), str(dest / "transformer-v7.txt")]
result = subprocess.run(command, capture_output=True, text=True)
receipts.append({"command_argv": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
(BASE / "source-download-receipt.json").write_text(json.dumps({"command": ".venv/bin/python docs/technical-reviews/artifacts/phase4-4_7-independent/acquire_sources.py", "python": sys.version, "accessed_on": "2026-10-05", "receipts": receipts}, indent=2)+"\n")
result.check_returncode()
