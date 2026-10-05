"""Archive authoritative originals with transport receipts; no review conclusions."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
COMMIT = "5c4886908584029761b579af026dcfb627c84070"
SOURCES = {
    "resnet-cvpr2016.pdf": "https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf",
    "identity-mappings-v3.pdf": "https://arxiv.org/pdf/1603.05027v3",
    "torch-_tensor.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_tensor.py",
    "torch-_tensor_docs.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/_tensor_docs.py",
    "torch-autograd.rst": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/docs/source/autograd.rst",
    "torch-autograd-init.py": f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/torch/autograd/__init__.py",
}
receipts = []
for name, url in SOURCES.items():
    try:
        with urlopen(Request(url, headers={"User-Agent": "independent-factual-review/4.2"}), timeout=25) as response:
            raw = response.read()
            record = {"file": name, "requested_url": url, "final_url": response.url, "http_status": response.status,
                      "content_type": response.headers.get("Content-Type"), "bytes": len(raw),
                      "accessed_at": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(raw).hexdigest()}
        (ROOT / name).write_bytes(raw)
        receipts.append(record)
        print(json.dumps(record), flush=True)
    except Exception as error:
        receipts.append({"file": name, "requested_url": url, "error": repr(error)})
        print(name, repr(error), flush=True)
(ROOT / "source-download-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
