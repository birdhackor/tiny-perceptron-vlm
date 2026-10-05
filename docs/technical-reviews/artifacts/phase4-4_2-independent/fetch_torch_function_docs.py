import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parent
url = "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py"
with urlopen(url, timeout=25) as response:
    raw = response.read()
    receipt = {"requested_url": url, "final_url": response.url, "http_status": response.status,
               "accessed_at": datetime.now(timezone.utc).isoformat(), "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest(), "file": "torch-_torch_docs.py"}
(root / "torch-_torch_docs.py").write_bytes(raw)
(root / "torch-function-docs.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
