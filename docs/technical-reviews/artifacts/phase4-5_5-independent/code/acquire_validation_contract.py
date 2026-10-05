"""Save the official validation-selection guidance and upstream source license."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

BASE = Path(__file__).resolve().parents[1]
items = [("sklearn-cross-validation-1.7.2.rst", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.7.2/doc/modules/cross_validation.rst"),
         ("pytorch-LICENSE", "https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/LICENSE")]
records = []
for name, url in items:
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
        record = {"url": url, "final_url": response.geturl(), "status": response.status,
                  "path": "sources/" + name, "accessed_at": datetime.now(timezone.utc).isoformat(),
                  "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "tls_verification": "default verified HTTPS"}
    (BASE / record["path"]).write_bytes(data)
    records.append(record)
(BASE / "results/validation-source-acquisition.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
