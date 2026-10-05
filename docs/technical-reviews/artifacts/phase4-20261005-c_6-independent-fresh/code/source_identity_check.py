"""Verify immutable original paper identity without downloading any model or data."""
import hashlib
import json
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
url = "https://arxiv.org/pdf/2501.12948v1"
with urllib.request.urlopen(url, timeout=30) as response:
    raw = response.read()
    final_url = response.url
    status = response.status
saved = (BASE / "sources/deepseek-r1-v1.pdf").read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert raw == saved
assert digest == "52d8ca3ac93e88cef9944e1fd03b0e04aec5954495a8250fb2fadf8fa20a4dad"
text = (BASE / "sources/deepseek-r1-v1.txt").read_text()
assert "arXiv:2501.12948v1 [cs.CL] 22 Jan 2025" in text
assert "2.2.2. Reward Modeling" in text and "Accuracy rewards:" in text and "Format rewards:" in text
print(json.dumps({"url": url, "final_url": final_url, "status": status,
                  "sha256": digest, "bytes": len(raw), "saved_pdf_equals_current_v1": True,
                  "version": "arXiv:2501.12948v1, 22 Jan 2025", "accessed_on": "2026-10-05"}, indent=2))
