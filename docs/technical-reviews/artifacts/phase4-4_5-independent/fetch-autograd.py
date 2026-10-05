import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

out = Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-4_5-independent/sources')
url = 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/docs/source/notes/autograd.rst'
with urllib.request.urlopen(url, timeout=25) as response:
    raw = response.read()
    final_url, status = response.url, response.status
(out / 'torch-autograd-v2.9.0.rst').write_bytes(raw)
receipt = {'url': url, 'final_url': final_url, 'status': status, 'retrieved_at': datetime.now(UTC).isoformat(),
           'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
           'reason': 'installed-commit .rst URL returned 404; use official v2.9.0 stable source for unchanged grad-mode contract'}
(out / 'autograd-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
