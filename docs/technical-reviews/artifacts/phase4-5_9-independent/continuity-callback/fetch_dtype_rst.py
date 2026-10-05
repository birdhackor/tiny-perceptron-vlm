import hashlib
import json
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parent
url = 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/docs/source/tensor_attributes.rst'
with urllib.request.urlopen(url, timeout=20) as response:
    raw = response.read()
(A / 'pytorch-tensor-attributes.rst').write_bytes(raw)
receipt = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
    'accessed_on': '2026-10-05', 'status': 'retrieved',
    'reason': 'The guessed .md source and .md contents API returned 503; retrieve the original .rst pathname at the same commit.'}
(A / 'fetch-rst-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
