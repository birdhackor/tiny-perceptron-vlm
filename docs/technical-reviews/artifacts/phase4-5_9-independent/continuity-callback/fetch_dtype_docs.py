import hashlib
import json
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parent
url = 'https://docs.pytorch.org/docs/stable/tensor_attributes.html'
with urllib.request.urlopen(url, timeout=20) as response:
    raw = response.read()
    final = response.url
(A / 'pytorch-tensor-attributes.html').write_bytes(raw)
receipt = {'url': url, 'resolved_url': final, 'sha256': hashlib.sha256(raw).hexdigest(),
    'bytes': len(raw), 'accessed_on': '2026-10-05', 'status': 'retrieved',
    'reason': 'Pinned raw guessed tensor_attributes paths returned 503; consult authoritative rendered API documentation for dtype bit-width table.'}
(A / 'fetch-docs-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
