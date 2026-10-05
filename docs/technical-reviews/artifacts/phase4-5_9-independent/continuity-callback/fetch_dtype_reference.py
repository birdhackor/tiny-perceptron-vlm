import hashlib
import json
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parent
url = 'https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/docs/source/tensor_attributes.rst'
with urllib.request.urlopen(url, timeout=20) as response:
    raw = response.read()
(A / 'pytorch-v2.4.1-tensor-attributes.rst').write_bytes(raw)
receipt = {'url': url, 'version': 'PyTorch v2.4.1 documentation, predating installed 2.14.1+cpu',
    'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'accessed_on': '2026-10-05',
    'status': 'retrieved', 'scope': 'Only stable dtype definition and float32 bit-width table; current installed float32 bits/size independently checked on CPU.'}
(A / 'fetch-reference-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
