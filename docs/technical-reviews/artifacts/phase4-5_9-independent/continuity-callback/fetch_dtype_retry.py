import base64
import hashlib
import json
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parent
commit = '5c4886908584029761b579af026dcfb627c84070'
url = f'https://api.github.com/repos/pytorch/pytorch/contents/docs/source/tensor_attributes.md?ref={commit}'
request = urllib.request.Request(url, headers={'User-Agent': 'independent-5.9-source-inspection'})
with urllib.request.urlopen(request, timeout=20) as response:
    value = json.load(response)
raw = base64.b64decode(value['content'])
(A / 'pytorch-tensor-attributes.md').write_bytes(raw)
receipt = {'url': url, 'download_url': value['download_url'], 'git_blob_sha': value['sha'],
    'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'accessed_on': '2026-10-05',
    'status': 'retrieved', 'reason': 'Raw source first returned HTTP 503; use GitHub official contents API at the identical fixed commit.'}
(A / 'fetch-retry-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
