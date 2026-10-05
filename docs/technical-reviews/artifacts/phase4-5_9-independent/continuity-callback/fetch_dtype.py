import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parent
items = {
    'pytorch-tensor-attributes.md': 'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/docs/source/tensor_attributes.md',
    'python-stdtypes.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst',
}
def fetch(item):
    name, url = item
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
        (A / name).write_bytes(raw)
        return {'path': name, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'accessed_on': '2026-10-05', 'status': 'retrieved'}
    except Exception as error:
        return {'path': name, 'url': url, 'status': 'failed', 'error': repr(error)}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    result = list(pool.map(fetch, items.items()))
(A / 'fetch-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
