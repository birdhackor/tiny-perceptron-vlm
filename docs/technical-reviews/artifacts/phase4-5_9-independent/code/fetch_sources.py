import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

A = Path(__file__).resolve().parents[1]
TORCH = '5c4886908584029761b579af026dcfb627c84070'
items = {
    'torch-grad-mode.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/autograd/grad_mode.py',
    'torch-module.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/nn/modules/module.py',
    'torch-adam.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/optim/adam.py',
    'torch-tensor-docs.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/_tensor_docs.py',
    'torch-cuda.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/cuda/__init__.py',
    'torch-autograd.rst': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/docs/source/notes/autograd.rst',
    'torch-benchmark-timer.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/utils/benchmark/utils/timer.py',
    'python-time.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/time.rst',
}

def fetch(item):
    name, url = item
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read()
            final = response.url
        (A / 'sources' / name).write_bytes(raw)
        return {'path': name, 'url': url, 'resolved_url': final, 'bytes': len(raw),
                'sha256': hashlib.sha256(raw).hexdigest(), 'status': 'retrieved',
                'accessed_on': '2026-10-05'}
    except Exception as error:
        return {'path': name, 'url': url, 'status': 'failed', 'error': repr(error)}

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    receipt = list(pool.map(fetch, items.items()))
(A / 'source-fetch-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
