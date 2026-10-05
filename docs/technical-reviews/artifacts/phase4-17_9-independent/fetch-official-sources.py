import hashlib
import json
import urllib.request
from pathlib import Path

base = Path(__file__).resolve().parent
sources = {
    'pytorch-v2.8.0-quantization.rst': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/docs/source/quantization.rst',
    'pytorch-v2.8.0-grad_mode.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/autograd/grad_mode.py',
    'pytorch-v2.8.0-module.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/module.py',
    'python-v3.13.5-copy.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/copy.rst',
}
receipts = []
for filename, url in sources.items():
    request = urllib.request.Request(url, headers={'User-Agent': 'independent-factual-review/17.9'})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read()
        receipt = {'url': url, 'final_url': response.url, 'status': response.status,
                   'file': filename, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'accessed_on': '2026-10-05'}
    (base / filename).write_bytes(raw)
    receipts.append(receipt)
    print(json.dumps(receipt))
(base / 'official-source-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
