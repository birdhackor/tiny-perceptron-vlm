from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from pathlib import Path
import hashlib
import json
import urllib.request

DEST = Path(__file__).resolve().parent / 'sources'
DEST.mkdir(exist_ok=True)
SOURCES = {
    'pytorch-v2.9.0-sparse.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/modules/sparse.py',
    'pytorch-v2.9.0-functional.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/nn/functional.py',
    'pytorch-v2.9.0-grad_mode.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/autograd/grad_mode.py',
    'pytorch-v2.9.0-Embedding.cpp': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/aten/src/ATen/native/Embedding.cpp',
    'pytorch-v2.9.0-_torch_docs.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_torch_docs.py',
    'pytorch-v2.9.0-_tensor_docs.py': 'https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor_docs.py',
    'bengio-2003-jmlr.pdf': 'https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf',
    'vaswani-2017-v7.pdf': 'https://arxiv.org/pdf/1706.03762v7',
}

def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={'User-Agent': 'Independent factual review of embedding teaching example'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
        metadata = {
            'name': name, 'url': url, 'final_url': response.url,
            'status': response.status, 'accessed_utc': datetime.now(UTC).isoformat(),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'content_type': response.headers.get('Content-Type'),
        }
    (DEST / name).write_bytes(data)
    return metadata

with ThreadPoolExecutor(max_workers=6) as pool:
    receipts = list(pool.map(fetch, [(name, url) for name, url in SOURCES.items() if not (DEST / name).exists()]))
existing_receipts = json.loads((DEST / 'receipts.json').read_text()) if (DEST / 'receipts.json').exists() else []
receipts = existing_receipts + receipts
(DEST / 'receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipts, ensure_ascii=False, indent=2))
