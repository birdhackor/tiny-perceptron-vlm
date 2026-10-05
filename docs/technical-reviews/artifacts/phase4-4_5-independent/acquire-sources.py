import hashlib
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-4_5-independent/sources'
OUT.mkdir(parents=True, exist_ok=True)
commit = '5c4886908584029761b579af026dcfb627c84070'
targets = {
    'xiong-2020-icml.pdf': 'https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf',
    'torch-transformer.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/transformer.py',
    'torch-normalization.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/normalization.py',
    'torch-linear.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/nn/modules/linear.py',
    'torch-tensor-docs.py': f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_tensor_docs.py',
    'torch-autograd.rst': f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/docs/source/notes/autograd.rst',
    'transformers-cache-explanation.md': 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/docs/source/en/cache_explanation.md',
}
receipts = []
for name, url in targets.items():
    request = urllib.request.Request(url, headers={'User-Agent': 'independent-factual-review/4.5'})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read()
            final_url = response.url
            status = response.status
        (OUT / name).write_bytes(raw)
        receipts.append({'file': name, 'url': url, 'final_url': final_url, 'http_status': status,
                         'retrieved_at': datetime.now(UTC).isoformat(), 'bytes': len(raw),
                         'sha256': hashlib.sha256(raw).hexdigest()})
        print(name, status, len(raw), hashlib.sha256(raw).hexdigest(), flush=True)
    except Exception as exc:
        receipts.append({'file': name, 'url': url, 'error': repr(exc),
                         'retrieved_at': datetime.now(UTC).isoformat()})
        print(name, repr(exc), flush=True)
shared = ROOT / 'docs/technical-reviews/artifacts/phase4-3_1-independent/sources/vaswani-2017-v7.pdf'
raw = shared.read_bytes()
shutil.copyfile(shared, OUT / 'vaswani-2017-v7.pdf')
receipts.append({'file': 'vaswani-2017-v7.pdf', 'url': 'https://arxiv.org/pdf/1706.03762v7',
                 'reused_original_path': str(shared.relative_to(ROOT)),
                 'read_at': datetime.now(UTC).isoformat(), 'bytes': len(raw),
                 'sha256': hashlib.sha256(raw).hexdigest(),
                 'provenance_receipt': 'docs/technical-reviews/artifacts/phase4-3_1-independent/sources/receipts.json'})
(OUT / 'receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n')
