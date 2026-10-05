"""Acquire original official texts; preserve provenance without review judgments."""
import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/workspace/tiny-perceptron-vlm')
OUT = ROOT / 'docs/technical-reviews/artifacts/phase4-6_4-independent/sources'
OUT.mkdir(exist_ok=True)
SOURCES = [
    ('sennrich-P16-1162.pdf', 'https://aclanthology.org/P16-1162.pdf'),
    ('sennrich-P16-1162.html', 'https://aclanthology.org/P16-1162/'),
    ('attention-v7-origin.html', 'https://arxiv.org/abs/1706.03762v7'),
    ('torch-v2.11.0-sparse.py', 'https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/modules/sparse.py'),
    ('torch-v2.11.0-linear.py', 'https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/nn/modules/linear.py'),
    ('torch-2.11-tensor-attributes.html', 'https://docs.pytorch.org/docs/2.11/tensor_attributes.html'),
    ('tokenizers-v0.22.2-bpe-trainer.rs', 'https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/src/models/bpe/trainer.rs'),
    ('tokenizers-v0.22.2-byte-level.rs', 'https://raw.githubusercontent.com/huggingface/tokenizers/v0.22.2/tokenizers/src/pre_tokenizers/byte_level.rs'),
]


def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={'User-Agent': 'independent-course-factual-review/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            receipt = {'url': url, 'final_url': response.url, 'status': response.status, 'accessed_on': '2026-10-05', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'content_type': response.headers.get('Content-Type')}
        (OUT / name).write_bytes(raw)
        receipt['path'] = (OUT / name).relative_to(ROOT).as_posix()
        return receipt
    except Exception as error:
        return {'url': url, 'error': repr(error), 'accessed_on': '2026-10-05'}


with ThreadPoolExecutor(max_workers=4) as pool:
    receipts = list(pool.map(fetch, SOURCES))
original = ROOT / 'docs/technical-reviews/artifacts/fact_finish_r_3-transformer.original.pdf'
raw = original.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697'
(OUT / 'attention-v7.pdf').write_bytes(raw)
receipts.append({'path': (OUT / 'attention-v7.pdf').relative_to(ROOT).as_posix(), 'original_locator': original.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'policy': 'Immutable original PDF copied solely from locator index; no other report read. Provenance/version checked separately against original arXiv landing and PDF identifier.'})
for name in ['sennrich-P16-1162', 'attention-v7']:
    if (OUT / (name + '.pdf')).is_file():
        result = subprocess.run(['pdftotext', '-layout', str(OUT / (name + '.pdf')), str(OUT / (name + '.txt'))], capture_output=True, timeout=30)
        receipts.append({'command': ['pdftotext', '-layout', name + '.pdf', name + '.txt'], 'exit_code': result.returncode, 'stderr': result.stderr.decode()})
(OUT / 'acquisition-receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipts, ensure_ascii=False, indent=2))
