import hashlib
import json
from pathlib import Path
import urllib.request

root = Path('outputs/natural-v4/factual-research/T.6')
root.mkdir(parents=True, exist_ok=True)
urls = {
    'pytorch-layernorm': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html',
    'pytorch-stft': 'https://docs.pytorch.org/docs/stable/generated/torch.stft.html',
    'soundfile': 'https://python-soundfile.readthedocs.io/en/latest/',
    'libsndfile': 'https://libsndfile.github.io/libsndfile/api.html',
    'fsdd': 'https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/26eb9aaf76e81b692f806f9140c2d2777410d7a1/README.md',
    'fashion': 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/README.md',
    'llava': 'https://arxiv.org/html/2304.08485v2',
}
receipts = []
for name, url in urls.items():
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
            final = response.url
        path = root / (name + '.original')
        path.write_bytes(raw)
        receipts.append({'id': name, 'url': url, 'final_url': final, 'accessed_on': '2026-10-04', 'bytes': len(raw), 'retrieval_sha256': hashlib.sha256(raw).hexdigest(), 'ignored_raw_path': str(path)})
    except Exception as error:
        receipts.append({'id': name, 'url': url, 'error': str(error)})
print(json.dumps(receipts, indent=2))
