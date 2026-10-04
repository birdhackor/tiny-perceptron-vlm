import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path('outputs/natural-v4/factual-research/R.4/originals')
root.mkdir(parents=True, exist_ok=True)
urls = {
    'attention.pdf': 'https://arxiv.org/pdf/1706.03762v7',
    'mixtral.pdf': 'https://arxiv.org/pdf/2401.04088v1',
    'distillation.pdf': 'https://arxiv.org/pdf/1503.02531v1',
    'lora.pdf': 'https://arxiv.org/pdf/2106.09685v2',
    'optimization.html': 'https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html',
    'saving.html': 'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html',
    'qwen-config.json': 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/config.json',
    'qwen-card.md': 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/README.md',
    'qwen-license.txt': 'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/LICENSE',
    'whisper-card.md': 'https://huggingface.co/openai/whisper-large-v3-turbo/resolve/41f01f3fe87f28c78e2fbf8b568835947dd65ed9/README.md',
}

def fetch(item):
    name, url = item
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read(10_000_001)
            if len(data) > 10_000_000:
                raise ValueError('Bounded original retrieval exceeded 10 MB')
            final_url = response.url
        (root / name).write_bytes(data)
        return {'name': name, 'url': url, 'final_url': final_url, 'accessed_on': '2026-10-04',
                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'status': 'retrieved'}
    except Exception as error:
        return {'name': name, 'url': url, 'status': 'failed', 'error': repr(error)}

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    receipts = list(executor.map(fetch, urls.items()))
Path('docs/technical-reviews/artifacts/natural-v4-factual/R.4/retrieval-receipts.json').write_text(
    json.dumps(receipts, indent=2) + '\n')
print(json.dumps(receipts, indent=2))
