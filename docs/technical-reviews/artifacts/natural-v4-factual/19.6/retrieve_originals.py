"""Retrieve small original authorities with TLS verification; full sources stay ignored."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import requests

ROOT = Path(__file__).resolve().parents[5]
ART = ROOT / 'docs/technical-reviews/artifacts/natural-v4-factual/19.6'
RAW = ROOT / 'outputs/natural-v4/factual-research/19.6'
URLS = {
    'torch-no-grad.html': 'https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html',
    'torch-pool.html': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html',
    'torch-mean.html': 'https://docs.pytorch.org/docs/stable/generated/torch.mean.html',
    'audio-melspectrogram.html': 'https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html',
    'vqa-v2.pdf': 'https://arxiv.org/pdf/1612.00837v3',
    'dpo.pdf': 'https://arxiv.org/pdf/2305.18290v3',
}
def fetch(item):
    name, url = item
    try:
        response = requests.get(url, timeout=35)
        response.raise_for_status()
        (RAW / name).write_bytes(response.content)
        return {'name': name, 'url': url, 'final_url': response.url, 'accessed_on': datetime.now(timezone.utc).date().isoformat(), 'http_status': response.status_code, 'bytes': len(response.content), 'retrieval_sha256': hashlib.sha256(response.content).hexdigest()}
    except Exception as exc:
        return {'name': name, 'url': url, 'error': str(exc)}
if __name__ == '__main__':
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, URLS.items()))
    (ART / 'original-retrieval-receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(json.dumps(receipts, indent=2))
