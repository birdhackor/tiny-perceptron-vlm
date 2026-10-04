"""Bounded original-authority retrieval for this independent DATA review."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request, datetime

BASE = Path('outputs/natural-v4/factual-research/data')
BASE.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'docci-card.md': 'https://huggingface.co/datasets/google/docci/raw/a0a43eaf34676ffd008fb6565dd8c2ba00d09100/README.md',
    'docci-site.html': 'https://google.github.io/docci/',
    'docci-paper.pdf': 'https://arxiv.org/pdf/2404.19753',
    'lora-paper.pdf': 'https://arxiv.org/pdf/2106.09685',
    'nvidia-card.md': 'https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v1/raw/69696a1cc543ef3a0f8e9892a89c17293e915263/README.md',
    'oasst2-card.md': 'https://huggingface.co/datasets/OpenAssistant/oasst2/raw/179dd21fc55192153d94adb0e0ce8f69e222bf75/README.md',
    'fleurs-card.md': 'https://huggingface.co/datasets/google/fleurs/raw/d7c758a6dceecd54a98cac43404d3d576e721f07/README.md',
    'aishell-card.md': 'https://huggingface.co/datasets/AISHELL/AISHELL-1/raw/bbe295d530192a4cd41644b711c9aecd087df653/README.md',
    'aishell-openslr.html': 'https://www.openslr.org/33',
    'python-venv.html': 'https://docs.python.org/3.13/library/venv.html',
    'git-clone.html': 'https://git-scm.com/docs/git-clone',
    'git-lfs-pointer.md': 'https://raw.githubusercontent.com/git-lfs/git-lfs/main/docs/spec.md',
    'git-lfs-config.html': 'https://www.mankier.com/5/git-lfs-config',
    'cc-by-4.html': 'https://creativecommons.org/licenses/by/4.0/legalcode.en',
    'cc-by-3.html': 'https://creativecommons.org/licenses/by/3.0/legalcode.en',
    'cc0.html': 'https://creativecommons.org/publicdomain/zero/1.0/legalcode.en',
    'cer-metric.md': 'https://raw.githubusercontent.com/huggingface/evaluate/main/metrics/cer/README.md',
    'pillow-12.3.0.json': 'https://pypi.org/pypi/Pillow/12.3.0/json',
    'h5py-3.16.0.json': 'https://pypi.org/pypi/h5py/3.16.0/json',
    'numpy-2.5.3.json': 'https://pypi.org/pypi/numpy/2.5.3/json',
    'soundfile-0.13.1.json': 'https://pypi.org/pypi/soundfile/0.13.1/json',
    'pinned-build.py': 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/9a61ecf524c9518f33f1501c28aa72997d4a82d0/scripts/build_natural_v4_assets.py',
    'pinned-fetch.py': 'https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/59a1eda4ed7b6e8609892ec2b9013c821ac93e69/scripts/fetch_natural_data.py',
}

def fetch(item):
    name, url = item
    result = {'name': name, 'url': url, 'accessed_on': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'Independent-DATA-factual-review/1.0', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=35) as response:
            raw = response.read(12 * 1024 * 1024 + 1)
            if len(raw) > 12 * 1024 * 1024:
                raise ValueError('Original text exceeds this review retrieval bound')
            result.update({'status': response.status, 'final_url': response.geturl(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        (BASE/name).write_bytes(raw)
    except Exception as exc:
        result.update({'error': type(exc).__name__ + ': ' + str(exc)})
    return result

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(fetch, SOURCES.items()))
    target = Path('docs/technical-reviews/artifacts/natural-v4-supplemental/data/authority-retrieval.json')
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
    for record in records:
        print(json.dumps(record, ensure_ascii=False))
