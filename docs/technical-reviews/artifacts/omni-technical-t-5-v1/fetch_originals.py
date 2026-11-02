"""Fetch original references outside Git; retain only a short receipt in this directory."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[4]
DEST = Path('/workspace/work/reference/omni-technical-t-5-v1')
DEST.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'instructgpt-v1.pdf': 'https://arxiv.org/pdf/2203.02155v1',
    'ifeval-v1.pdf': 'https://arxiv.org/pdf/2311.07911v1',
    'xstest-v1.pdf': 'https://arxiv.org/pdf/2308.01263v1',
    'lora-v2.pdf': 'https://arxiv.org/pdf/2106.09685v2',
    'pku-card.md': 'https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md',
    'transformers-4.57.1-config.py': 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/generation/configuration_utils.py',
}

def fetch(item):
    name, url = item
    entry = {'file': str(DEST / name), 'url': url, 'accessed_at': datetime.now(UTC).isoformat()}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'technical-review/1.0'}), timeout=45) as response:
            body = response.read()
            entry.update(status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'))
        (DEST / name).write_bytes(body)
        entry.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        if name.endswith('.pdf'):
            out = DEST / name.replace('.pdf', '.txt')
            result = subprocess.run(['pdftotext', '-layout', str(DEST / name), str(out)], capture_output=True, text=True)
            entry.update(extraction_returncode=result.returncode, extraction_stderr=result.stderr)
            if out.exists():
                entry.update(text_file=str(out), text_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    except Exception as error:
        entry['error'] = f'{type(error).__name__}: {error}'
    return entry

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, SOURCES.items()))
    output = Path(__file__).with_name('original-fetch-receipts.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipts, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(receipts, ensure_ascii=False, indent=2))
