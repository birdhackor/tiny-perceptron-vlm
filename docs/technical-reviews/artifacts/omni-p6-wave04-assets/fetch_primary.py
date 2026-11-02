"""Read-only, bounded primary-source acquisition; never fetch model payloads."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-p6-wave04-assets')
MAX_BYTES = 3 * 1024**2
URLS = {
    'lfs-spec-v3.6.1.md': 'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.6.1/docs/spec.md',
    'lfs-smudge-v3.6.1.adoc': 'https://raw.githubusercontent.com/git-lfs/git-lfs/v3.6.1/docs/man/git-lfs-smudge.adoc',
    'github-lfs-about.html': 'https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage',
    'github-lfs-billing.html': 'https://docs.github.com/en/billing/concepts/product-billing/git-lfs',
    'hf-storage-limits.html': 'https://huggingface.co/docs/hub/storage-limits',
    'pytorch-save-load.html': 'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html',
    'pytorch-randomness.html': 'https://docs.pytorch.org/docs/2.9/notes/randomness.html',
    'peft-checkpoint.html': 'https://huggingface.co/docs/peft/developer_guides/checkpoint',
    'github-release-api.json': 'https://api.github.com/repos/birdhackor/tiny-perceptron-vlm/releases/tags/omni-core-candidate-cpu-core-40988311100-1',
    'hf-v2-tree.json': 'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2?recursive=true&expand=false',
    'hf-v2-root-readme.md': 'https://huggingface.co/birdhackor/tiny-perceptron-course-models/raw/979cdfacc588ad0536f1c64fff96f264571cf054/README.md',
    'hf-natural-v4-tree.json': 'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/tree/d3954d6900b3cf81e593d99d9b8b1a91e6f9741d/natural-v3/assistant-2b-v4?recursive=true&expand=false',
}
for component in ['moe-pretrain', 'moe-sft', 'moe-joint', 'dense-joint']:
    for name in ['inference-manifest.json', 'model-config.json', 'tokenizer.json']:
        URLS[f'hf-v2-{component}-{name}'] = (
            'https://huggingface.co/birdhackor/tiny-perceptron-course-models/raw/'
            f'979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/{component}/{name}'
        )
for name in ['README.md', 'release-provenance.json']:
    URLS[f'hf-natural-v4-{name}'] = (
        'https://huggingface.co/birdhackor/tiny-perceptron-course-models/raw/'
        f'd3954d6900b3cf81e593d99d9b8b1a91e6f9741d/natural-v3/assistant-2b-v4/{name}'
    )


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.fragments = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.skip += 1
        if tag in {'p', 'div', 'li', 'h1', 'h2', 'h3', 'tr', 'pre', 'section'}:
            self.fragments.append('\n')

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.skip -= 1
        if tag in {'p', 'div', 'li', 'h1', 'h2', 'h3', 'tr', 'pre', 'section'}:
            self.fragments.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.fragments.append(data)


def acquire(item):
    name, url = item
    started = datetime.now(UTC).isoformat()
    request = urllib.request.Request(url, headers={'User-Agent': 'independent-assets-technical-review', 'Accept': 'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError('Metadata exceeds explicit 3 MiB acquisition limit')
            record = {
                'name': name, 'url': url, 'final_url': response.url,
                'status': response.status, 'accessed_at': started,
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                'cache_path': str(CACHE / name),
                'response_headers': {key: response.headers[key] for key in ['Content-Type', 'ETag', 'Last-Modified', 'X-Repo-Commit', 'X-Linked-Size', 'Link'] if key in response.headers},
            }
        (CACHE / name).write_bytes(raw)
        if name.endswith('.html'):
            parser = PlainText()
            parser.feed(raw.decode('utf-8'))
            lines = [re.sub(r'\s+', ' ', line).strip() for line in ''.join(parser.fragments).splitlines()]
            (CACHE / (name + '.txt')).write_text('\n'.join(line for line in lines if line) + '\n')
        return record
    except Exception as exc:
        return {'name': name, 'url': url, 'accessed_at': started, 'error': f'{type(exc).__name__}: {exc}'}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(acquire, URLS.items()))
    output = {'created_at': datetime.now(UTC).isoformat(), 'command': 'fetch_primary.py', 'python': sys.version, 'scope': 'Unauthenticated GET of official docs and public machine metadata only; no model tensor/archive requests.', 'sources': records}
    (Path(__file__).parent / 'primary-receipts-v1.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    for row in records:
        print(row['name'], row.get('status'), row.get('bytes'), row.get('error', 'OK'))
    if any('error' in row for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
