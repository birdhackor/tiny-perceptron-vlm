import hashlib
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / 'official'
OUT.mkdir(exist_ok=True)
URLS = {
    'pytorch-saving-loading.html': 'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html',
    'pytorch-randomness.html': 'https://docs.pytorch.org/docs/stable/notes/randomness.html',
    'hf-download.html': 'https://huggingface.co/docs/huggingface_hub/v1.33.0/guides/download',
    'hf-file-download.py': 'https://raw.githubusercontent.com/huggingface/huggingface_hub/v1.33.0/src/huggingface_hub/file_download.py',
    'safetensors-readme.md': 'https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/README.md',
    'safetensors-torch.py': 'https://raw.githubusercontent.com/huggingface/safetensors/v0.8.0/bindings/python/py_src/safetensors/torch.py',
    'hf-immutable-metadata.json': 'https://huggingface.co/api/models/birdhackor/tiny-perceptron-course-models/revision/979cdfacc588ad0536f1c64fff96f264571cf054?blobs=true',
}
rows = []
for filename, url in URLS.items():
    request = urllib.request.Request(url, headers={'User-Agent': 'factual-review-19.11/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            (OUT / filename).write_bytes(raw)
            rows.append({'name': filename, 'url': url, 'final_url': response.url,
                         'http_status': response.status, 'bytes': len(raw),
                         'sha256': hashlib.sha256(raw).hexdigest(),
                         'authentication': 'No Authorization header; anonymous urllib request',
                         'accessed_on': '2026-10-06'})
    except Exception as error:
        rows.append({'name': filename, 'url': url, 'error': repr(error)})
(OUT / 'fetch-results.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
