"""Preserve publicly cited original papers, TLS-verified with curl defaults."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys

OUT = Path(__file__).resolve().parent
sources = [
 ('instructgpt-v1', 'https://arxiv.org/pdf/2203.02155v1'),
 ('deepseek-r1-v1', 'https://arxiv.org/pdf/2501.12948v1'),
 ('lora-v1', 'https://arxiv.org/pdf/2106.09685v1'),
 ('lora-v2', 'https://arxiv.org/pdf/2106.09685v2'),
 ('dpo-v1', 'https://arxiv.org/pdf/2305.18290v1'),
 ('rslora-v1', 'https://arxiv.org/pdf/2312.03732v1'),
]
def fetch(item):
    name, url = item
    pdf = OUT / 'primary-sources' / f'{name}.pdf'
    cmd = ['curl', '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '20', '--max-time', '60', url, '-o', str(pdf)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    record = {'name': name, 'url': url, 'command': cmd, 'exit_code': completed.returncode,
              'stderr': completed.stderr, 'fetched_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if completed.returncode == 0:
        record['pdf_sha256'] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        txt = pdf.with_suffix('.txt')
        extracted = subprocess.run(['pdftotext', '-layout', str(pdf), str(txt)], capture_output=True, text=True)
        record['text_extraction_exit_code'] = extracted.returncode
        record['text_extraction_stderr'] = extracted.stderr
        if extracted.returncode == 0:
            record['text_sha256'] = hashlib.sha256(txt.read_bytes()).hexdigest()
    return record
if sys.argv[1:]:
    sources = [item for item in sources if item[0] in sys.argv[1:]]
with ThreadPoolExecutor(max_workers=5) as pool:
    records = list(pool.map(fetch, sources))
receipt = OUT / 'primary-sources' / 'fetch-receipt.json'
previous = json.loads(receipt.read_text()) if receipt.exists() else []
receipt.write_text(json.dumps(previous+records, ensure_ascii=False, indent=2)+'\n')
print(json.dumps([{key: row[key] for key in ['name','exit_code']} for row in records]))
