"""Run exact chapter fences selected for numerical/interface verification only."""
from pathlib import Path
import contextlib
import datetime
import hashlib
import io
import json
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import tokenizers

torch.set_num_threads(1)
selected = ['5.1', '5.2', '5.5', '5.6', '5.7', '5.9', '6.3', '6.9',
            '7.1', '7.4', '7.5', '7.6', '7.7', '7.10', '8.3', '8.8', '8.13']
records = []
for section in selected:
    source = OUT / 'sources' / 'chapters' / f'{section}.md'
    match = re.search(r'^```python\n(.*?)^```', source.read_text(), re.M | re.S)
    if not match:
        raise RuntimeError(f'No Python fence in {section}')
    code = match.group(1)
    (OUT / 'numeric-fences').mkdir(exist_ok=True)
    fence = OUT / 'numeric-fences' / f'{section}.py'
    fence.write_text(code)
    stdout, stderr = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    record = {'section_id': section, 'source': str(source.relative_to(ROOT)),
              'fence': str(fence.relative_to(ROOT)),
              'fence_sha256': hashlib.sha256(code.encode()).hexdigest()}
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, str(fence), 'exec'), {'__name__': '__main__'})
        record['status'] = 'completed_without_exception'
    except Exception:
        record['status'] = 'failed'
        stderr.write(traceback.format_exc())
    record.update(seconds=time.perf_counter()-started, stdout=stdout.getvalue(), stderr=stderr.getvalue())
    records.append(record)

result = {'task_name': '/root/v4_wholebook_scan_05_08',
          'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'command': '.venv/bin/python outputs/natural-v4/wholebook-review/05-08/run_numeric_checks.py',
          'python': sys.version, 'torch': torch.__version__, 'tokenizers': tokenizers.__version__,
          'device': 'cpu', 'torch_threads': 1,
          'scope': '17 exact chapter fences; only 5.1 performs 40 optimizer updates; no complete notebook or long experiment run',
          'checks': records}
(OUT / 'numeric-checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
with (OUT / 'numeric-checks.log').open('w') as stream:
    for record in records:
        stream.write(f"[{record['section_id']}] {record['status']}\n{record['stdout']}{record['stderr']}\n")
print(json.dumps({'completed': len(records), 'failed': [r['section_id'] for r in records if r['status']=='failed']}, ensure_ascii=False))
if any(record['status']=='failed' for record in records):
    raise SystemExit(1)
