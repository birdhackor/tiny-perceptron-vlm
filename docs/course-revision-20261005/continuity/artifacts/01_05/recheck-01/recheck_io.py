import sys, json, re, hashlib, html
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-revision-20261005/continuity'
OUT = BASE / 'artifacts/01_05/recheck-01'
TRACE = BASE / 'traces/01_05-recheck-01.jsonl'
INV = {x['page_id']: x for x in json.loads((BASE / 'revised-01/inventory.json').read_text())['pages']}

def snapshot(p):
    path = ROOT / INV[p]['snapshot']
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == INV[p]['source_sha256']
    return raw.decode('utf-8')

def outputs(p):
    url = f'http://127.0.0.1:8765/{p}.html'
    raw = urlopen(url, timeout=15).read()
    (OUT / f'{p}.html').write_bytes(raw)
    values = [html.unescape(re.sub(r'<[^>]*>', '', x)) for x in re.findall(r'<pre[\s\S]*?</pre>', raw.decode()) if '<code' not in x]
    return values

if sys.argv[1] in ('read', 'outputs'):
    p = sys.argv[2]
    if sys.argv[1] == 'read':
        text = snapshot(p)
        if len(sys.argv) > 3:
            units = re.split(r'(?=^## )', text, flags=re.M)
            n = int(sys.argv[3])
            print(f'PAGE {p}, UNIT {n + 1}/{len(units)}')
            print(units[n])
        else:
            print(f'PAGE {p}')
            print(text)
    if len(sys.argv) == 3:
        print('ACTUAL WEBSITE OUTPUTS:', '\n'.join(outputs(p)) or '(none)')
elif sys.argv[1] == 'record':
    data = json.load(sys.stdin)
    p = data['page_id']
    snapshot(p)
    data.update(recorded_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=INV[p]['source_sha256'], figures_sha256=INV[p]['figures_sha256'], source_snapshot=INV[p]['snapshot'], source_sha_verified=True)
    if data.get('visual_verified'):
        rows = [json.loads(x) for x in (OUT / 'pageviews.jsonl').read_text().splitlines()]
        for row in rows:
            if row['page_id'] == p and not row.get('viewed'):
                row.update(viewed=True, viewed_at_utc=data['recorded_at_utc'])
        (OUT / 'pageviews.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    with TRACE.open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(data['recorded_at_utc'], p, data['section'])
