import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/selftrained-v2')
BASE = ROOT / 'docs/course-revision-20261006-phase6/reading-times'
ART = BASE / 'artifacts/e'
ASSIGNMENT = json.loads((BASE / 'assignment-e.json').read_text())
TASK = '/root/p6_time_e'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def checked(page_id):
    assert sha(ROOT / ASSIGNMENT['inventory']) == ASSIGNMENT['inventory_sha256']
    p = next(p for p in ASSIGNMENT['pages'] if p['page_id'] == page_id)
    assert sha(ROOT / p['snapshot']) == p['source_sha256'], page_id
    for k, v in p['figures_sha256'].items():
        assert sha(ROOT / k) == v, k
    for prefix in ('notebook_source', 'notebook_executed'):
        if prefix in p:
            assert sha(ROOT / p[prefix]) == p[prefix + '_sha256'], p[prefix]
    return p

def read(page_id):
    p = checked(page_id)
    print('PAGE', page_id, p['title'])
    print((ROOT / p['snapshot']).read_text())
    for f in p['figures_sha256']:
        out = ART / (Path(f).stem + '.png')
        if not out.exists():
            subprocess.run(['inkscape', str(ROOT / f), '--export-width=960', '--export-filename=' + str(out)], check=True, capture_output=True)
        print('REQUIRED FIGURE ARTIFACT', str(out))
    if 'notebook_executed' in p:
        nb = json.loads((ROOT / p['notebook_executed']).read_text())
        print('EXECUTED NOTEBOOK', p['notebook_executed'], p['notebook_executed_sha256'])
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'code':
                continue
            print('CODE CELL', i, 'execution_count', c.get('execution_count'))
            print(''.join(c['source']))
            for j, o in enumerate(c.get('outputs', [])):
                print('OUTPUT', j, o['output_type'])
                if 'text' in o:
                    print(''.join(o['text']))
                if 'traceback' in o:
                    print('\n'.join(o['traceback']))
                for mime, v in o.get('data', {}).items():
                    if mime in ('image/png', 'image/jpeg'):
                        ext = 'png' if mime == 'image/png' else 'jpg'
                        out = ART / f'notebook-{page_id}-cell{i}-output{j}.{ext}'
                        out.write_bytes(base64.b64decode(''.join(v)))
                        print('REQUIRED NOTEBOOK IMAGE', str(out))
                    elif mime == 'image/svg+xml':
                        svg = ART / f'notebook-{page_id}-cell{i}-output{j}.svg'
                        svg.write_text(''.join(v))
                        out = svg.with_suffix('.png')
                        subprocess.run(['inkscape', str(svg), '--export-width=960', '--export-filename=' + str(out)], check=True, capture_output=True)
                        print('REQUIRED NOTEBOOK IMAGE', str(out))
                    else:
                        print(mime, ''.join(v) if isinstance(v, list) else v)

def record(d):
    p = checked(d['page_id'])
    report = BASE / 'reports/e.json'
    trace = BASE / 'traces/e.jsonl'
    report.parent.mkdir(exist_ok=True)
    trace.parent.mkdir(exist_ok=True)
    data = json.loads(report.read_text()) if report.exists() else {'schema_version': 1, 'estimate_kind': 'ai_estimate', 'pages': []}
    assert d['page_id'] == ASSIGNMENT['page_ids'][len(data['pages'])], 'preserve assigned read order'
    assert 0 < d['minutes_min'] <= d['minutes_max']
    row = {k: d[k] for k in ('page_id', 'minutes_min', 'minutes_max', 'reason')}
    row.update(reviewer_task=TASK, source_sha256=p['source_sha256'], figures_sha256=p['figures_sha256'])
    tr = dict(row, sequence=len(data['pages'])+1, read_at=datetime.now(timezone.utc).isoformat(), source=p['source'], selector=p['selector'], snapshot=p['snapshot'], summary=d['summary'], figure_views=d.get('figure_views', []), notebook_output_images_viewed=d.get('notebook_output_images_viewed', []))
    for k in ('notebook_source', 'notebook_source_sha256', 'notebook_executed', 'notebook_executed_sha256'):
        if k in p:
            tr[k] = p[k]
    for view in tr['figure_views']:
        artifact = ROOT / view['artifact']
        assert artifact.exists()
        view['artifact_sha256'] = sha(artifact)
        view['figure_sha256'] = p['figures_sha256'][view['figure']]
    data['pages'].append(row)
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    with trace.open('a') as f:
        f.write(json.dumps(tr, ensure_ascii=False) + '\n')
    print('RECORDED', d['page_id'], d['minutes_min'], d['minutes_max'], 'completed', len(data['pages']))

ART.mkdir(parents=True, exist_ok=True)
if sys.argv[1] == 'read':
    read(sys.argv[2])
elif sys.argv[1] == 'record':
    record(json.load(sys.stdin))
elif sys.argv[1] == 'validate':
    rows = json.loads((BASE / 'reports/e.json').read_text())['pages']
    traces = [json.loads(line) for line in (BASE / 'traces/e.jsonl').read_text().splitlines()]
    assert [p['page_id'] for p in rows] == ASSIGNMENT['page_ids']
    assert [p['page_id'] for p in traces] == ASSIGNMENT['page_ids']
    for r in rows:
        p = checked(r['page_id'])
        assert r['source_sha256'] == p['source_sha256'] and r['figures_sha256'] == p['figures_sha256']
    print('VALIDATED', len(rows), 'pages; total range', sum(r['minutes_min'] for r in rows), sum(r['minutes_max'] for r in rows))
