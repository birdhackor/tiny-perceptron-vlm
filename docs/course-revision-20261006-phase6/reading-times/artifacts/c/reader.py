import base64
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/workspace/selftrained-v2')
BASE = ROOT / 'docs/course-revision-20261006-phase6/reading-times'
ART = BASE / 'artifacts/c'
ASSIGNMENT = json.loads((BASE / 'assignment-c.json').read_text())
PAGES = {p['page_id']: p for p in ASSIGNMENT['pages']}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verified(page):
    assert sha(ROOT / page['snapshot']) == page['source_sha256']
    for file, digest in page['figures_sha256'].items():
        assert sha(ROOT / file) == digest
    for kind in ('source', 'executed'):
        if f'notebook_{kind}' in page:
            assert sha(ROOT / page[f'notebook_{kind}']) == page[f'notebook_{kind}_sha256']

mode, page_id = sys.argv[1:3]
p = PAGES[page_id]
verified(p)
if mode == 'read':
    print((ROOT / p['snapshot']).read_text())
    for fig in p['figures_sha256']:
        out = ART / (Path(fig).stem + '.png')
        if not out.exists():
            subprocess.run(['inkscape', str(ROOT / fig), '--export-type=png', '--export-width=960', '--export-filename='+str(out)], check=True, capture_output=True)
        print('FIGURE ARTIFACT:', out)
    if 'notebook_executed' in p:
        nb = json.loads((ROOT / p['notebook_executed']).read_text())
        print('EXECUTED NOTEBOOK:', p['notebook_executed'])
        for ci, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            print('CODE CELL', ci, 'execution_count', cell.get('execution_count'))
            print(''.join(cell['source']))
            for oi, output in enumerate(cell.get('outputs', [])):
                print('OUTPUT', oi, output['output_type'])
                if 'text' in output:
                    print(''.join(output['text']))
                if 'traceback' in output:
                    print('\n'.join(output['traceback']))
                for mime, data in output.get('data', {}).items():
                    if mime == 'image/png':
                        target = ART / f'{page_id}-cell{ci}-output{oi}.png'
                        target.write_bytes(base64.b64decode(''.join(data)))
                        print('NOTEBOOK IMAGE ARTIFACT:', target)
                    elif mime == 'image/svg+xml':
                        target = ART / f'{page_id}-cell{ci}-output{oi}.svg'
                        target.write_text(''.join(data))
                        raster = target.with_suffix('.png')
                        subprocess.run(['inkscape', str(target), '--export-type=png', '--export-width=960', '--export-filename='+str(raster)], check=True, capture_output=True)
                        print('NOTEBOOK SVG SHA:', sha(target))
                        print('NOTEBOOK IMAGE ARTIFACT:', raster)
                    elif mime.startswith('text/'):
                        print(mime, ''.join(data))
                    else:
                        print('ADDITIONAL OUTPUT MIME', mime)
elif mode == 'record':
    manual = json.load(sys.stdin)
    low, high = manual['minutes_min'], manual['minutes_max']
    assert isinstance(low, int) and isinstance(high, int) and 0 < low <= high
    report_path = BASE / 'reports/c.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'schema_version':1, 'estimate_kind':'ai_estimate', 'pages':[]}
    assert len(report['pages']) < len(ASSIGNMENT['page_ids'])
    assert ASSIGNMENT['page_ids'][len(report['pages'])] == page_id
    views = manual.get('viewed_artifacts', [])
    for v in views:
        assert (ROOT / v['artifact']).exists()
        v['artifact_sha256'] = sha(ROOT / v['artifact'])
    entry = {k:p[k] for k in ('page_id','source_sha256','figures_sha256')}
    entry.update(minutes_min=low, minutes_max=high, reviewer_task='/root/p6_time_c', reason=manual['reason'])
    trace = dict(entry)
    trace.update(reading_order=len(report['pages'])+1, recorded_at_utc=datetime.now(timezone.utc).isoformat(), source_snapshot=p['snapshot'], summary=manual['summary'], viewed_artifacts=views)
    for k in ('notebook_source','notebook_source_sha256','notebook_executed','notebook_executed_sha256'):
        if k in p: trace[k] = p[k]
    trace['notebook_outputs_read'] = manual.get('notebook_outputs_read', [])
    report['pages'].append(entry)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    trace_path = BASE / 'traces/c.jsonl'
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open('a') as f: f.write(json.dumps(trace, ensure_ascii=False)+'\n')
    print('Recorded',page_id,low,high)
