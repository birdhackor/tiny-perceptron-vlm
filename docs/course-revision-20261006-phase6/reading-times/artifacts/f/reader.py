import base64, datetime, hashlib, json, subprocess, sys
from pathlib import Path

BASE = Path('docs/course-revision-20261006-phase6/reading-times')
ART = BASE / 'artifacts/f'
ASSIGN = json.loads((BASE / 'assignment-f.json').read_text())
PAGES = {p['page_id']: p for p in ASSIGN['pages']}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def render(source, target):
    subprocess.run(['inkscape', str(source), '--export-type=png', '--export-width=960', '--export-filename=' + str(target)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read(pid):
    p = PAGES[pid]
    assert sha(p['snapshot']) == p['source_sha256']
    print('PAGE', pid, 'SNAPSHOT', p['snapshot'], 'SHA', p['source_sha256'])
    print(Path(p['snapshot']).read_text())
    views = []
    for source, expected in p['figures_sha256'].items():
        assert sha(source) == expected
        target = ART / (Path(source).stem + '.png')
        if not target.exists(): render(source, target)
        views.append({'source': source, 'source_sha256': expected, 'artifact': str(target), 'artifact_sha256': sha(target)})
    notebook = None
    output_images = []
    if p.get('notebook_executed'):
        for key in ['notebook_source', 'notebook_executed']:
            assert sha(p[key]) == p[key + '_sha256']
        notebook = {'path': p['notebook_executed'], 'sha256': p['notebook_executed_sha256'], 'source_path': p['notebook_source'], 'source_sha256': p['notebook_source_sha256']}
        n = json.loads(Path(p['notebook_executed']).read_text())
        print('\nEXECUTED NOTEBOOK', notebook)
        for i, c in enumerate(n['cells']):
            if c['cell_type'] != 'code': continue
            code = ''.join(c['source'])
            print('\nCODE CELL', i)
            if '# @title 準備本節的工具' in code:
                print('[common preparation code already read in 20.1; no installation performed]')
            else: print(code)
            for j, o in enumerate(c.get('outputs', [])):
                if o.get('text'): print('ACTUAL OUTPUT:', ''.join(o['text']))
                if o.get('ename'): print('ERROR:', o['ename'], o.get('evalue'))
                for k, v in o.get('data', {}).items():
                    if k.startswith('text/') and k != 'text/html': print(k, ''.join(v))
                    if k == 'image/svg+xml':
                        raw = ''.join(v).encode()
                        same = next((a for a in views if Path(a['source']).read_bytes() == raw), None)
                        if same:
                            output_images.append({'cell': i, 'output': j, 'mime': k, 'matches_figure': same['source'], 'artifact': same['artifact']})
                        else:
                            source = ART / f'{pid}-cell{i}-output{j}.svg'; source.write_bytes(raw)
                            target = source.with_suffix('.png'); render(source, target)
                            output_images.append({'cell': i, 'output': j, 'mime': k, 'artifact': str(target), 'artifact_sha256': sha(target)})
                    if k == 'image/png':
                        target = ART / f'{pid}-cell{i}-output{j}.png'; target.write_bytes(base64.b64decode(''.join(v)))
                        output_images.append({'cell': i, 'output': j, 'mime': k, 'artifact': str(target), 'artifact_sha256': sha(target)})
    inputs = {'page_id': pid, 'snapshot': p['snapshot'], 'source_sha256': p['source_sha256'], 'figures': views, 'notebook': notebook, 'notebook_output_images': output_images}
    (ART / f'{pid}-inputs.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')
    print('\nIMAGES TO VIEW:', json.dumps(views + output_images, ensure_ascii=False))

def record(pid, low, high, reason, summary, view_note):
    p = PAGES[pid]
    target = BASE / 'reports/f.json'
    report = json.loads(target.read_text()) if target.exists() else {'schema_version': 1, 'estimate_kind': 'ai_estimate', 'pages': []}
    assert pid not in [q['page_id'] for q in report['pages']]
    assert ASSIGN['page_ids'][len(report['pages'])] == pid
    assert 0 < low <= high
    item = {'page_id': pid, 'minutes_min': low, 'minutes_max': high, 'reviewer_task': '/root/p6_time_f', 'reason': reason, 'source_sha256': p['source_sha256'], 'figures_sha256': p['figures_sha256']}
    inputs = json.loads((ART / f'{pid}-inputs.json').read_text())
    trace = {**inputs, 'reviewer_task': '/root/p6_time_f', 'read_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'summary': summary, 'actual_view_record': view_note, 'minutes_min': low, 'minutes_max': high, 'reason': reason}
    report['pages'].append(item)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (BASE / 'traces/f.jsonl').open('a') as f: f.write(json.dumps(trace, ensure_ascii=False) + '\n')
    print('SAVED', pid, low, high)

if __name__ == '__main__':
    if sys.argv[1] == 'read': read(sys.argv[2])
    elif sys.argv[1] == 'record': record(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *sys.argv[5:])
