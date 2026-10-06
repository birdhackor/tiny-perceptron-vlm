import base64, datetime, hashlib, json, pathlib, subprocess, sys

ROOT = pathlib.Path('/workspace/selftrained-v2')
BASE = ROOT / 'docs/course-revision-20261006-phase6/reading-times'
ASSIGN = json.loads((BASE / 'assignment-b.json').read_text())
PAGES = {p['page_id']: p for p in ASSIGN['pages']}
ART = BASE / 'artifacts/b'
TASK = '/root/p6_time_b'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verified(p):
    assert sha(ROOT / p['snapshot']) == p['source_sha256']
    for f, h in p['figures_sha256'].items():
        assert sha(ROOT / f) == h
    for k in ('notebook_source', 'notebook_executed'):
        if k in p:
            assert sha(ROOT / p[k]) == p[k + '_sha256']

def read(pid):
    p = PAGES[pid]
    verified(p)
    print('SOURCE', p['snapshot'], 'SHA', p['source_sha256'])
    print((ROOT / p['snapshot']).read_text())
    for f in p['figures_sha256']:
        out = ART / (pathlib.Path(f).stem + '.png')
        if not out.exists():
            subprocess.run(['inkscape', str(ROOT/f), '--export-type=png', '--export-width=960', '--export-filename='+str(out)], check=True, capture_output=True)
        print('FIGURE_ARTIFACT', out)
    if 'notebook_executed' in p:
        print('EXECUTED_NOTEBOOK', p['notebook_executed'], 'SHA', p['notebook_executed_sha256'])
        nb = json.loads((ROOT/p['notebook_executed']).read_text())
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'code':
                continue
            print('CELL', i, c['cell_type'])
            print(''.join(c.get('source', [])))
            if c['cell_type'] == 'code':
                print('EXECUTION_COUNT', c.get('execution_count'))
                for j, o in enumerate(c.get('outputs', [])):
                    if 'text' in o:
                        print('OUTPUT', ''.join(o['text']))
                    if 'data' in o:
                        d=o['data']
                        if 'text/plain' in d:
                            print('OUTPUT', ''.join(d['text/plain']))
                        for mime, ext in [('image/png','png'), ('image/jpeg','jpg')]:
                            if mime in d:
                                out=ART/f'{pid}-cell-{i}-output-{j}.{ext}'
                                out.write_bytes(base64.b64decode(''.join(d[mime])))
                                print('NOTEBOOK_IMAGE_ARTIFACT',out)
                        if 'image/svg+xml' in d:
                            out=ART/f'{pid}-cell-{i}-output-{j}.svg'
                            out.write_text(''.join(d['image/svg+xml']))
                            png=out.with_suffix('.png')
                            subprocess.run(['inkscape', str(out), '--export-type=png', '--export-width=960','--export-filename='+str(png)], check=True,capture_output=True)
                            print('NOTEBOOK_IMAGE_ARTIFACT',png)
                    if o.get('output_type') == 'error':
                        print('ERROR',o.get('ename'),o.get('evalue'),o.get('traceback'))

def record(data):
    p=PAGES[data['page_id']]
    verified(p)
    report=BASE/'reports/b.json'
    result=json.loads(report.read_text()) if report.exists() else {'schema_version':1,'estimate_kind':'ai_estimate','pages':[]}
    n=len(result['pages'])
    assert ASSIGN['page_ids'][n] == p['page_id'], ('Unexpected reading order',n,p['page_id'])
    assert isinstance(data['minutes_min'],int) and 0<data['minutes_min']<=data['minutes_max']
    entry={k:data[k] for k in ('page_id','minutes_min','minutes_max','reason')}
    entry.update(reviewer_task=TASK,source_sha256=p['source_sha256'],figures_sha256=p['figures_sha256'])
    result['pages'].append(entry)
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    trace=dict(entry)
    trace.update(read_sequence=n+1,recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot=p['snapshot'],summary=data['summary'],figure_views=data.get('figure_views',[]),notebook_output_observation=data.get('notebook_output_observation'))
    for k in ('notebook_source','notebook_source_sha256','notebook_executed','notebook_executed_sha256'):
        if k in p: trace[k]=p[k]
    with (BASE/'traces/b.jsonl').open('a') as f: f.write(json.dumps(trace,ensure_ascii=False)+'\n')
    print('RECORDED',p['page_id'],data['minutes_min'],data['minutes_max'])

if sys.argv[1]=='read': read(sys.argv[2])
elif sys.argv[1]=='record': record(json.loads(sys.stdin.read()))
elif sys.argv[1]=='verify':
    result=json.loads((BASE/'reports/b.json').read_text())
    traces=[json.loads(x) for x in (BASE/'traces/b.jsonl').read_text().splitlines()]
    assert [p['page_id'] for p in result['pages']]==ASSIGN['page_ids']
    assert [p['page_id'] for p in traces]==ASSIGN['page_ids']
    for p in ASSIGN['pages']: verified(p)
    print('COMPLETE',len(result['pages']),'TOTAL_MIN',sum(p['minutes_min'] for p in result['pages']),'TOTAL_MAX',sum(p['minutes_max'] for p in result['pages']))
