import json,hashlib,pathlib,sys,subprocess,base64,datetime
root=pathlib.Path('/workspace/selftrained-v2')
base=root/'docs/course-revision-20261006-phase6/reading-times'
assignment=json.loads((base/'assignment-d.json').read_text())
art=base/'artifacts/d'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def meta(pid): return next(p for p in assignment['pages'] if p['page_id']==pid)
def check(p):
    checks=[(p['snapshot'],p['source_sha256'])]+list(p['figures_sha256'].items())
    if p.get('notebook_executed'): checks += [(p['notebook_source'],p['notebook_source_sha256']),(p['notebook_executed'],p['notebook_executed_sha256'])]
    for path,expected in checks:
        actual=sha(root/path)
        if actual!=expected: raise RuntimeError(f'SHA MISMATCH {path}: {actual} != {expected}')
if sys.argv[1]=='read':
    p=meta(sys.argv[2]);check(p)
    print('PAGE METADATA',json.dumps(p,ensure_ascii=False))
    print('=== FULL FROZEN PAGE ===\n'+(root/p['snapshot']).read_text())
    for fig in p['figures_sha256']:
        out=art/(pathlib.Path(fig).stem+'.png')
        if not out.exists(): subprocess.run(['inkscape',str(root/fig),'--export-type=png','--export-width=960','--export-filename='+str(out)],check=True,capture_output=True)
        print('FIGURE RENDER',str(out.relative_to(root)),sha(out))
    if p.get('notebook_executed'):
        nb=json.loads((root/p['notebook_executed']).read_text())
        print('=== EXECUTED NOTEBOOK: ALL CELLS AND REAL OUTPUTS ===')
        for i,c in enumerate(nb['cells']):
            if c['cell_type'] != 'code': continue
            if ''.join(c.get('source',[])).startswith('# @title 準備本節的工具'): 
                print(f'CELL {i}: repeated installation/preparation boilerplate, excluded from estimate; output types', [o['output_type'] for o in c.get('outputs',[])])
                continue
            print(f'CELL {i} {c["cell_type"]}\n'+''.join(c.get('source',[])))
            for j,o in enumerate(c.get('outputs',[])):
                print(f'OUTPUT {i}.{j} {o["output_type"]}')
                if 'text' in o: print(''.join(o['text']))
                if 'traceback' in o: print('\n'.join(o['traceback']))
                for mime,data in o.get('data',{}).items():
                    if mime=='image/png':
                        out=art/f'{p["page_id"]}-nb-{i}-{j}.png';out.write_bytes(base64.b64decode(''.join(data)))
                        print('NOTEBOOK IMAGE',str(out.relative_to(root)),sha(out))
                    elif mime=='image/svg+xml':
                        svg=art/f'{p["page_id"]}-nb-{i}-{j}.svg';svg.write_text(''.join(data))
                        out=svg.with_suffix('.png');subprocess.run(['inkscape',str(svg),'--export-type=png','--export-width=960','--export-filename='+str(out)],check=True,capture_output=True)
                        print('NOTEBOOK IMAGE',str(out.relative_to(root)),sha(out))
                    else: print(mime, ''.join(data) if isinstance(data,list) else data)
        print('=== END PAGE ===')
elif sys.argv[1]=='record':
    v=json.load(sys.stdin);p=meta(v['page_id']);check(p)
    rp=base/'reports/d.json';tp=base/'traces/d.jsonl'
    report=json.loads(rp.read_text()) if rp.exists() else {'schema_version':1,'estimate_kind':'ai_estimate','pages':[]}
    assert len(report['pages'])<len(assignment['page_ids'])
    assert assignment['page_ids'][len(report['pages'])]==v['page_id'],'Must record in assignment order'
    assert isinstance(v['minutes_min'],int) and isinstance(v['minutes_max'],int) and 0<v['minutes_min']<=v['minutes_max']
    row={k:v[k] for k in ('page_id','minutes_min','minutes_max','reason')}
    row.update(reviewer_task='/root/p6_time_d',source_sha256=p['source_sha256'],figures_sha256=p['figures_sha256'])
    trace=dict(row,snapshot=p['snapshot'],read_summary=v['summary'],figure_views=v.get('figure_views',[]),notebook=p.get('notebook_executed'),notebook_sha256=p.get('notebook_executed_sha256'),notebook_source=p.get('notebook_source'),notebook_source_sha256=p.get('notebook_source_sha256'),notebook_observation=v.get('notebook_observation'),notebook_image_views=v.get('notebook_image_views',[]),recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for item in trace['figure_views']+trace['notebook_image_views']:
        item['artifact_sha256']=sha(root/item['artifact'])
    with tp.open('a') as f:f.write(json.dumps(trace,ensure_ascii=False)+'\n')
    report['pages'].append(row);rp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('RECORDED',v['page_id'],'PROGRESS',len(report['pages']),len(assignment['page_ids']))
elif sys.argv[1]=='validate':
    report=json.loads((base/'reports/d.json').read_text());traces=[json.loads(s) for s in (base/'traces/d.jsonl').read_text().splitlines()]
    assert [p['page_id'] for p in report['pages']]==assignment['page_ids']
    assert [p['page_id'] for p in traces]==assignment['page_ids']
    for p,t in zip(report['pages'],traces):
        m=meta(p['page_id']);check(m)
        assert p['source_sha256']==m['source_sha256'] and p['figures_sha256']==m['figures_sha256']
        assert 0<p['minutes_min']<=p['minutes_max'] and p['reason'] and t['read_summary']
        assert len(t['figure_views'])==len(m['figures_sha256'])
        for item in t['figure_views']+t['notebook_image_views']: assert sha(root/item['artifact'])==item['artifact_sha256']
    print(json.dumps({'pages':len(report['pages']),'minutes_min':sum(p['minutes_min'] for p in report['pages']),'minutes_max':sum(p['minutes_max'] for p in report['pages']),'status':'SHA/scope/record consistency checked'},ensure_ascii=False))
