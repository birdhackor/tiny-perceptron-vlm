import json, hashlib, base64, sys, subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/workspace/selftrained-v2')
BASE=ROOT/'docs/course-revision-20261006-phase6/reading-times'
ASSIGN=json.loads((BASE/'assignment-a.json').read_text())
PAGES={p['page_id']:p for p in ASSIGN['pages']}
ART=BASE/'artifacts/a'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def check(p):
    assert sha(ROOT/p['snapshot'])==p['source_sha256'], p['page_id']
    for f,h in p['figures_sha256'].items(): assert sha(ROOT/f)==h,f
    for kind in ['source','executed']:
        k='notebook_'+kind
        if k in p: assert sha(ROOT/p[k])==p[k+'_sha256'],p[k]
def read(pid):
    p=PAGES[pid];check(p)
    print((ROOT/p['snapshot']).read_text())
    for f in p['figures_sha256']:
        dest=ART/(Path(f).stem+'.png')
        if not dest.exists():
            subprocess.run(['inkscape',str(ROOT/f),'--export-type=png','--export-width=960','--export-filename='+str(dest)],check=True,capture_output=True)
        print('FIGURE_ARTIFACT',dest)
    if 'notebook_executed' in p:
        nb=json.loads((ROOT/p['notebook_executed']).read_text())
        print('\nEXECUTED NOTEBOOK',p['notebook_executed'])
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            print('\nCODE CELL',i,'execution_count',c.get('execution_count'))
            code=''.join(c['source'])
            if pid!='1.1' and code.startswith('# @title 準備本節的工具'):
                first=json.loads((ROOT/PAGES['1.1']['notebook_executed']).read_text())
                original=next(''.join(x['source']) for x in first['cells'] if x['cell_type']=='code')
                if code==original:print('Exact same preparation code already read in 1.1; SHA256',hashlib.sha256(code.encode()).hexdigest())
                else:print(code)
            else:print(code)
            for j,o in enumerate(c.get('outputs',[])):
                print('OUTPUT',j,o.get('output_type'))
                if 'text' in o: print(''.join(o['text']))
                for mime,value in o.get('data',{}).items():
                    if mime.startswith('image/'):
                        ext={'image/png':'png','image/jpeg':'jpg','image/svg+xml':'svg'}.get(mime)
                        if ext:
                            out=ART/f'{pid}-cell{i}-output{j}.{ext}'
                            data=''.join(value) if isinstance(value,list) else value
                            out.write_bytes(data.encode() if ext=='svg' else base64.b64decode(data))
                            print('NOTEBOOK_IMAGE_ARTIFACT',out)
                            if ext=='svg':
                                matches=[f for f,h in p['figures_sha256'].items() if sha(out)==h]
                                print('NOTEBOOK_SVG_MATCHES_SOURCE_SHA',matches)
                                if not matches:
                                    png=out.with_suffix('.png')
                                    subprocess.run(['inkscape',str(out),'--export-type=png','--export-width=960','--export-filename='+str(png)],check=True,capture_output=True)
                                    print('NOTEBOOK_IMAGE_RENDER',png)
                    else:print(mime, ''.join(value) if isinstance(value,list) else value)
                if o.get('output_type')=='error': print(o)
def record(payload):
    pid=payload['page_id'];p=PAGES[pid];check(p)
    report=BASE/'reports/a.json';trace=BASE/'traces/a.jsonl'
    data=json.loads(report.read_text()) if report.exists() else {'schema_version':1,'estimate_kind':'ai_estimate','pages':[]}
    assert pid==ASSIGN['page_ids'][len(data['pages'])],(pid,len(data['pages']))
    row={k:payload[k] for k in ['page_id','minutes_min','minutes_max','reason']}
    assert isinstance(row['minutes_min'],int) and 0<row['minutes_min']<=row['minutes_max']
    row.update(reviewer_task='/root/p6_time_a',source_sha256=p['source_sha256'],figures_sha256=p['figures_sha256'])
    data['pages'].append(row)
    report.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    t={**row,'read_sequence':len(data['pages']),'recorded_at':datetime.now(timezone.utc).isoformat(),'snapshot':p['snapshot'],'summary':payload['summary'],'viewed_artifacts':payload.get('viewed_artifacts',[]),'notebook_output_observation':payload.get('notebook_output_observation',''),'notebooks':{k:p[k] for k in p if k.startswith('notebook_')},'notebook_image_artifacts':{str(f.relative_to(ROOT)):sha(f) for f in ART.glob(pid+'-cell*-output*') if f.is_file()}}
    for v in t['viewed_artifacts']:
        if isinstance(v,dict) and 'artifact' in v:v['artifact_sha256']=sha(ROOT/v['artifact'])
    with trace.open('a') as f:f.write(json.dumps(t,ensure_ascii=False)+'\n')
    print('recorded',pid,row['minutes_min'],row['minutes_max'])
if __name__=='__main__':
    if sys.argv[1]=='read':read(sys.argv[2])
    elif sys.argv[1]=='record':record(json.loads(sys.argv[2]))
