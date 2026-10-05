import sys,json,re,hashlib,html
from urllib.request import urlopen
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/course-revision-20261005/continuity'
INV={x['page_id']:x for x in json.loads((BASE/'inventory.json').read_text())['pages']}
def units(page):
    text=(ROOT/INV[page]['snapshot']).read_text()
    return re.split(r'(?=^## )',text,flags=re.M)
if sys.argv[1] in ('read','outputs'):
    p=sys.argv[2]; n=int(sys.argv[3]) if len(sys.argv)>3 else 0
    u=units(p) if len(sys.argv)>3 else [(ROOT/INV[p]['snapshot']).read_text()]
    print(f'PAGE {p}; UNIT {n+1}/{len(u)}; source_sha256 {INV[p]["source_sha256"]}')
    if sys.argv[1]=='read':print(u[n])
    if len(sys.argv)==3:
        url=f'http://127.0.0.1:8765/{p}.html'
        rendered=urlopen(url,timeout=15).read()
        dest=BASE/f'artifacts/01_05/{p}.html'
        dest.write_bytes(rendered)
        outputs=[html.unescape(re.sub(r'<[^>]*>','',x)) for x in re.findall(r'<pre[\s\S]*?</pre>',rendered.decode()) if '<code' not in x]
        print('ACTUAL WEBSITE OUTPUTS:', '\n'.join(outputs) or '(none)')
elif sys.argv[1]=='record':
    data=json.load(sys.stdin); p=data['page_id']; meta=INV[p]
    data['recorded_at_utc']=datetime.now(timezone.utc).isoformat()
    data['source_sha256']=meta['source_sha256']; data['figures_sha256']=meta['figures_sha256']
    data['source_snapshot']=meta['snapshot']
    data['source_sha_verified']=hashlib.sha256((ROOT/meta['snapshot']).read_bytes()).hexdigest()==meta['source_sha256']
    if data.get('visual_verified'):
        pv=BASE/'artifacts/01_05/pageviews.jsonl'
        rows=[json.loads(x) for x in pv.read_text().splitlines()]
        for r in rows:
            if r['page_id']==p:r.update(viewed=True,viewed_at_utc=data['recorded_at_utc'])
        pv.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (BASE/'traces').mkdir(parents=True,exist_ok=True)
    with (BASE/'traces/01_05.jsonl').open('a') as f: f.write(json.dumps(data,ensure_ascii=False)+'\n')
    print(data['recorded_at_utc'],p,data['section'])
