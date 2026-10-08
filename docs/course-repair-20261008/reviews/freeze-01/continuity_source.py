"""Source delivery helper, not proof of comprehension or a blind-reading test."""
from pathlib import Path
from datetime import datetime, timezone
import json, re, hashlib, argparse

B=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text())
P={p['page_id']:p for p in M['inventory']['pages']}
ap=argparse.ArgumentParser()
ap.add_argument('group',choices=list(M['groups']))
ap.add_argument('phase',choices=['main','extras'])
ap.add_argument('page_ids',nargs='+')
a=ap.parse_args()
folder=B/'work'/f'continuity-{a.group}'
folder.mkdir(parents=True,exist_ok=True)
log=folder/'source-delivery.jsonl'
old=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if a.phase=='extras':
    delivered={r.get('page_id') for r in old if r['phase']=='main'}
    missing=set(M['groups'][a.group]['pages'])-delivered
    notes=B/'reports'/f'continuity-{a.group}-main-notes.json'
    if missing or not notes.exists():
        raise SystemExit('Before extras: finish all group main sources and save your actual main notes; missing='+str(sorted(missing)))
    seal=folder/'main-notes-seal.json'
    if not seal.exists():
        seal.write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'notes_path':str(notes),'sha256':sha(notes),'meaning':'notes byte seal before helper delivers supplements; source delivery alone is not reading evidence'},ensure_ascii=False,indent=2)+'\n')
    elif json.loads(seal.read_text())['sha256']!=sha(notes):
        raise SystemExit('Main notes changed after supplement delivery')
for pid in a.page_ids:
    p=P[pid];raw=Path(p['snapshot']).read_text()
    if sha(Path(p['snapshot']))!=p['source_sha256']: raise SystemExit('Frozen drift '+pid)
    details=list(re.finditer(r'<details\b[^>]*>.*?</details>',raw,re.S))
    if a.phase=='main':
        def hide(m):
            label=re.search(r'<summary[^>]*>(.*?)</summary>',m[0],re.S)
            return '\n[原位置選讀折疊區：'+(re.sub('<[^>]+>','',label[1]).strip() if label else '選讀')+'；完成全組正文並保存main notes後補讀]\n'
        shown=re.sub(r'<details\b[^>]*>.*?</details>',hide,raw,flags=re.S)
    else:
        shown='\n\n'.join(f'選讀{n+1}\n'+m[0] for n,m in enumerate(details)) or '[本頁沒有折疊區]'
    print(json.dumps({'page_id':pid,'phase':a.phase,'source_sha256':p['source_sha256'],'characters_delivered':len(shown),'scope':'delivery, not automatic pass'},ensure_ascii=False))
    print(shown)
    with log.open('a') as f:
        f.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'phase':a.phase,'page_id':pid,'source_sha256':p['source_sha256'],'meaning':'source delivered; actual understanding must be recorded by reviewer'},ensure_ascii=False)+'\n')
