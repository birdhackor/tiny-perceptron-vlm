from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,platform,time
root=Path.cwd();out=root/'outputs/natural-r4-public-checkpoints';out.mkdir(parents=True,exist_ok=True)
items=[];counts={}
for name in ['capstone-public','public-models']:
 d=json.loads((root/'docs/course-experiments'/f'{name}.json').read_text())
 pts=[(m,f) for m in d['models'] for f in m['files'] if f['path'].endswith('.pt')]
 counts[name]={'groups':len(d['models']),'checkpoint_files':len(pts)}
 for model,file in pts:items.append((name,model['id'],model.get('repo',d.get('repo')),model.get('revision',d.get('revision')),file))
def verify(item):
 manifest,model,repo,rev,f=item
 url=f'https://huggingface.co/{repo}/resolve/{rev}/{f["path"]}'
 record={'manifest':manifest,'model':model,'repo':repo,'revision':rev,'path':f['path'],'url':url,'expected_bytes':f['bytes'],'expected_sha256':f['sha256']}
 try:
  with urlopen(Request(url,headers={'User-Agent':'TechnicalReview/1.0'}),timeout=45) as r:data=r.read();status=r.status
  digest=hashlib.sha256(data).hexdigest();record.update(http_status=status,bytes=len(data),sha256=digest,verified=len(data)==f['bytes'] and digest==f['sha256'])
  if record['verified']:
   dest=out/manifest/model/f['output'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 except Exception as e:record.update(verified=False,error=str(e))
 return record
start=time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(verify,items))
r={'command':'python docs/technical-reviews/artifacts/natural-r4-check-public.py','environment':{'python':platform.python_version(),'network':'anonymous HTTPS; no Authorization header or token; TLS verification retained'},'counts':counts,'seconds':time.perf_counter()-start,'checked_checkpoint_count':len(records),'verified_count':sum(r['verified'] for r in records),'records':records}
Path('docs/technical-reviews/artifacts/natural-r4-public-downloads.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print({k:r[k] for k in ['counts','seconds','checked_checkpoint_count','verified_count']})
for record in records:
 if not record['verified']:print(record)
assert len(records)==131 and all(r['verified'] for r in records)
