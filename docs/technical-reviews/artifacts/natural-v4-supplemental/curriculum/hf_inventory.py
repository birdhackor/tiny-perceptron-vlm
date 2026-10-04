"""Read public metadata, never download model weights."""
from pathlib import Path
import concurrent.futures, hashlib, json, urllib.request
ROOT=Path(__file__).resolve().parents[5]; HERE=Path(__file__).resolve().parent
RAW=ROOT/'outputs/natural-v4/factual-research/curriculum'
manifest=json.loads((ROOT/'docs/course-experiments/public-models.json').read_text())
def verify(model):
 prefix=model['files'][0]['path'].rsplit('/',1)[0]
 url=f"https://huggingface.co/api/models/{model['repo']}/tree/{model['revision']}/{prefix}?recursive=true&expand=false&limit=1000"
 row={'id':model['id'],'url':url,'version':model['revision'],'accessed_on':'2026-10-04'}
 try:
  with urllib.request.urlopen(url,timeout=30) as response: raw=response.read();row['status']=response.status
  path=RAW/f"hf-{model['id']}.json";path.write_bytes(raw);row.update(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())
  entries={e['path']:e for e in json.loads(raw)}; checks=[]
  for f in model['files']:
   if not f['path'].endswith('.pt'): continue
   entry=entries.get(f['path']);checks.append({'path':f['path'],'expected_bytes':f['bytes'],'observed_bytes':entry.get('size') if entry else None,'sha_equal':entry.get('lfs',{}).get('oid')==f['sha256'] if entry else False})
  row['weights']=checks;row['all_weights_match']=all(c['expected_bytes']==c['observed_bytes'] and c['sha_equal'] for c in checks)
 except Exception as e:row['error']=type(e).__name__+': '+str(e)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(verify,manifest['models']))
(HERE/'hf-inventory.results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False,indent=2));assert len(rows)==30 and all(r.get('all_weights_match') for r in rows)
