from pathlib import Path
import json,hashlib,shutil
repo=Path('/workspace/tiny-perceptron-vlm');b=Path('/workspace/work/tutorial-audit-20261008')
idx=json.loads((repo/'course/lesson-index.json').read_text())
paths={}
for p in (repo/'outputs').rglob('*.ipynb'):
 if p.stem in {x['id'] for x in idx}:paths.setdefault(p.stem,[]).append(p)
def equivalent(original,found):
 if len(original['cells'])!=len(found.get('cells',[])):return False
 for a,c in zip(original['cells'],found['cells']):
  if a['cell_type']!=c['cell_type'] or a['source']!=c['source']:return False
  if c['cell_type']=='code' and (c.get('execution_count') is None or any(x.get('output_type')=='error' for x in c.get('outputs',[]))):return False
 return True
records=[];missing=[]
for item in idx:
 original=json.loads((repo/item['notebook']).read_text())
 found=None
 for p in paths.get(item['id'],[]):
  try:obj=json.loads(p.read_text())
  except (OSError,json.JSONDecodeError):continue
  if equivalent(original,obj):found=p;break
 if found:
  out=b/'executed'/Path(item['notebook']).relative_to('notebooks');out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(found,out)
  records.append({'id':item['id'],'source':str(found),'sha256':hashlib.sha256(found.read_bytes()).hexdigest()})
 else:missing.append(item['id'])
(b/'checks/executed-notebook-matches.json').write_text(json.dumps({'matched':len(records),'missing':missing,'records':records,'scope':'Exact authored cells matched to existing execution artifacts; this is provenance reuse, not a new kernel run.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'matched':len(records),'missing':missing},ensure_ascii=False))
