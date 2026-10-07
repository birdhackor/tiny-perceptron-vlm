import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
spec=json.loads(Path(sys.argv[1]).read_text())
manifest=json.loads((ROOT/'docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json').read_text())
meta=next(p for p in manifest['pages'] if p['page_id']==spec['page_id'])
for k in ['source_sha256','figures_sha256','notebook','notebook_sha256']:
 if k in meta:spec[k]=meta[k]
for collection in ['artifacts','sources']:
 for item in spec.get(collection,[]):
  if 'path' in item:item['sha256']=hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()
spec.setdefault('trace_file','docs/course-revision-20261007-phase7/reviews/freeze-03/traces/technical/b/e33ab62ea31d48e6ba034bcd77d0058f.jsonl')
spec.setdefault('question_refs',[])
out=BASE/'pages'/f"{spec['page_id']}.json";out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():raise SystemExit('Refusing to overwrite an existing page record')
out.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))

