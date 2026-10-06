import json,sys,datetime,hashlib,pathlib,re
ROOT=pathlib.Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/course-revision-20261005/continuity'
PAGES={p['page_id']:p for p in json.loads((BASE/'revised-04/inventory.json').read_text())['pages']}
TRACE=BASE/'traces/13_16-recheck-03.jsonl'
def selected(page,unit=None,sub=None):
 source=(ROOT/PAGES[page]['snapshot']).read_text()
 if unit is not None:source=re.split(r'(?=^## )',source,flags=re.M)[int(unit)]
 if sub is not None:source=re.split(r'(?=^### )',source,flags=re.M)[int(sub)]
 return source
def read(page,unit=None,sub=None):
 p=PAGES[page]; s=selected(page,unit,sub)
 print(json.dumps({k:p[k] for k in ['page_id','title','source','selector','source_sha256','figures_sha256']},ensure_ascii=False))
 if unit is not None:print('unit_sha256',hashlib.sha256(s.encode()).hexdigest())
 print(s)
def append(d):
 d['recorded_at']=datetime.datetime.now(datetime.timezone.utc).isoformat(); p=PAGES[d['page_id']]
 d.update(source_snapshot=p['snapshot'],source_sha256=p['source_sha256'],figures_sha256=p['figures_sha256'])
 s=selected(d['page_id'],d.get('unit_index'),d.get('subunit_index'))
 if 'unit_index' in d:d['unit_sha256']=hashlib.sha256(s.encode()).hexdigest()
 d['original_text']=s
 with TRACE.open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
if __name__=='__main__':
 if sys.argv[1]=='read':read(*sys.argv[2:])
 elif sys.argv[1]=='append':append(json.load(sys.stdin))
