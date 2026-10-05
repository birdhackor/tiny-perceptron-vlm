import json,sys,datetime,hashlib,pathlib,re
ROOT=pathlib.Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/course-revision-20261005/continuity'
PAGES={p['page_id']:p for p in json.loads((BASE/'inventory.json').read_text())['pages']}
TRACE=BASE/'traces/13_16.jsonl'
def read(page):
 p=PAGES[page]; print(json.dumps({k:p[k] for k in ['page_id','title','source','selector','source_sha256','figures_sha256']},ensure_ascii=False)); print((ROOT/p['snapshot']).read_text())
def append(d):
 d['recorded_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 if 'page_id' in d:
  p=PAGES[d['page_id']]; d.update(source_snapshot=p['snapshot'],source_sha256=p['source_sha256'],figures_sha256=p['figures_sha256'])
  source=(ROOT/p['snapshot']).read_text()
  if 'unit_index' in d:
   source=re.split(r'(?=^## )',source,flags=re.M)[d['unit_index']]
   if 'reference_subunit' in d:
    parts=re.split(r'(?=^### )',source,flags=re.M);source=parts[d['reference_subunit']]
   if d.get('unit_subheading')=='main':source=source.split('### 閱讀與實作路線')[0]
   elif d.get('unit_subheading')=='reading-route':source='### 閱讀與實作路線'+source.split('### 閱讀與實作路線',1)[1]
   d['unit_sha256']=hashlib.sha256(source.encode()).hexdigest()
  d['original_text']=source
 with TRACE.open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
if __name__=='__main__':
 if sys.argv[1]=='read':read(sys.argv[2])
 elif sys.argv[1]=='append':append(json.load(sys.stdin))
 elif sys.argv[1]=='unit':
  p=PAGES[sys.argv[2]]; i=int(sys.argv[3]);s=re.split(r'(?=^## )',(ROOT/p['snapshot']).read_text(),flags=re.M)[i];print('UNIT',i,'SHA',hashlib.sha256(s.encode()).hexdigest());print(s)
