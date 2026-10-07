import json,urllib.request,hashlib,concurrent.futures
from pathlib import Path
base=Path('docs/technical-reviews/artifacts/p7_technical_f');m=json.loads(Path('docs/natural-assistant/v4/manifest.json').read_text());photos=[r for r in m['rows'] if r['split']=='validation' and r['id'].endswith('/scene')];orig={}
for line in Path('outputs/p7_technical_f/docci-descriptions-original.jsonl').open():
 x=json.loads(line)
 if x['example_id'] in {r['source']['original_id'] for r in photos}:orig[x['example_id']]=x
out=base/'20-8-photos';out.mkdir(exist_ok=True)
def one(r):
 s=r['source'];id=s['original_id'];url='https://storage.googleapis.com/docci/thumbnails/'+id+'.jpg?generation='+s['image_generation'];p=out/(id+'.jpg')
 if not p.exists():p.write_bytes(urllib.request.urlopen(url,timeout=35).read())
 sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==s['image_sha256'];assert r['references']['official_description']==orig[id]['description'];return {'id':id,'path':str(p),'sha256':sha,'url':url,'caption':orig[id]['description']}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:files=list(e.map(one,photos))
(base/'20-8-photo-originals.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n');print('downloaded and source SHA verified',len(files))
raw=Path('docs/natural-assistant/evidence/v4-runtime/validation-37219466611/blind-review/raw');variants={v:json.loads((raw/('generations-'+v+'.json')).read_text()) for v in ['base','adapter-step-001039','adapter-step-002077']};deck=[]
for p in files:
 cases=[]
 for r in m['rows']:
  if r['split']=='validation' and r.get('image')=='vision/images/'+p['id']+'.jpg':
   cases.append({'id':r['id'],'task':r['task'],'question':r['user'],'reference':r['answer'],'rubric':r['references'].get('rubric','Correct requested visible fact; equivalent wording accepted; unsupported/reversed/invented facts fail.'),'predictions':{v:next(q for q in rows if q['id']==r['id'])['prediction'] for v,rows in variants.items()}})
 deck.append(dict(p,cases=cases))
(base/'20-8-photo-deck.json').write_text(json.dumps(deck,ensure_ascii=False,indent=2)+'\n');print('deck cases',sum(len(p['cases']) for p in deck),'NO peer/status fields exported')
