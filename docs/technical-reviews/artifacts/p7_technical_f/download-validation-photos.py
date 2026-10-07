import json,urllib.request,hashlib,concurrent.futures
from pathlib import Path
m=json.loads(Path('docs/natural-assistant/v4/manifest.json').read_text());rows=[r for r in m['rows'] if r['split']=='validation' and r['id'].endswith('/scene')]
out=Path('docs/technical-reviews/artifacts/p7_technical_f/photos');out.mkdir(exist_ok=True)
def get(r):
 name=r['source']['original_id']; generation=r['source']['image_generation'];url=f'https://storage.googleapis.com/docci/thumbnails/{name}.jpg?generation={generation}';p=out/(name+'.jpg');data=urllib.request.urlopen(url,timeout=40).read();p.write_bytes(data);h=hashlib.sha256(data).hexdigest();assert h==r['source']['image_sha256'];return {'id':name,'url':url,'path':str(p),'sha256':h,'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:receipt=list(e.map(get,rows))
Path('docs/technical-reviews/artifacts/p7_technical_f/validation-photo-downloads.json').write_text(json.dumps(receipt,indent=2)+'\n');print('downloaded and source SHA matched',len(receipt))
