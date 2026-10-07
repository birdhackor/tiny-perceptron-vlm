from pathlib import Path
import json,hashlib
b=Path('outputs/selftrained/data');d=Path('outputs/p7-technical-e-cache/data');assets=set()
for family in ['ocr','text-tools','vision','voice']:
 for s in ['test','validation']:
  name=family+'-'+s+'.jsonl';assert (b/name).read_bytes()==(d/name).read_bytes()
  for l in (d/name).read_text().splitlines():
   r=json.loads(l)
   for obj in [r,*r['messages']]:
    for k in ['image','audio']:
     if obj.get(k):assets.add(obj[k])
for a in assets:assert (b/a).read_bytes()==(d/a).read_bytes()
print('heldout_record_files',8,'heldout_media_byte_identical',len(assets));print('image_files',sum(x.startswith('images/') for x in assets),'audio_files',sum(x.startswith('audio/') for x in assets))
