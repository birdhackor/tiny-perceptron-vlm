from pathlib import Path
from PIL import Image
from collections import defaultdict
import json, hashlib, shutil
ROOT=Path(__file__).resolve().parents[4]
ART=Path(__file__).resolve().parent
DATA=ROOT/'outputs/selftrained-v2/data'
records=[]
for p in DATA.glob('ocr-*.jsonl'):
 records.extend(json.loads(line) for line in p.read_bytes().splitlines())
images=defaultdict(list)
for relative in sorted({r['image'] for r in records}):
 with Image.open(DATA/relative) as image:
  digest=hashlib.sha256(image.tobytes()).hexdigest()
 images[digest].append(relative)
collisions=[]
for digest,files in images.items():
 rows=[r for r in records if r['image'] in files]
 if len({r['split'] for r in rows})>1:
  collision={'decoded_pixels_sha256':digest,'records':[{k:r[k] for k in ['id','split','group_id','image','roi']}|{'target':r['supervision']['ocr_text'],'case':r['supervision']['case'],'condition':r['supervision']['condition'],'render_source_id':r['supervision']['render_source_id'],'image_file_sha256':hashlib.sha256((DATA/r['image']).read_bytes()).hexdigest()} for r in rows]}
  collisions.append(collision)
  for index,relative in enumerate(files):
   target=ART/f'ocr-pixel-collision-{index}.png'
   shutil.copyfile(DATA/relative,target)
   assert target.read_bytes()==(DATA/relative).read_bytes()
assert len(collisions)==1
target=ART/'ocr-pixel-collision.json'
target.write_text(json.dumps(collisions,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'full_canvas_cross_split_collisions':len(collisions),'verified_original_png_copies':2,'result':str(target.relative_to(ROOT))},ensure_ascii=False))
