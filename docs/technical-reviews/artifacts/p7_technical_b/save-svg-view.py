import hashlib,json,sys
from pathlib import Path
from datetime import datetime,UTC
from PIL import Image
spec=json.loads(Path(sys.argv[1]).read_text());artifacts=[]
for width in [640,360]:
 p=Path(spec['render_directory'])/f'{width}.png';w,h=Image.open(p).size
 artifacts.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'width':w,'height':h})
r={'tool':'view_image','reviewer_task':'/root/p7_technical_b','received_at':datetime.now(UTC).isoformat(),'source_sha256':spec['source_sha256'],'artifacts':artifacts,'observation':spec['observation']}
p=Path('docs/technical-reviews/artifacts/p7_technical_b')/f"view-{spec['page_id']}.json";assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figure':spec['figure'],'source_sha256':spec['source_sha256'],'status':spec['status'],'details':spec['details'],'observation':spec['observation'],'receipt':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'artifacts':artifacts},ensure_ascii=False))

