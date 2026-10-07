import hashlib,json,sys
from pathlib import Path
from datetime import datetime,UTC
spec=json.loads(Path(sys.argv[1]).read_text());m=json.load(open('docs/course-revision-20261007-phase7/reviews/freeze-03/manifest.json'));sha=next(p['source_sha256'] for p in m['pages'] if p['page_id']==spec['page_id'])
for item in spec['artifacts']:assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
r={'tool':'view_image','reviewer_task':'/root/p7_technical_b','received_at':datetime.now(UTC).isoformat(),'source_sha256':sha,'artifacts':spec['artifacts'],'observation':spec['observation']}
p=Path('docs/technical-reviews/artifacts/p7_technical_b')/f"page-view-{spec['page_id']}.json";assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':spec['status'],'required':spec['required'],'source_sha256':sha,'details':spec['details'],'observation':spec['observation'],'receipt':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'artifacts':spec['artifacts']},ensure_ascii=False))

