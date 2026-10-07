"""Persist only a supplied observation after this owner actually viewed images."""
import datetime
import hashlib
import json
import sys
from pathlib import Path

row=json.load(sys.stdin)
row.update(tool='tools.view_image', reviewer_task='/root/p7_technical_c', received_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
if 'page_id' in row:
    manifest=json.loads(Path('docs/course-revision-20261007-phase7/reviews/freeze-06/manifest.json').read_text())
    row['source_sha256']=next(p['source_sha256'] for p in manifest['pages'] if p['page_id']==row['page_id'])
for a in row['artifacts']:
    a['sha256']=hashlib.sha256(Path(a['path']).read_bytes()).hexdigest()
path=Path('docs/technical-reviews/artifacts/p7_technical_c')/f'{sys.argv[1]}-view.json'
if path.exists():raise SystemExit(f'Preserve prior receipt: {path}')
path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
