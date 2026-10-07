"""Save owner-supplied observation only after actual view_image returns; hashes metadata."""
import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).parent
j=json.load(sys.stdin);name=j.pop('name');j.update(tool='tools.view_image',reviewer_task='/root/p7_technical_a',received_at=datetime.now(timezone.utc).isoformat())
for a in j['artifacts']:a['sha256']=hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()
p=BASE/name
if p.exists():raise SystemExit('refuse overwrite')
p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');print(p.relative_to(ROOT));print(hashlib.sha256(p.read_bytes()).hexdigest())
