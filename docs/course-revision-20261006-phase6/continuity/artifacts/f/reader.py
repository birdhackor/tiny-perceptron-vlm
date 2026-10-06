import json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[5]
base=root/'docs/course-revision-20261006-phase6/continuity'
out=base/'artifacts/f'
manifest=json.loads((out/'manifest.json').read_text())
units=[(m,u) for m in manifest for u in m['units']]
trace=base/'traces/f.jsonl'
trace.parent.mkdir(parents=True,exist_ok=True)
rows=[json.loads(s) for s in trace.read_text().splitlines()] if trace.exists() else []
i=len(rows)
if i>=len(units):raise SystemExit('All assigned units already recorded.')
payload=json.load(sys.stdin)
assert len(payload['five'])==5
m,u=units[i]
assert payload.pop('ref')==m['ref']
assert payload.pop('unit_index')==u['unit_index']
record={'event':'section-understanding','sequence':i+1,'reviewer_task':'/root/p6_continuity_f','recorded_at':datetime.now(timezone.utc).isoformat(),'ref':m['ref'],'kind':m['kind'],'source':m['source'],'raw_sha256':m['raw_sha256'],'scope_sha256':m['scope_sha256'],'unit':u,'page_complete':u==m['units'][-1],**payload}
with trace.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
print('APPENDED',record['sequence'],m['ref'],u['unit_index'])
if i+1<len(units):
 nm,nu=units[i+1];print('NEXT',nm['ref'],'unit',nu['unit_index'],nu['title']);print((root/nu['path']).read_text())
else:print('ALL ASSIGNED UNITS RECORDED')
