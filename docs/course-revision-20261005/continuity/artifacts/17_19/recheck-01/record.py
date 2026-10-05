import datetime, hashlib, json, sys
from pathlib import Path
root=Path('/workspace/tiny-perceptron-vlm')
base=root/'docs/course-revision-20261005/continuity'
pages={p['page_id']:p for p in json.loads((base/'revised-01/inventory.json').read_text())['pages']}
data=json.load(sys.stdin);p=pages[data['page_id']]
data.update(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer_task='/root/continuity_17_19',review_phase='recheck-01',source_sha256=p['source_sha256'],figures_sha256=p['figures_sha256'],source_snapshot=p['snapshot'])
assert hashlib.sha256((root/p['snapshot']).read_bytes()).hexdigest()==p['source_sha256']
with (base/'traces/17_19-recheck-01.jsonl').open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
print(data['recorded_at'],data['page_id'],data['unit'],'recorded')
