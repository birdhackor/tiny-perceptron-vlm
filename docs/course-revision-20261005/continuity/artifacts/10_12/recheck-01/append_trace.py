import datetime, hashlib, json, pathlib, sys, urllib.request
root=pathlib.Path('/workspace/tiny-perceptron-vlm')
base=root/'docs/course-revision-20261005/continuity'
inventory={p['page_id']:p for p in json.loads((base/'revised-01/inventory.json').read_text())['pages']}
entry=json.load(sys.stdin)
page=inventory[entry['page_id']]
assert hashlib.sha256((root/page['snapshot']).read_bytes()).hexdigest()==page['source_sha256']
for path, expected in page['figures_sha256'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected
    url='http://127.0.0.1:8765/figures/'+pathlib.Path(path).name
    assert hashlib.sha256(urllib.request.urlopen(url,timeout=10).read()).hexdigest()==expected
entry.update(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer_task='/root/continuity_10_12',round='recheck-01',source_sha256=page['source_sha256'],figures_sha256=page['figures_sha256'],snapshot=page['snapshot'])
with (base/'traces/10_12-recheck-01.jsonl').open('a') as f:
    f.write(json.dumps(entry,ensure_ascii=False)+'\n')
print(entry['timestamp_utc'],entry['page_id'])
