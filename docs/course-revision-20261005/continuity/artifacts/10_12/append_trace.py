import datetime, hashlib, json, pathlib, sys
root = pathlib.Path('/workspace/tiny-perceptron-vlm')
base = root / 'docs/course-revision-20261005/continuity'
pages = {p['page_id']: p for p in json.loads((base/'inventory.json').read_text())['pages']}
entry = json.load(sys.stdin)
page = pages[entry['page_id']]
assert hashlib.sha256((root/page['snapshot']).read_bytes()).hexdigest() == page['source_sha256']
entry.update(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reviewer_task='/root/continuity_10_12', source_sha256=page['source_sha256'], figures_sha256=page['figures_sha256'])
with (base/'traces/10_12.jsonl').open('a') as f:
    f.write(json.dumps(entry, ensure_ascii=False)+'\n')
print(entry['timestamp_utc'],entry['page_id'],entry.get('unit','full_page'))
