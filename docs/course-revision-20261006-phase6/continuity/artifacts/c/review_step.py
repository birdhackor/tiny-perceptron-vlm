from pathlib import Path
import datetime
import hashlib
import json
import re
import sys

BASE = Path('/workspace/selftrained-v2/docs/course-revision-20261006-phase6/continuity')
ART = BASE / 'artifacts/c'

def units(page):
    raw = (BASE / 'source-pages' / f'{page}.md').read_text()
    lines = raw.splitlines(keepends=True)
    starts = [0] + [i for i, line in enumerate(lines) if i and re.match(r'^#{2,6} ', line)]
    starts += [len(lines)]
    return raw, [(a + 1, z, ''.join(lines[a:z])) for a, z in zip(starts, starts[1:])]

if sys.argv[1] == 'read':
    page, index = sys.argv[2], int(sys.argv[3])
    raw, parts = units(page)
    a, z, content = parts[index]
    print(f'PAGE {page} UNIT {index}/{len(parts)-1} LINES {a}-{z}')
    print(content)
elif sys.argv[1] == 'append':
    record = json.load(sys.stdin)
    page = record['page_id']
    data = (BASE / 'source-pages' / f'{page}.md').read_bytes()
    record['source_sha256'] = hashlib.sha256(data).hexdigest()
    record['recorded_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record.setdefault('phase', 'first-read')
    if isinstance(record.get('unit'), int):
        raw, parts = units(page)
        a, z, content = parts[record['unit']]
        record['line_range'] = [a, z]
        record['unit_sha256'] = hashlib.sha256(content.encode()).hexdigest()
    assert len(record['five_points']) == 5
    with (BASE / 'traces/c.jsonl').open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    (ART / 'source-pages' / f'{page}.md').write_bytes(data)
    print('APPENDED', page, record['unit'])
