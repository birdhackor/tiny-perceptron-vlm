from pathlib import Path
import json, re, hashlib, sys
from datetime import datetime, timezone

base = Path('/workspace/selftrained-v2/docs/course-revision-20261006-phase6/continuity')
art = base / 'artifacts/e/recheck-01'
state_path = art / 'reading-state.json'
state = json.loads(state_path.read_text())
item = state['queue'][state['index']]
path = Path('/workspace/selftrained-v2') / item['path']
content = path.read_text()
pieces = [s for s in re.split(r'(?=^#{1,3} )', content, flags=re.M) if s.strip()]
piece = pieces[item['section_index']]
data = json.loads(sys.argv[1])
fields = ['known_context', 'next_problem_and_prerequisites', 'changes_and_signal', 'open_question_and_references', 'artifact_status']
assert all(k in data for k in fields)
start = content.index(piece)
record = {'sequence': state['sequence'] + 1, 'recorded_at': datetime.now(timezone.utc).isoformat(), 'page_id': item['page_id'], 'section_index': item['section_index'], 'section_heading': piece.splitlines()[0], 'source_snapshot': item['path'], 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'section_sha256': hashlib.sha256(piece.encode()).hexdigest(), 'lines': [content[:start].count('\n') + 1, content[:start + len(piece)].count('\n')], **data}
with (base/'traces/e-recheck-01.jsonl').open('a') as f: f.write(json.dumps(record, ensure_ascii=False)+'\n')
state['sequence'] = record['sequence']
state['index'] += 1
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
print('TRACE APPENDED', record['sequence'], item['page_id'], piece.splitlines()[0])
if state['index'] >= len(state['queue']): print('RECHECK READING COMPLETE'); sys.exit(0)
next_item = state['queue'][state['index']]
next_path = Path('/workspace/selftrained-v2') / next_item['path']
next_pieces = [s for s in re.split(r'(?=^#{1,3} )', next_path.read_text(), flags=re.M) if s.strip()]
print('READING', next_item['page_id'], 'SECTION', next_item['section_index'])
print(next_pieces[next_item['section_index']])
