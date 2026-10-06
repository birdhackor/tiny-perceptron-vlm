from pathlib import Path
import json, re, hashlib, sys
from datetime import datetime, timezone

base = Path('/workspace/selftrained-v2/docs/course-revision-20261006-phase6/continuity')
art = base / 'artifacts/e'
state_path = art / 'reading-state.json'
state = json.loads(state_path.read_text())
page_id = state['pages'][state['position']]
path = base / ('external-source-pages/docs/selftrained/TRAINING.md' if page_id.startswith('external:') else f'source-pages/{page_id}.md')
content = path.read_text()
sections = [s for s in re.split(r'(?=^#{1,3} )', content, flags=re.M) if s.strip()]
idx = state['section']
piece = sections[idx]
data = json.loads(sys.argv[1])
required = ['known_context', 'next_problem_and_prerequisites', 'changes_and_signal', 'open_question_and_references', 'artifact_status']
assert all(k in data for k in required)
start = content.index(piece)
record = {'sequence': state.get('sequence', 0)+1, 'recorded_at': datetime.now(timezone.utc).isoformat(), 'page_id': page_id, 'section_index': idx, 'section_heading': piece.splitlines()[0], 'source_snapshot': str(path.relative_to(base.parent.parent.parent)), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'section_sha256': hashlib.sha256(piece.encode()).hexdigest(), 'lines': [content[:start].count('\n')+1, content[:start+len(piece)].count('\n')], **data}
with (base/'traces/e.jsonl').open('a') as f: f.write(json.dumps(record, ensure_ascii=False)+'\n')
state['sequence'] = record['sequence']
if idx+1 < len(sections): state['section'] += 1
else: state['position'] += 1; state['section'] = 0
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
print('TRACE APPENDED', record['sequence'], page_id, piece.splitlines()[0])
if state['position'] >= len(state['pages']): print('ASSIGNED READING COMPLETE'); sys.exit(0)
new_id = state['pages'][state['position']]
new_path = base / ('external-source-pages/docs/selftrained/TRAINING.md' if new_id.startswith('external:') else f'source-pages/{new_id}.md')
raw = new_path.read_bytes()
if state['section'] == 0: (art / ('external-TRAINING.raw.md' if new_id.startswith('external:') else new_id+'.raw.md')).write_bytes(raw)
new_sections = [s for s in re.split(r'(?=^#{1,3} )', raw.decode(), flags=re.M) if s.strip()]
print('READING', new_id, 'SECTION', state['section'])
print(new_sections[state['section']])
