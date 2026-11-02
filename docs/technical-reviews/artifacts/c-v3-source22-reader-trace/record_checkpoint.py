"""Record manually supplied reread observations, then let native reader open next unit."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
session = sys.argv[1]
payload = json.load(sys.stdin)
state_path = ROOT / 'docs/course-revision-20261016/reader-sessions' / session / 'state.json'
state = json.loads(state_path.read_text())
assert payload['unit_index'] == state['next_index']
assert state['reviewer_task'] == '/root/c_v3_reader_entries'
assert state['reading_plan']['path'] == 'docs/course-revision-20261101-c-v3/reader-source22-trace-reading-plan-v1.json'
for c in payload['concept_checks']:
    for key in ('concept','identity','mechanism','need','usage','example_scope','source_quote','verdict'):
        assert isinstance(c[key], str) and c[key].strip(), key
payload['reading_kind'] = 'same_reader_current_source_reread_not_blind'
payload['prior_context'] = '同原讀者：已真讀source19七canonical/三附頁，之後source21有界修後、11折疊及指定視覺；保留全部舊判，不將本次重讀算新blind首讀。'
payload['collection_role'] = 'technical'
payload['wave'] = 'c-v3-integration-07'
payload['execution_and_visual_limits'] = '本次只重讀當前unit，不執行課程/模型/原驗收或重拍圖；既有同SHA視覺另有舊證據，未重驗設備或全部頁面。'
payload.setdefault('issues', [])
target = HERE / 'checkpoints' / session / f"unit-{payload['unit_index']:03d}.json"
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x') as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
subprocess.run([sys.executable, str(ROOT/'docs/course-revision-20261016/tools/reader.py'),
                'next', session, '--checkpoint', str(target)], check=True, cwd=ROOT)
