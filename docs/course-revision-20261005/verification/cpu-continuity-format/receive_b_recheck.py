import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

root = Path.cwd()
base = root / 'docs/course-revision-20261005/continuity'
group = '06_09'
progress_path = base / 'progress.json'
progress = json.loads(progress_path.read_text())
assignment = progress['assignments'][group]
report_path = base / 'reports/06_09-recheck-01.json'
trace_path = base / 'traces/06_09-recheck-01.jsonl'
report = json.loads(report_path.read_text())
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert report['overall_verdict'] == 'pass_with_retained_and_new_optionals'
assert sha(base / 'reports/06_09.json') == assignment['initial_report_sha256']
assert sha(base / 'traces/06_09.jsonl') == assignment['initial_trace_sha256']
current = {p['page_id']:p for p in json.loads((base / 'revised-02/inventory.json').read_text())['pages']}
original = {p['page_id']:p for p in json.loads((base / 'reports/06_09.json').read_text())['primary_pages']}
changed = {p['page_id']:p for p in report['changed_primary_pages']}
context = {p['page_id']:p for p in report['necessary_previous_next_and_after_context']}
assert len(changed) == 25
rows = []
for page_id in assignment['primary_page_ids']:
    if page_id in changed:
        row, basis = changed[page_id], 'actual_same_owner_changed_primary_recheck'
    elif page_id in context:
        row, basis = context[page_id], 'actual_same_owner_unchanged_primary_necessary_context_reread'
    else:
        row, basis = original[page_id], 'unchanged_true_initial_pass_inherited'
    live = current[page_id]
    assert row['source_sha256'] == live['source_sha256'] and row['figures_sha256'] == live['figures_sha256'], page_id
    assert row['verdict'] == 'pass', page_id
    assert sha(root / row.get('snapshot', live['snapshot'])) == row['source_sha256']
    for path, digest in row['figures_sha256'].items():
        assert sha(root / path) == digest
    rows.append({'page_id':page_id,'source_sha256':row['source_sha256'],'figures_sha256':row['figures_sha256'],'basis':basis})
for row in report['necessary_previous_next_and_after_context']:
    live = current[row['page_id']]
    assert row['source_sha256'] == live['source_sha256'] and row['figures_sha256'] == live['figures_sha256']
    assert row['verdict'] == 'pass'
for view in report['pageviews']:
    assert sha(root / view['screenshot_path']) == view['screenshot_sha256']
assert sha(trace_path) == report['trace_sha256']
receipt = {'recorded_at':datetime.now(UTC).isoformat(),'group':group,
    'report_path':str(report_path.relative_to(root)),'report_sha256':sha(report_path),
    'trace_path':str(trace_path.relative_to(root)),'trace_sha256':sha(trace_path),
    'initial_report_unchanged':True,'initial_trace_unchanged':True,
    'primary_current_versions':len(rows),'actual_changed_primary_reread':25,
    'unchanged_primary_actual_context_reread':17,'unchanged_primary_initial_pass_inherited':17,'primary_versions':rows,
    'root_schema_assumptions_corrected':[
        'First helper expected overall pass; actual report transparently says pass_with_retained_and_new_optionals.',
        'Second helper incorrectly inherited every unchanged row; actual17 unchanged primary rows were reread as necessary context. 7.12 initial revise due7.11 navigation is resolved by genuine7.12 reread. No reviewer report changed.'
    ],
    'scope':'Root version/screenshot metadata receipt only. Actual25 changed+17 unchanged necessary primary rereads and17 inherited genuine unchanged first passes; no full59 new read claimed; does not prove pedagogical/scientific judgment.'}
destination = base / 'receipts/06_09-recheck-01.json'
assert not destination.exists()
destination.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
assignment.update(status='current_continuity_pass',current_report_path=receipt['report_path'],current_report_sha256=receipt['report_sha256'],current_receipt_path=str(destination.relative_to(root)))
progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+'\n')
triage_path = base / 'triage.json'
triage = json.loads(triage_path.read_text())
triage['deferred_optional'].append({'issue':'10.3 scene() identity after explicit10.2','actual_reader':'/root/continuity_06_09',
    'reason':'Root read10.3 and actual scene default red/square. Same actual image, this step tests48→8 feature shapes without claiming colour understanding; actual reader independently reports nonblocking. Preserve original question and separate post-reading author response, avoid gratuitous code/version change.'})
triage_path.write_text(json.dumps(triage,ensure_ascii=False,indent=2)+'\n')
print('Actual59 current primary/31 screenshot hashes:25 changed reread +17 unchanged context reread +17 unchanged initial PASS.')
