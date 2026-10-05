import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

root = Path('/workspace/tiny-perceptron-vlm')
base = root / 'docs/course-revision-20261005/continuity'
progress_path = base / 'progress.json'
progress = json.loads(progress_path.read_text())
current_inventory = json.loads((base / 'revised-02/inventory.json').read_text())
current = {p['page_id']: p for p in current_inventory['pages']}
initial_inventory = json.loads((base / 'inventory.json').read_text())
initial = {p['page_id']: p for p in initial_inventory['pages']}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(row):
    p = current[row['page_id']]
    assert row['source_sha256'] == p['source_sha256'], row['page_id']
    assert row['figures_sha256'] == p['figures_sha256'], row['page_id']
    for path, digest in row['figures_sha256'].items():
        assert sha(root / path) == digest
    snapshot = row.get('source_snapshot', row.get('snapshot'))
    if snapshot:
        assert sha(root / snapshot) == p['source_sha256']
for group in ['01_05', '10_12', '17_19']:
    report_path = base / 'reports' / (group + '-recheck-01.json')
    trace_path = base / 'traces' / (group + '-recheck-01.jsonl')
    report = json.loads(report_path.read_text())
    assert report['decision'] == 'pass'
    assignment = progress['assignments'][group]
    expected = set(assignment['primary_page_ids'])
    assert sha(base / 'reports' / (group + '.json')) == assignment['initial_report_sha256']
    assert sha(base / 'traces' / (group + '.jsonl')) == assignment['initial_trace_sha256']
    if group == '01_05':
        rows = report['page_reviews']
        for row in rows:
            verify(row)
            assert row['decision'] == 'pass'
        primary = [r for r in rows if r['coverage_role'] == 'primary']
        changed_ids = report['changed_primary_page_ids']
        assert all(next(r for r in primary if r['page_id']==p)['actual_recheck_trace_lines'] for p in changed_ids)
        actual_reread = len(report['actual_read_order'])
    elif group == '10_12':
        for row in report['pages']:
            verify(row)
            assert row['decision'] == 'pass'
        reread = set(report['primary_actually_reread'])
        unchanged = {r['page_id'] for r in report['unchanged_primary_not_reread']}
        assert not reread & unchanged and reread | unchanged == expected
        for p in unchanged:
            assert (initial[p]['source_sha256'], initial[p]['figures_sha256']) == (current[p]['source_sha256'], current[p]['figures_sha256'])
        original = json.loads((base / 'reports/10_12.json').read_text())
        initial_rows = {r['page_id']: r for r in original['pages']}
        for p in unchanged:
            assert initial_rows[p]['decision'] == 'pass'
        primary = [{'page_id': p} for p in reread | unchanged]
        changed_ids = report['changed_primary_actually_reread']
        actual_reread = len(report['actual_read_order'])
    else:
        primary = report['primary_pages']
        for row in primary:
            verify(row)
            assert row['decision'] == 'pass'
            if not row['actual_units_reread']:
                p=row['page_id']
                assert (initial[p]['source_sha256'], initial[p]['figures_sha256']) == (current[p]['source_sha256'], current[p]['figures_sha256'])
        for row in report['actual_read_order']:
            verify(row)
        changed_ids = report['changed_primary_page_ids']
        assert all(next(r for r in primary if r['page_id']==p)['actual_units_reread'] for p in changed_ids)
        actual_reread = report['actual_reread_page_count']
    assert {r['page_id'] for r in primary} == expected
    assert len(primary) == len(expected)
    for row in report.get('actual_pageviews', report.get('pageviews', [])):
        path = row['screenshot']
        digest = row.get('screenshot_sha256', row.get('sha256'))
        assert sha(root / path) == digest
    receipt = {'recorded_at': datetime.now(UTC).isoformat(), 'group': group,
        'report_path': str(report_path.relative_to(root)), 'report_sha256': sha(report_path),
        'trace_path': str(trace_path.relative_to(root)), 'trace_sha256': sha(trace_path),
        'initial_report_unchanged': True, 'initial_trace_unchanged': True,
        'primary_current_versions': len(expected), 'actual_changed_primary_reread': len(changed_ids),
        'actual_reread_pages_or_order_entries_as_declared': actual_reread,
        'scope': 'Root metadata/version receipt only; genuine same-reader report and raw trace retained. Does not substitute for firsthand reading or factual verification.'}
    destination = base / 'receipts' / (group + '-recheck-01.json')
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    assignment['status'] = 'current_continuity_pass'
    assignment['current_report_path'] = receipt['report_path']
    assignment['current_report_sha256'] = receipt['report_sha256']
    assignment['current_receipt_path'] = str(destination.relative_to(root))
    print(group, len(expected), 'current primary versions accepted; source/fig/screenshot bytes verified')
progress['revised_inventory_path'] = 'docs/course-revision-20261005/continuity/revised-02/inventory.json'
progress['revision_freeze_receipt'] = 'docs/course-revision-20261005/continuity/revised-02/freeze-receipt.json'
for group in ['06_09','13_16']:
    assignment = progress['assignments'][group]
    if not any(r.get('round')==1 for r in assignment.get('rechecks', [])):
        assignment.setdefault('rechecks', []).append({'round':1,'status':'actually_dispatched','reviewer_task':assignment['reviewer_task'],
            'dispatch_recorded_at':datetime.now(UTC).isoformat(),'exact_dispatched_at':None,
            'dispatch_evidence':'Actual root followup_task to same original continuity reader; exact server timestamp unavailable. 13_16 used revised-02 after actual current preview parity; 06_09 used revised-01 unchanged in its scope.',
            'report_path':f'docs/course-revision-20261005/continuity/reports/{group}-recheck-01.json',
            'trace_path':f'docs/course-revision-20261005/continuity/traces/{group}-recheck-01.jsonl'})
progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2)+'\n')
