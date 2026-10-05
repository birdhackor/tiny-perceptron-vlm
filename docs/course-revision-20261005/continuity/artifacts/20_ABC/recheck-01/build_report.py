import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from review_helper import ART, BASE, PAGES, ROOT, TRACE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


initial_report_path = BASE / 'reports/20_ABC.json'
initial_trace_path = BASE / 'traces/20_ABC.jsonl'
assert sha(initial_report_path) == 'e9b5fc50bf9abbe64436e601f2a195275ac6c9144af3a646aaa4cd5fb663afca'
assert sha(initial_trace_path) == 'e5c2608eec2eaaa0c7185c108146cc21515d828fc61b1dcaf104c50570c9a930'
initial = json.loads(initial_report_path.read_text())
trace = [json.loads(line) for line in TRACE.read_text().splitlines()]
expected_order = ['20.7', '20.8', '20.9', 'natural-v4-training', 'A.6', 'A.7', 'A.8', 'C.5', 'C.6', 'C.7']
assert [r['page_id'] for r in trace] == expected_order
assert all(r['reading_kind'] == 'recheck' and r['status'] == 'pass' for r in trace)
for r in trace:
    assert sha(ROOT / PAGES[r['page_id']]['snapshot']) == r['source_sha256']
    for source, expected in r['figures_sha256'].items():
        assert sha(ROOT / source) == expected
        assert sha(BASE / 'source-figures' / source) == expected

views = [json.loads(line) for line in (ART / 'pageviews.jsonl').read_text().splitlines()]
inspection = {}
for i, r in enumerate(trace, 1):
    for name in r.get('viewed_screenshots', []):
        inspection[name] = {'trace_line': i, 'recorded_at': r['recorded_at']}
for v in views:
    assert v['status'] == 200
    for s in v['screenshots']:
        assert sha(ROOT / s['path']) == s['sha256']
        assert Path(s['path']).name in inspection
        s['view_image_inspected'] = True
        s['inspection_trace_evidence'] = inspection[Path(s['path']).name]
for p in expected_order:
    for width, height in [(1280, 900), (390, 844)]:
        assert any(v['page_id'] == p and v['viewport'] == {'width': width, 'height': height} for v in views)
        if PAGES[p]['figures_sha256']:
            assert any(v['page_id'] == p and v['viewport'] == {'width': width, 'height': height} and v['screenshots'] for v in views)

changed_ids = ['20.8', 'A.7', 'C.7']
primary = []
for old in initial['pages']:
    p = old['page_id']
    current = PAGES[p]
    entry = {**current, 'status': 'pass'}
    if p in changed_ids:
        r = next(r for r in trace if r['page_id'] == p)
        assert r['source_sha256'] != old['source_sha256']
        assert r['original_issue_resolved']
        entry.update({'verification_basis': 'actual same-reader recheck-01', 'initial_source_sha256': old['source_sha256'], 'issue_id': r['original_issue_id'], 'pass_reason': r['understanding'], 'rechecked_at': r['recorded_at']})
    else:
        assert current['source_sha256'] == old['source_sha256']
        assert current['figures_sha256'] == old['figures_sha256']
        assert old['status'] == 'pass'
        entry.update({'verification_basis': 'unchanged source/figure hashes; genuine initial pass inherited', 'initial_report_path': str(initial_report_path.relative_to(ROOT)), 'initial_handoff_trace_lines': old['handoff_trace_lines'], 'whole_primary_page_newly_reread': False})
    primary.append(entry)

issues = []
for old in initial['issues']:
    r = next(r for r in trace if r.get('original_issue_id') == old['id'] and r.get('scope_role') == 'changed_primary')
    item = copy.deepcopy(old)
    item['initial_issue_status'] = old['status']
    item['status'] = 'pass'
    item['actual_reread_after_author_fix'] = {'recorded_at': r['recorded_at'], 'new_source_sha256': r['source_sha256'], 'figures_sha256': r['figures_sha256'], 'new_understanding': r['understanding'], 'resolution_location': r['resolution_location'], 'original_issue_resolved': True, 'new_issues': r['new_issues']}
    issues.append(item)

now = datetime.now(timezone.utc).isoformat()
report = {
    'schema_version': 1,
    'review_stage': 'independent_continuity_recheck',
    'assignment_id': '20_ABC',
    'reviewer_task': '/root/continuity_20_abc',
    'fork_turns': 'none',
    'recheck_round': '01',
    'reviewer_identity': 'same fresh third-round student reader who made the original 20_ABC initial report',
    'reader_background': initial['reader_background'],
    'reading_kind': 'recheck; not rewritten first reading',
    'status': 'pass',
    'completed_at': now,
    'initial_preserved': {'report_path': str(initial_report_path.relative_to(ROOT)), 'report_sha256': sha(initial_report_path), 'trace_path': str(initial_trace_path.relative_to(ROOT)), 'trace_sha256': sha(initial_trace_path), 'initial_files_unchanged': True},
    'current_source_version': {'inventory_path': str((BASE / 'revised-01/inventory.json').relative_to(ROOT)), 'inventory_sha256': sha(BASE / 'revised-01/inventory.json'), 'all_actually_rechecked_source_hashes_verified': True, 'necessary_figure_hashes_equal_current_source_and_original_frozen_figure_snapshots': True},
    'scope': {'changed_primary_ids': changed_ids, 'changed_primary_actual_pass': 3, 'unchanged_primary_inherited_pass': 42, 'effective_primary_pass_count': 45, 'newly_read_page_or_unit_count': 10, 'not_a_new_full_45_page_reading': True, 'additional_actual_context': ['20.7', '20.9', 'A.6', 'A.8', 'C.5', 'C.6', 'natural-v4-training section 6 only']},
    'actual_read_order': [{'page_id': r['page_id'], 'unit': r['unit'], 'unit_index': r.get('unit_index'), 'recorded_at': r['recorded_at'], 'trace_line': i, 'reading_kind': r['reading_kind']} for i, r in enumerate(trace, 1)],
    'primary_pages_effective_status': primary,
    'rechecked_pages_or_units': [{**PAGES[r['page_id']], 'unit': r['unit'], 'unit_index': r.get('unit_index'), 'unit_sha256': r.get('unit_sha256'), 'only_this_unit': r.get('scope_only_this_unit', False), 'status': r['status'], 'understanding': r['understanding'], 'recorded_at': r['recorded_at'], 'trace_line': i} for i, r in enumerate(trace, 1)],
    'handoffs': [{'trace_line': i, **r} for i, r in enumerate(trace, 1)],
    'issue_rechecks': issues,
    'remaining_issues': [],
    'pageviews': views,
    'visual_review': {'method': '本人實際 Chromium/Playwright 打開正文，1280x900 與 390x844，正常頁面捲動、full_page=False 截圖後逐張 view_image 真看；root HTML parity 未當作本人的看圖證據。', 'browser_executable': '/usr/bin/chromium', 'browser_args': ['--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking'], 'goto_wait_until': 'domcontentloaded', 'goto_timeout_ms': 15000, 'screenshot_timeout_ms': 15000, 'all_captured_screenshots_actually_viewed': True, 'screenshots_inspected': sum(len(v['screenshots']) for v in views), 'real_pageviews': len(views), 'necessary_figure_pages': ['20.7', '20.9', 'A.8', 'C.7'], 'verdict': 'pass; required labels, order and relationships readable in actual desktop and mobile article'},
    'unverified_and_not_reread_scope': [
        '其他42個未改primary沿用真初讀與一致指紋；沒有重新完整閱讀全部45頁。六個前後文有此次真回讀，training只有第6節另回讀，沒有重讀整指南。',
        '原初讀其他 context（19.11/19.12、14.7–14.10、16.12/16.13、8.13、1.15、1.12、1.11）本次未重新讀。',
        '沒有讀 repair manifests、作者修正筆記、其他 reviewer 或技術報告、實作或外部解釋。',
        '只看頁面的真CPU小例输出，沒有重新訓練或執行模型／資料下載、成熟應用、GPU、歷史完整實驗或選讀重做命令。',
        '未新讀2.3/5.4/5.5、W.1/T.1，沒有使用專家知識推導未教步驟。',
        '無圖頁有真桌機與手機 browser navigation，但沒有把整頁排版審核或所有截圖當成已做。',
    ],
    'artifacts': {'report_path': str((BASE / 'reports/20_ABC-recheck-01.json').relative_to(ROOT)), 'trace_path': str(TRACE.relative_to(ROOT)), 'trace_sha256': sha(TRACE), 'pageviews_path': str((ART / 'pageviews.jsonl').relative_to(ROOT)), 'pageviews_sha256': sha(ART / 'pageviews.jsonl'), 'artifact_directory': str(ART.relative_to(ROOT))},
}
path = BASE / 'reports/20_ABC-recheck-01.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report_path': str(path.relative_to(ROOT)), 'report_sha256': sha(path), 'trace_sha256': sha(TRACE), 'changed_primary_pass': 3, 'inherited_primary_pass': 42, 'actual_recheck_units': len(trace), 'real_pageviews': len(views), 'actual_screenshots_inspected': report['visual_review']['screenshots_inspected'], 'completed_at': now}, ensure_ascii=False))
