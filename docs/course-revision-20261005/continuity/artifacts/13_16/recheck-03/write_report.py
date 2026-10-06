import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


base = Path('docs/course-revision-20261005/continuity')
out = base / 'artifacts/13_16/recheck-03'
report_path = base / 'reports/13_16-recheck-03.json'
trace_path = base / 'traces/13_16-recheck-03.jsonl'
previous_path = base / 'reports/13_16-recheck-02.json'
inventory_path = base / 'revised-04/inventory.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


expected_inventory_sha = '4a834b09084d92360e8ed472705a2c3eb89273815b9b9431e8ff5f583704100a'
assert sha(inventory_path) == expected_inventory_sha
inventory = {p['page_id']: p for p in read_json(inventory_path)['pages']}
previous = read_json(previous_path)
previous_results = {p['page_id']: p for p in previous['primary_page_results']}
primary_ids = previous['primary_page_ids']
assert len(primary_ids) == 58
trace = read_jsonl(trace_path)
actual_ids = ['13.11', '13.12', '13.13', '13.14', '13.15']
assert [p['page_id'] for p in trace] == actual_ids
assert all(p['event'] == 'actual_reread' and p['status'] == 'pass' for p in trace)
assert [p['recorded_at'] for p in trace] == sorted(p['recorded_at'] for p in trace)
changed_ids = ['13.12', '13.14']
context_ids = ['13.11', '13.13', '13.15']
trace_by_id = {p['page_id']: p for p in trace}
pageviews_path = out / 'pageviews.jsonl'
pageviews = read_jsonl(pageviews_path)
assert len(pageviews) == 12 and all(p['status'] == 200 for p in pageviews)
screenshots = [s for p in pageviews for s in p['screenshots']]
assert len(screenshots) == 5
for s in screenshots:
    assert sha(s['path']) == s['sha256']

preservation_path = out / 'prior-preservation.json'
preservation = read_json(preservation_path)
assert len(preservation['files']) == 149
for p in preservation['files']:
    assert sha(p['path']) == p['sha256']
    assert Path(p['path']).stat().st_size == p['bytes']
verified_at = datetime.now(timezone.utc).isoformat()

source_bytes_path = out / 'current-source-bytes.json'
source_bytes = read_json(source_bytes_path)
for p in source_bytes['source_pages'] + source_bytes['source_figures']:
    assert sha(p['artifact_path']) == p['sha256']

new_reasons = {
    '13.11': '必要前文實讀：更新前critic平均估計與本批固定優勢，為下一節保留old收集機率作準備。',
    '13.12': '新正文實讀：old固定於本輪收集，reference固定於PPO起點（通常先示範微調），同稱舊模型的混淆在本節當場解開。',
    '13.13': '中間必要前文實讀：ratio仍以本輪old為分母；正負優勢裁切鼓勵方向，沒有把reference用途混進ratio。圖沿用本人同SHA實看。',
    '13.14': '新正文與新角色圖實讀實看：reference的PPO起點及整段PPO固定，和old同輪固定下輪重收分開；圖後SFT起點、reward先教固定、policy/critic更新的時間清楚。',
    '13.15': '必要接續實讀：收集舊紀錄、保存優勢、策略與critic更新相接；隨機網路流程示範和已保存先示範微調再PPO的實驗有明確通知。',
}
results = []
for pid in primary_ids:
    current = inventory[pid]
    old = previous_results[pid]
    assert old['decision'] == 'pass'
    assert sha(current['snapshot']) == current['source_sha256']
    same = (current['source_sha256'] == old['source_sha256'] and
            current['figures_sha256'] == old['figures_sha256'])
    assert same == (pid not in changed_ids)
    if pid in trace_by_id:
        own = trace_by_id[pid]
        assert own['source_sha256'] == current['source_sha256']
        assert own['figures_sha256'] == current['figures_sha256']
        assert hashlib.sha256(own['original_text'].encode()).hexdigest() == current['source_sha256']
        basis = ('changed_current_version_actually_reread' if pid in changed_ids
                 else 'same_exact_version_necessary_context_actually_reread')
        reason = new_reasons[pid]
        own_record = {k: v for k, v in own.items() if k != 'original_text'}
    else:
        basis = 'same_exact_version_inherited_without_new_full_read'
        reason = '本輪完全未再次全文閱讀此頁；当前正文與全部圖SHA和本人已PASS的recheck-02相同，沿用本人原判定。'
        own_record = None
    results.append({
        'page_id': pid, 'title': current['title'], 'source': current['source'],
        'selector': current['selector'], 'source_snapshot': current['snapshot'],
        'source_sha256': current['source_sha256'], 'figures_sha256': current['figures_sha256'],
        'decision': 'pass', 'status': 'pass', 'basis': basis, 'reason': reason,
        'same_exact_text_and_figures_as_recheck_02': same,
        'previous_decision': old['decision'], 'previous_source_sha256': old['source_sha256'],
        'previous_figures_sha256': old['figures_sha256'],
        'previous_report_path': str(previous_path), 'previous_reason': old['reason'],
        'actual_reread_record': own_record,
    })

prior_pageviews = read_jsonl(base / 'artifacts/13_16/recheck-01/pageviews.jsonl')
inherited_clip_views = [p for p in prior_pageviews if p.get('page_id') == '13.13' and p['screenshots']]
assert len(inherited_clip_views) == 2
for p in inherited_clip_views:
    for s in p['screenshots']:
        assert sha(s['path']) == s['sha256']

artifact_manifest_path = out / 'artifact-manifest.json'
artifact_files = []
for path in sorted(out.rglob('*')):
    if path.is_file() and path != artifact_manifest_path:
        artifact_files.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size})
save_json(artifact_manifest_path, {
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'scope': 'only new own recheck-03 artifacts; excludes this manifest to avoid self-reference',
    'file_count': len(artifact_files), 'files': artifact_files,
})

report = {
    'schema_version': 3, 'stage': 'continuity_recheck', 'phase': 4,
    'assignment': '13_16', 'recheck_id': 'recheck-03',
    'reviewer_task': '/root/continuity_13_16', 'fork_turns': 'none',
    'identity_statement': '本人是原第三輪13_16及recheck-01/02實際讀者；這次是同人必要銜接回查，非新盲讀。只使用當前正文、圖與本人既有閱讀記憶/紀錄。',
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'status': 'done', 'current_verdict': 'pass', 'must_fix': [], 'optional': [],
    'verdict_reason': '13.11→13.12→13.13→13.14→13.15依序真回讀後，PPO起點reference、本輪old、reward/critic與示範/既有實驗的交接可由正文理解；新圖與相鄰文字在桌機及手機可讀。本輪未遇到必改或可選問題。',
    'source_inventory_path': str(inventory_path), 'source_inventory_sha256': expected_inventory_sha,
    'changed_primary_page_ids': changed_ids, 'changed_figure_page_ids': ['13.14'],
    'actual_reread_page_ids': actual_ids,
    'actual_read_order': [{'sequence': n + 1, **{k: v for k, v in t.items() if k != 'original_text'}}
                          for n, t in enumerate(trace)],
    'primary_page_ids': primary_ids, 'primary_page_count': 58,
    'primary_page_results': results,
    'same_exact_version_primary_count': 56, 'actual_reread_primary_count': 5,
    'changed_version_actually_reread_primary_count': 2,
    'same_exact_version_with_context_reread_primary_count': 3,
    'same_exact_version_without_new_read_primary_count': 53,
    'necessary_context_current_versions': [r for r in results if r['page_id'] in context_ids],
    'original_records': [p for p in preservation['files']
                         if '/reports/' in p['path'] or '/traces/' in p['path'] or '/receipts/' in p['path']],
    'prior_bytes_preservation': {
        'manifest_path': str(preservation_path), 'manifest_sha256': sha(preservation_path),
        'file_count': 149, 'all_prior_bytes_unchanged': True, 'verified_at': verified_at,
        'scope': 'initial/recheck-01/recheck-02 reports, traces, receipts and all existing own evidence; no earlier understanding or decision edited',
        'receipt_scope': 'receipt contents not consulted for understanding or verdict; raw bytes hashed only',
    },
    'issue_recheck': [],
    'current_stage_checks': [
        {'page_id': '13.12', 'quote': 'old與reference也不同：old對照本輪收集那一刻，下輪重新收集時換新紀錄；reference對照PPO開始時的策略（通常已完成示範微調），整段PPO期間可以固定。兩者都叫「舊模型」會混淆用途。',
         'understanding_at_first_current_read': trace_by_id['13.12']['understanding'],
         'later_explanation': '13.14新圖把reference寫為PPO起點/整段PPO期間不更新，old寫同輪固定/下輪重新收集；與本節相接。',
         'missing_bridge': None, 'severity': None, 'recommendation': None, 'decision': 'pass'},
        {'page_id': '13.14', 'quote': 'reference保留PPO開始時、已完成示範微調的策略，在這段PPO期間固定。',
         'understanding_at_first_current_read': trace_by_id['13.14']['understanding'],
         'later_explanation': '13.15在隨機網路程式前明說只觀察更新流、reward未教好；既有保存實驗另明說先示範微調與評分員比較，再凍結reward做PPO。因此沒有把隨機程式當成已教好策略。',
         'missing_bridge': None, 'severity': None, 'recommendation': None, 'decision': 'pass'},
    ],
    'handoffs': [
        {'from': '13.11', 'to': '13.12', 'decision': 'pass', 'reason': '固定本批優勢需配同批收集機率；ratio的old和較長時間reference各有本地定義。'},
        {'from': '13.12', 'to': '13.13', 'decision': 'pass', 'reason': '裁切仍對同張卡的current/收集old比；不把固定reference錯作本輪ratio分母。'},
        {'from': '13.13', 'to': '13.14', 'decision': 'pass', 'reason': '裁切的同輪鼓勵與相對PPO起點reference的KL偏移另列；reward選後評分與critic事前平均也另列。'},
        {'from': '13.14', 'to': '13.15', 'decision': 'pass', 'reason': '保存更新前critic/old及重算current policy相接；隨機流程示範與既有先示範微調再PPO的實驗明確分開。'},
    ],
    'figure_visual_inheritance': [{
        'page_id': '13.13', 'current_figures_sha256': inventory['13.13']['figures_sha256'],
        'reason_no_new_figure_view': '本輪只改13.12/13.14與角色图；13.13裁切圖、正文與圖說用途未變，先前本人recheck-01真實桌機/手機view_image仍同SHA。此次正文真回讀，但不聲稱此圖新視讀。',
        'prior_actual_pageviews': inherited_clip_views,
        'actual_prior_viewed_screenshots': [s for p in inherited_clip_views for s in p['screenshots']],
    }],
    'current_changed_figure_visual_review': {
        'page_id': '13.14', 'url': 'http://127.0.0.1:8765/13.14.html',
        'source_figure': 'course/figures/rewrite-13-model-roles.svg',
        'source_figure_sha256': inventory['13.14']['figures_sha256']['course/figures/rewrite-13-model-roles.svg'],
        'viewed_current_screenshots': screenshots, 'view_image_actually_used_for_all_five': True,
        'understanding': '桌機分兩段/手機整圖看到五角色输入输出；reference PPO起點及整段PPO不更新，old同輪固定下輪重收。另兩張正文圖後段落截图實看，確定SFT起點、reward先教固定、policy/critic更新時間。',
        'missing_figure_or_unreadable_text': False,
    },
    'current_source_bytes_path': str(source_bytes_path), 'current_source_bytes_sha256': sha(source_bytes_path),
    'current_figure_http_fingerprints': source_bytes['source_figures'],
    'trace_path': str(trace_path), 'trace_sha256': sha(trace_path),
    'artifacts_directory': str(out), 'artifact_manifest_path': str(artifact_manifest_path),
    'artifact_manifest_sha256': sha(artifact_manifest_path),
    'pageviews_path': str(pageviews_path), 'pageviews_sha256': sha(pageviews_path),
    'pageviews': pageviews, 'actual_pageview_count': 12,
    'actual_current_figure_screenshots_viewed_count': 3,
    'actual_current_adjacent_text_screenshots_viewed_count': 2,
    'actual_current_screenshots_viewed_count': 5,
    'browser_method': {
        'browser': '/usr/bin/chromium via Playwright',
        'args': ['--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking'],
        'goto_wait_until': 'domcontentloaded', 'goto_timeout_ms': 15000,
        'screenshot_full_page': False, 'screenshot_timeout_ms': 15000,
        'viewports': [{'width': 1280, 'height': 900}, {'width': 390, 'height': 844}],
        'view_image_actually_used': True, 'cli_screenshot_used': False,
    },
    'unverified_scope': [
        '這次只按順序全文回讀五頁；其餘53primary完全未再次全文閱讀，沿用本人同文字/全部圖SHA的既有PASS。',
        '13.13同SHA圖只繼承本人先前真實視讀；本輪新的視讀限13.14角色圖及相鄰段落桌機/手機。',
        '沒有閱讀作者repair notes、其他reader或technical報告、實作或外部教材補理解；公開補充只讀正文，沒有開所連報告或論文。',
        '沒有執行程式、重新CPU/GPU訓練或用工程結果判學生懂；教材數字讀作既有紀錄/手算示範，未独立驗證技術真偽。',
        '本人的repair回讀結論不能替代全新盲讀，也不宣稱其他範圍新全文閱讀或Phase4整體收閉。',
    ],
}
save_json(report_path, report)

for p in preservation['files']:
    assert sha(p['path']) == p['sha256']
for p in read_json(artifact_manifest_path)['files']:
    assert sha(p['path']) == p['sha256']
print(json.dumps({
    'status': 'done', 'current_verdict': 'pass', 'must_fix': 0, 'optional': 0,
    'actual_read_order': actual_ids, 'primary_count': 58, 'actual_reread_count': 5,
    'inherited_without_new_read_count': 53, 'preserved_prior_files': 149,
    'report_path': str(report_path), 'report_sha256': sha(report_path),
    'trace_path': str(trace_path), 'trace_sha256': sha(trace_path),
    'artifact_manifest_path': str(artifact_manifest_path), 'artifact_manifest_sha256': sha(artifact_manifest_path),
    'artifact_count': len(artifact_files),
    'pageviews_sha256': sha(pageviews_path), 'preservation_manifest_sha256': sha(preservation_path),
}, ensure_ascii=False, indent=2))
