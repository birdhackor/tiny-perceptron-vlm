import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-revision-20261005/continuity'
OUT = BASE / 'artifacts/01_05/recheck-01'
TRACE = BASE / 'traces/01_05-recheck-01.jsonl'
REPORT = BASE / 'reports/01_05-recheck-01.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rel(path):
    return str(path.relative_to(ROOT))

original = json.loads((BASE / 'reports/01_05.json').read_text())
original_inventory = {p['page_id']: p for p in json.loads((BASE / 'inventory.json').read_text())['pages']}
current_inventory = {p['page_id']: p for p in json.loads((BASE / 'revised-01/inventory.json').read_text())['pages']}
original_pages = {p['page_id']: p for p in original['page_reviews']}
original_issues = {i['issue_id']: i for i in original['issues']}
trace = [json.loads(x) for x in TRACE.read_text().splitlines()]
records = [(i + 1, row) for i, row in enumerate(trace) if row.get('record_type') != 'metadata_correction']
corrections = [row for row in trace if row.get('record_type') == 'metadata_correction']
pageviews = [json.loads(x) for x in (OUT / 'pageviews.jsonl').read_text().splitlines()]
fig_checks = json.loads((OUT / 'figure-version-checks.json').read_text())
baseline = json.loads((OUT / 'initial-record-hashes.json').read_text())
preservation = []
for name, expected in baseline['initial_files'].items():
    actual = sha(BASE / name)
    assert actual == expected, name
    preservation.append({'path': rel(BASE / name), 'before_sha256': expected, 'after_sha256': actual, 'unchanged': True})

read_order = list(dict.fromkeys(row['page_id'] for _, row in records))
changed = ['2.5', '5.1', '5.8', '5.9', '5.16', 'readme']
primary = original['primary_page_ids']
assert len(primary) == 62
actual_changed = [p for p in primary if original_inventory[p]['source_sha256'] != current_inventory[p]['source_sha256'] or original_inventory[p]['figures_sha256'] != current_inventory[p]['figures_sha256']]
assert set(actual_changed) == set(changed)
assert read_order == ['2.4', '2.5', 'chapter-03', '4.8', 'chapter-05', '5.1', '5.2', '5.7', '5.8', '5.9', '5.10', '5.15', '5.16', '5.17', 'readme', 'chapter-06', '6.1', '6.2']
assert all(row['decision'] == 'pass' for _, row in records)
assert all(p['viewed'] and sha(Path(p['screenshot_absolute'])) == p['screenshot_sha256'] for p in pageviews)
assert len(pageviews) == 8
assert all(c['matches'] for c in fig_checks)

def corrected_issue_id(page, iid):
    if page == '5.7' and iid == '01_05-08':
        return '01_05-09'
    if page == '5.8' and iid == '01_05-09':
        return '01_05-10'
    return iid

def effective_record(row):
    value = json.loads(json.dumps(row))
    for field in ('current_questions', 'issue_rechecks'):
        for item in value.get(field, []):
            if 'issue_id' in item:
                old_id = item['issue_id']
                new_id = corrected_issue_id(value['page_id'], old_id)
                if new_id != old_id:
                    item.update(issue_id=new_id, raw_trace_issue_reference=old_id, metadata_correction_applied=True)
    return value

issue_rechecks = []
for line, row in records:
    for check in row.get('issue_rechecks', []):
        check = dict(check)
        iid = corrected_issue_id(row['page_id'], check['issue_id'])
        check.update(issue_id=iid, page_id=row['page_id'], recheck_recorded_at_utc=row['recorded_at_utc'], trace_line=line, current_source_sha256=row['source_sha256'], current_figures_sha256=row['figures_sha256'], original_issue=original_issues[iid], decision='pass')
        issue_rechecks.append(check)
required_rechecks = [r for r in issue_rechecks if r['issue_id'] in original['required_revision_issue_ids']]
assert len(required_rechecks) == 4 and all(r['status'] == 'resolved' for r in required_rechecks)

reviews = []
for p in primary + ['chapter-06', '6.1', '6.2']:
    meta = current_inventory[p]
    actual = sha(ROOT / meta['snapshot'])
    assert actual == meta['source_sha256']
    reread = [(line, r) for line, r in records if r['page_id'] == p]
    unchanged = original_inventory[p]['source_sha256'] == meta['source_sha256'] and original_inventory[p]['figures_sha256'] == meta['figures_sha256']
    if not reread:
        assert unchanged and original_pages[p]['decision'] == 'pass'
    evidence = [v for v in pageviews if v['page_id'] == p]
    reviews.append({
        'page_id': p, 'coverage_role': 'primary' if p in primary else 'context_after_only',
        'title': meta['title'], 'source_snapshot': meta['snapshot'],
        'source_sha256': meta['source_sha256'], 'source_snapshot_sha_verified': True,
        'figures_sha256': meta['figures_sha256'],
        'initial_source_sha256': original_inventory[p]['source_sha256'],
        'initial_figures_sha256': original_inventory[p]['figures_sha256'],
        'source_and_figures_unchanged_since_initial': unchanged,
        'coverage_basis': 'actual_reread_recheck_01' if reread else 'own_initial_reading_inherited_for_identical_version',
        'actual_recheck_trace_lines': [line for line, _ in reread],
        'actual_recheck_sections': [r['section'] for _, r in reread],
        'boundary_understanding': [effective_record(r) for _, r in reread] if reread else original_pages[p]['boundary_understanding'],
        'initial_trace_lines': original_pages[p]['actual_trace_lines'],
        'decision': 'pass',
        'reason': '；'.join(r['understood_task'] for _, r in reread) if reread else original_pages[p]['reason'],
        'issue_ids': [i['issue_id'] for i in original['issues'] if i['page_id'] == p],
        'actual_recheck_figure_pageviews': evidence,
        'figure_visual_evidence_basis': 'new_actual_screenshots_and_view_image' if evidence else ('own_initial_actual_visual_check_for_same_figure_hash' if meta['figures_sha256'] else 'no_embedded_necessary_figure'),
        'inherited_initial_figure_pageviews': original_pages[p].get('figure_pageviews', []) if not evidence else [],
        'unverified': ['本輪未重讀此頁；判定沿用本人初讀及完全相同來源／圖SHA。'] if not reread else [],
    })

boundary_specs = [
    (['2.4', '2.5', 'chapter-03'], '非線性先擴充可表示規則，再問線索是否送進視窗；答案圖已分清圓／方與實際輸入。章3沿紅／藍同例問可見線索如何按需求讀取。'),
    (['4.8', 'chapter-05', '5.1', '5.2'], '未訓練結構計數先說明不證品質；導讀明示改固定題改善，5.1區分梯度大小和代價下降，5.2沿五候選與-100說明多答案等票平均。'),
    (['5.7', '5.8', '5.9', '5.10'], '可接續固定題訓練不等於新題能力；5.8分手寫預測和真生成，byte/token就在故事長度旁说明。5.9先定義儲存單位再估成本；5.10明示假設驗證數只用選設定，test仍封存。'),
    (['5.15', '5.16', '5.17'], '比較波動之外還要核對監督量。5.16先定義每有效位置token與PAD補格，再數分母；5.17以獨立Linear區分評估行為／求導，接上5.9的eval/no_grad。'),
    (['readme'], '全頁六單位逐heading重讀；閱讀、安裝、最小通路、維護和授權入口各有具體對象，編號圖手機字／ID／雙向箭頭／圖說都能直接讀。'),
    (['5.17', 'chapter-06', '6.1', '6.2'], '第6章改問每格代表什麼，展開5.8已就地說明的拆分單位；6.1用實際碼點／byte數與成本相連，6.2玩具BPE明示按文件合併，同字數不等於同token數。此概念邊界在指定實讀順序中隔著README，並非聲稱兩頁連續開啟。'),
]
boundaries = [{'page_ids': ids, 'understanding': reason, 'decision': 'pass', 'trace_lines': [line for line, r in records if r['page_id'] in ids]} for ids, reason in boundary_specs]

rendered = []
for p in read_order:
    path = OUT / f'{p}.html'
    assert path.is_file()
    rendered.append({'page_id': p, 'url': f'http://127.0.0.1:8765/{p}.html', 'snapshot': rel(path), 'sha256': sha(path), 'evidence_kind': 'HTTP正文及真CPU輸出讀取，不代表非圖頁已做完整響應版面目視檢查'})
(OUT / 'rendered-page-manifest.json').write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + '\n')

report = {
    'schema_version': 1, 'stage': 'third-round-continuity-original-reader-recheck',
    'assignment_id': '01_05', 'recheck_id': 'recheck-01',
    'reviewer_task': '/root/continuity_01_05', 'fork_turns': 'none',
    'reader_background': original['reader_background'],
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'original_report': rel(BASE / 'reports/01_05.json'),
    'original_trace': rel(BASE / 'traces/01_05.jsonl'),
    'initial_record_preservation': preservation,
    'source_inventory': rel(BASE / 'revised-01/inventory.json'),
    'source_inventory_sha256': sha(BASE / 'revised-01/inventory.json'),
    'decision': 'pass',
    'overall_reason': '原四個必改問題均在需要的位置實際回讀解開；2.5與README圖在桌機與手機正文實看可讀。六個指定改動主頁及必要前後文沒有新增必要橋梁缺口。其餘主頁只沿用本人真初讀及完全相同來源／圖版本，不聲稱本輪62頁全重讀。原有選讀改善與後文才解開的初疑完整保留。',
    'primary_page_ids': primary, 'changed_primary_page_ids': changed,
    'context_after': ['chapter-06', '6.1', '6.2'],
    'coverage': {
        'initial_primary_pages_actually_read': 62,
        'changed_primary_pages_actually_reread': 6,
        'unchanged_primary_context_pages_actually_reread': len([p for p in read_order if p in primary and p not in changed]),
        'unchanged_primary_pages_inherited_without_this_round_reread': len([p for p in primary if p not in read_order]),
        'context_after_pages_actually_reread': 3,
        'actual_unique_pages_reread': len(read_order),
        'actual_reading_units': len(records), 'trace_rows_including_metadata_correction': len(trace),
        'actual_screenshots_viewed': len(pageviews),
        'context_is_primary_coverage': False,
    },
    'actual_read_order': read_order,
    'actual_section_read_order': [{'trace_line': line, 'page_id': r['page_id'], 'section': r['section'], 'recorded_at_utc': r['recorded_at_utc']} for line, r in records],
    'methodology': {
        'original_reader_recheck_not_fresh_blind_read': True,
        'rules_read': ['docs/review-tools/continuity-reviewer-instructions.md', '.agents/skills/clear-tutorial/references/review-protocol.md'],
        'unit_method': '指定課節與章導讀逐頁；README開場及五個## heading逐單位。讀完當場追加新UTC紀錄，再揭露下一單位；沒有倒填。',
        'incidental_visual_exposure': 'README手機正常正文圖截圖下緣露出下一heading及首段，已在開場trace如實記下；此輪是已完整初讀過該頁的原讀者非盲複查。',
        'recheck_trace_metadata_correction': corrections,
        'materials_read': '目前凍結教材、本人原報告及初讀記憶（原trace本輪只核對SHA）以及兩份方法；沒有讀作者manifest、作者筆記、其他讀者／技術報告、實作程式或外部解釋。',
        'visual_method': '/usr/bin/chromium + Playwright；1280x900及390x844；no-sandbox、disable-dev-shm-usage、disable-background-networking；goto domcontentloaded及screenshot full_page=False timeout15000；所有八張截圖均用view_image实际看。',
        'cpu_evidence_scope': '讀目前網站已有真CPU輸出以判定正文銜接；未執行教材程式、訓練或技術核驗。',
        'new_delegation': False,
    },
    'boundary_reviews': boundaries,
    'original_required_revision_issue_ids': original['required_revision_issue_ids'],
    'required_revision_issue_ids': [],
    'required_issue_rechecks': required_rechecks,
    'additional_issue_rechecks': [r for r in issue_rechecks if r not in required_rechecks],
    'original_issues_preserved_verbatim': original['issues'],
    'new_issues': [],
    'remaining_initial_optional_issue_ids': [i['issue_id'] for i in original['issues'] if i['severity'] == '選讀改善' and i['issue_id'] not in {r['issue_id'] for r in issue_rechecks}],
    'page_reviews': reviews,
    'actual_pageviews': pageviews,
    'source_figure_version_checks': fig_checks,
    'rendered_page_snapshots': rendered,
    'trace_path': rel(TRACE), 'trace_sha256': sha(TRACE),
    'unverified_scope': [
        '本輪只有18個不同頁面實際重讀；47個未變主頁不聲稱重讀，依本人原初讀與來源／圖SHA一致繼承。',
        '除2.5、README、6.2的必要圖之外，本輪未重新目視驗證所有頁的桌機／手機完整版面；2.4等未變圖沿用本人原初讀目視證據。',
        '沒有執行教材程式、練習、安裝、Colab、權重下載、GPU、训练、网站建置或技術數值核驗；網站CPU輸出只作讀者所見教材。',
        '沒有開啟6.3以後的新正文、選讀操作／證據連結、作者資料、其他審閱者資料、實作程式或外部來源。',
        '這是原AI讀者的前後銜接複查，非新的真人目標讀者測試；不能以此代替獨立技術查證。',
    ],
    'writes_scope': [rel(OUT), rel(TRACE), rel(REPORT)],
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report': rel(REPORT), 'report_sha256': sha(REPORT), 'decision': report['decision'], 'coverage': report['coverage'], 'resolved_required_issue_ids': [r['issue_id'] for r in required_rechecks], 'initial_records_unchanged': True}, ensure_ascii=False))
