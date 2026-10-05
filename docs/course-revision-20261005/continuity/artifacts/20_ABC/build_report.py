import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from review_helper import ART, BASE, PAGES, ROOT, TRACE, units


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


trace = [json.loads(line) for line in TRACE.read_text().splitlines()]
assignment = json.loads((BASE / 'progress.json').read_text())['assignments']['20_ABC']
primary = assignment['primary_page_ids']
initial = [(i + 1, r) for i, r in enumerate(trace) if r.get('page_id') and not r.get('event')]
by_page = {p: [(i, r) for i, r in initial if r['page_id'] == p] for p in primary}
assert len(primary) == 45 and all(by_page.values())
guide_ids = ['natural-v4-student', 'natural-v4-data', 'natural-v4-training']
for p in guide_ids:
    assert [r['unit_index'] for _, r in by_page[p]] == list(range(len(units(p))))

read_ids = list(dict.fromkeys(r['page_id'] for _, r in initial))
figures = json.loads((BASE / 'figure-snapshots.json').read_text())['figures']
for p in read_ids:
    assert sha(ROOT / PAGES[p]['snapshot']) == PAGES[p]['source_sha256']
    for source, expected in PAGES[p]['figures_sha256'].items():
        assert figures[source]['sha256'] == expected
        assert sha(ROOT / figures[source]['snapshot']) == expected
        assert sha(ROOT / source) == expected

inspections = {}
for i, r in enumerate(trace, 1):
    for name in r.get('viewed_screenshots', []):
        inspections.setdefault(name, []).append({'trace_line': i, 'recorded_at': r['recorded_at'], 'page_id': r['page_id']})
pageviews = [json.loads(line) for line in (ART / 'pageviews.jsonl').read_text().splitlines()]
for v in pageviews:
    assert v['status'] == 200
    for s in v['screenshots']:
        assert sha(ROOT / s['path']) == s['sha256']
        evidence = inspections.get(Path(s['path']).name)
        assert evidence, s['path']
        s['view_image_inspected'] = True
        s['inspection_trace_evidence'] = evidence
for p in read_ids:
    assert any(v['page_id'] == p for v in pageviews)
for p in read_ids:
    if PAGES[p]['figures_sha256']:
        for viewport in [{'width': 1280, 'height': 900}, {'width': 390, 'height': 844}]:
            assert any(v['page_id'] == p and v['viewport'] == viewport and v['screenshots'] for v in pageviews)

pages = []
for p in primary:
    records = by_page[p]
    verdict = 'revise' if any(r.get('issues') for _, r in records) else 'pass'
    entry = copy.deepcopy(PAGES[p])
    entry.update({
        'status': verdict,
        'first_read_completed_at': records[0][1]['recorded_at'],
        'last_unit_read_completed_at': records[-1][1]['recorded_at'],
        'pass_reason_or_revise_reason': [r['understanding'] for _, r in records],
        'handoff_trace_lines': [i for i, _ in records],
        'issue_ids': [issue['id'] for _, r in records for issue in r.get('issues', [])],
        'browser_view_indices': [i + 1 for i, v in enumerate(pageviews) if v['page_id'] == p],
    })
    if p in guide_ids:
        entry['reading_units'] = [{
            'unit_index': r['unit_index'],
            'unit': r['unit'],
            'unit_sha256': hashlib.sha256(units(p)[r['unit_index']].encode()).hexdigest(),
            'unit_hash_method': 'SHA-256 of frozen UTF-8 source unit; computed during report audit, not an invented first-read timestamp',
            'read_completed_at': r['recorded_at'],
            'trace_line': i,
            'status': r['status'],
        } for i, r in records]
    pages.append(entry)

issues = []
for i, r in initial:
    for issue in r.get('issues', []):
        item = copy.deepcopy(issue)
        item.update({
            'page_id': r['page_id'],
            'unit': r['unit'],
            'source_sha256': r['source_sha256'],
            'figures_sha256': r['figures_sha256'],
            'first_encounter_recorded_at': r['recorded_at'],
            'first_encounter_trace_line': i,
            'can_answer_from_already_read_text': False,
            'status': 'revise',
            'actual_reread_after_author_fix': 'pending; no revised source has been assigned to this same reader yet',
        })
        if item['id'] == '20ABC-01':
            item['can_answer_from_already_read_text_detail'] = '可看出正確逐字稿聊天 4/4 降到 2/4；當時未讀到門檻數值與 primary 的各組權重，不能自行重算接受條件。'
            later = [r for r in trace if r.get('event') == 'later_clarification' and r.get('issue_id') == item['id']]
            item['later_resolution'] = later
            item['remaining_bridge'] = 'natural-v4-training 第 6 節已提供條件；20.8 的讀者當下仍需要一句「兩條语音聊天路线不得退步、全部完整結束」與直達該節的連結。'
            item['suggestion'] = '20.8 就近補說語音聊天兩路至少保持原底座正確率、全部 EOS 完整；嚴格綜合比較可直鏈訓練指南第 6 節。不必把完整表格重複塞入正文。'
        elif item['id'] == '20ABC-02':
            item['can_answer_from_already_read_text_detail'] = '能用 C=100、原 P=75、G=25 算出歷史增加 30 後 P=105；少留回答最多減 25，仍超 5。能察覺矛盾，但沒有一句說明本例必須先刪輸入。'
            item['later_resolution'] = '後面的檢索 40 縮成 10 可使總需求 100；沒有解開前面「或」暗示只縮回答也能處理本例的問題。'
        elif item['id'] == '20ABC-03':
            item['can_answer_from_already_read_text_detail'] = '能從已讀 C.5、C.6 自行找出固定權重作答計算在 C.5；但「上節」把參照指向緊鄰的 C.6，需回頭猜作者所指。'
            item['later_resolution'] = '補讀 1.12、1.11 解開梯度更新前置，未改變這個小節指稱問題。'
        issues.append(item)

def reading_order(ids):
    return [{'page_id': r['page_id'], 'unit': r['unit'], 'unit_index': r.get('unit_index'),
             'read_completed_at': r['recorded_at'], 'trace_line': i}
            for i, r in initial if r['page_id'] in ids]

now = datetime.now(timezone.utc).isoformat()
report = {
    'schema_version': 1,
    'review_stage': 'independent_continuity',
    'assignment_id': '20_ABC',
    'reviewer_task': '/root/continuity_20_abc',
    'fork_turns': 'none',
    'reviewer_identity': 'fresh third-round continuity student reader; not author, initial reader, or technical reviewer',
    'reader_background': '高中數學與入門 Python；可做代數與閱讀短程式。對模型概念只使用這份 trace 的實際先讀正文；未讀整個前章。',
    'independence': {
        'other_reviewer_reports_read': False,
        'author_notes_read': False,
        'technical_reports_read': False,
        'implementation_read': False,
        'external_explanations_read': False,
        'delegated_work': False,
        'root_provided_sources_not_expected_answers': True,
    },
    'method_paths': ['docs/review-tools/continuity-reviewer-instructions.md', '.agents/skills/clear-tutorial/references/review-protocol.md'],
    'initial_review_completed_at': now,
    'status': 'revise',
    'primary_page_count': len(primary),
    'primary_pass_count': sum(p['status'] == 'pass' for p in pages),
    'primary_revise_count': sum(p['status'] == 'revise' for p in pages),
    'blocking_main_concept_issue_count': 0,
    'necessary_figure_missing_issue_count': 0,
    'source_versions': {
        'inventory_path': str((BASE / 'inventory.json').relative_to(ROOT)),
        'inventory_sha256_at_report': sha(BASE / 'inventory.json'),
        'figure_snapshots_path': str((BASE / 'figure-snapshots.json').relative_to(ROOT)),
        'figure_snapshots_sha256_at_report': sha(BASE / 'figure-snapshots.json'),
        'source_and_figure_snapshot_hashes_verified': True,
        'live_original_figure_hashes_equal_snapshots': True,
        'note': '所有 source_sha256 是派工時保存的原始 UTF-8 版本。指紋核對僅確認版本一致；讀過的證據是逐節 trace 與實際 pageviews。',
    },
    'actual_read_order': reading_order(read_ids),
    'context_before': [{**PAGES[p], 'trace_lines': [i for i, r in initial if r['page_id'] == p]} for p in ['19.11', '19.12']],
    'other_actually_read_context': [{**PAGES[p], 'trace_lines': [i for i, r in initial if r['page_id'] == p]} for p in read_ids if p not in primary and p not in ['19.11', '19.12']],
    'branch_entries': [
        {
            'branch': '20',
            'actual_entry': '先讀 19.11、19.12，再按第 20 章的延伸應用導言讀 20.1–20.13，最後逐節讀它指向的三份操作／資料／訓練指南。',
            'handoff_understanding': '19.11 的 step 0 封裝檢查、19.12 的待驗收共同 MoE 與既有合成 joint 證據均不是成熟自然資料成品。第 20 章另採已預訓練 Qwen3-VL-2B-Instruct 與 Whisper-large-v3-turbo，驗證未接受 LoRA，所以現用原底座。',
            'actual_read_order': reading_order(['19.11', '19.12', 'chapter-20'] + [f'20.{i}' for i in range(1, 14)] + guide_ids),
        },
        {
            'branch': 'A',
            'actual_entry': '第 20 章與指南完成後進入 A 導言；先真讀提供定位的長文前文 14.7–14.10、16.12–16.13，再順讀 A.1–A.10。',
            'handoff_understanding': '長度擴展、座標／RoPE 縮放、滑窗、切塊計算與可見範圍的差異有已讀正文依據；A 中紙上書店資料與各個隨機起步 ASCII 實測另行辨認，未當成 Qwen 續訓或同一個模型。',
            'actual_read_order': reading_order(['chapter-0A', '14.7', '14.8', '14.9', '14.10', '16.12', '16.13'] + [f'A.{i}' for i in range(1, 11)]),
        },
        {
            'branch': 'B',
            'actual_entry': 'A 完成後依派工定位先讀 8.13，發現其實是 LoRA；保留真順序。B 導言及 B.1 自行交代工具往返，依此讀 B.1–B.8。',
            'handoff_understanding': '工具請求文字不會自行執行；驗證、真正執行、回傳結果、後續回答各有角色。舊 ASCII 工具模型、策略選擇器與紙上 request 列表可分辨，未當作成熟應用實測。',
            'dispatch_locator_correction': 'root 明確確認 8.13 建議是派工定位錯誤，不是教材工具前置缺漏；原紀錄不改寫。',
            'actual_read_order': reading_order(['8.13', 'chapter-0B'] + [f'B.{i}' for i in range(1, 9)]),
        },
        {
            'branch': 'C',
            'actual_entry': '獨立 C 導言後讀 C.1；依 C.1 正文連結真讀 1.15 抽樣溫度，再讀 C.2–C.7；依 C.7 的更新連結讀 1.12，再补读其中已知梯度來源 1.11。',
            'handoff_understanding': '寫步驟、多候選、驗證選擇、固定權重的回答計算與 reward 更新是可分开的問題。最後有限 17 動作策略與兩個語言模型分開，policy 檔不是 TinyLM 權重。數字例子、既有實測、oracle coverage 与实际selector分開理解。',
            'actual_read_order': reading_order(['chapter-0C', 'C.1', '1.15'] + [f'C.{i}' for i in range(2, 8)] + ['1.12', '1.11']),
        },
    ],
    'pages': pages,
    'handoffs': [{'trace_line': i, **r} for i, r in initial],
    'later_clarifications_and_supplementary_events': [{'trace_line': i + 1, **r} for i, r in enumerate(trace) if r.get('event')],
    'issues': issues,
    'pageviews': pageviews,
    'visual_review': {
        'required_viewports': [{'width': 1280, 'height': 900}, {'width': 390, 'height': 844}],
        'method': '實際 http://127.0.0.1:8765/<page_id>.html 正文，Chromium /usr/bin/chromium + Playwright。圖頂端／過長圖底端正常捲動截圖 full_page=False，之後用 view_image 真看；沒有以 SVG 文字、DOM 尺寸或原 SVG 放大代替。',
        'browser_args': ['--no-sandbox', '--disable-dev-shm-usage', '--disable-background-networking'],
        'goto_wait_until': 'domcontentloaded',
        'goto_timeout_ms': 15000,
        'screenshot_timeout_ms': 15000,
        'all_captured_screenshots_actually_viewed': True,
        'screenshot_count': sum(len(v['screenshots']) for v in pageviews),
        'figure_page_count': sum(bool(PAGES[p]['figures_sha256']) for p in read_ids),
        'assessment': '必要圖能在正文桌機／手機中辨讀順序與關鍵標籤；長圖以普通頁面捲動續看。未留必要圖缺漏或必須另開放大才懂的問題。',
    },
    'runtime_review_scope': '真瀏覽器頁面的內嵌 CPU 結果已讀並保存。這不是重新執行成熟模型、訓練、下載資料或驗證歷史完整實驗。早期 pageviews 未存 CPU 字段者另有後來補充事件，保留首次理解不倒填。',
    'unverified_scope': [
        '未讀完整第 1、8、14、16、19 章；僅列出的 context 頁是真讀。19.9 未追讀，未假裝有其前文知識。',
        '未讀 C.7 延伸的 2.3、5.4、5.5 詳細 one-hot／交叉熵／AdamW 方法，未用模型專家知識補其推導。',
        '未執行 clone、安裝、模型與資料下载、成熟應用啟動、離線操作、CPU 成熟推論、GPU 或任何訓練。',
        '未重現歷史大模型／ASCII／有限策略成績，未查外部來源或實作程式；這一輪只判讀者能否辨認正文證據與範圍。',
        'W.1、T.1 遠端／GPU 指南連結未追讀；不是本 assignment primary 頁。',
        '無正文圖頁面有真桌機／手機瀏覽，但未每頁存整頁截圖，不宣稱完成全頁視覺排版審核。',
        '作者修正後的同一讀者回讀尚未開始；三個原問題與初讀判定保留，等待集中修正後的明確受影響版本。',
    ],
    'artifacts': {
        'report_path': assignment['report_path'],
        'trace_path': assignment['trace_path'],
        'trace_sha256_at_report': sha(TRACE),
        'pageviews_path': str((ART / 'pageviews.jsonl').relative_to(ROOT)),
        'pageviews_sha256_at_report': sha(ART / 'pageviews.jsonl'),
        'artifact_directory': assignment['artifacts_directory'],
    },
}
path = ROOT / assignment['report_path']
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({
    'report_path': str(path.relative_to(ROOT)), 'report_sha256': sha(path),
    'primary_page_count': len(primary), 'pass': report['primary_pass_count'],
    'revise': report['primary_revise_count'], 'initial_units': len(initial),
    'read_unique_pages': len(read_ids), 'pageviews': len(pageviews),
    'screenshots': report['visual_review']['screenshot_count'],
    'issues': [i['id'] for i in issues], 'completed_at': now,
}, ensure_ascii=False))
