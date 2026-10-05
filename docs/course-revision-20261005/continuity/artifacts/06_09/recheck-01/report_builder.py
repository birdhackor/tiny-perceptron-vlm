import datetime
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path('/workspace/tiny-perceptron-vlm')
BASE = ROOT / 'docs/course-revision-20261005/continuity'
ART = BASE / 'artifacts/06_09/recheck-01'
TRACE = BASE / 'traces/06_09-recheck-01.jsonl'
REPORT = BASE / 'reports/06_09-recheck-01.json'
INVENTORY = BASE / 'revised-01/inventory.json'
INITIAL_REPORT = BASE / 'reports/06_09.json'
rows = [json.loads(line) for line in TRACE.read_text().splitlines()]
inventory = {p['page_id']: p for p in json.loads(INVENTORY.read_text())['pages']}
initial = json.loads(INITIAL_REPORT.read_text())
protection = json.loads((ART / 'initial-files-protection.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

changed = ['6.1', '6.2', '6.4', '6.5', '6.6', '6.9', '7.1', '7.4', '7.5', '7.8', '7.10', '7.11', '7.13', '7.14', '7.15'] + [f'9.{i}' for i in range(1, 11)]
assert [r['page_id'] for r in rows if r['scope'] == 'changed_primary'] == changed
assert len(rows) == 50 and len({r['page_id'] for r in rows}) == 46
for path, expected in protection.items():
    assert sha(pathlib.Path(path).read_bytes()) == expected, path

by_page = {}
source_integrity = []
for row in rows:
    by_page.setdefault(row['page_id'], []).append(row)
for page, page_rows in by_page.items():
    entry = inventory[page]
    snapshot = (ROOT / entry['snapshot']).read_bytes()
    assert sha(snapshot) == entry['source_sha256']
    source = (ROOT / entry['source']).read_text()
    parts = re.split(r'(?=^## )', source, flags=re.M)
    selector = entry['selector']
    if selector == 'whole-file':
        selected = source
    elif selector == 'before-first-section':
        selected = parts[0]
    else:
        section = selector.split(':', 1)[1]
        selected = next(p for p in parts if re.match(r'^## ' + re.escape(section) + r'(?:\s|$)', p))
    actual = sha(selected.encode())
    assert actual == entry['source_sha256']
    assert all(r['source_sha256'] == actual for r in page_rows)
    source_integrity.append({'page_id': page, 'source': entry['source'], 'selector': selector, 'snapshot': entry['snapshot'], 'source_sha256': actual, 'frozen_snapshot_matches': True, 'live_selected_source_matches': True, 'checked_at': now})

view_records = [json.loads(line) for line in (ART / 'pageviews.jsonl').read_text().splitlines()]
viewed = {}
for row in rows:
    for filename in row.get('visual_review', {}).get('viewed_screens', []):
        assert filename not in viewed
        viewed[filename] = row['utc']
assert len(viewed) == 31
for view in view_records:
    filename = pathlib.Path(view['screenshot_path']).name
    assert filename in viewed
    assert sha(pathlib.Path(view['screenshot_path']).read_bytes()) == view['screenshot_sha256']
    figure_path = 'course/' + view['figure_src'].split('/figures/', 1)[1]
    figure_path = 'course/figures/' + view['figure_src'].split('/figures/', 1)[1]
    expected = inventory[view['page_id']]['figures_sha256'][figure_path]
    assert view['rendered_resource_sha256'] == expected
    assert sha((ROOT / figure_path).read_bytes()) == expected
    view.update({'viewed': True, 'viewing_method': 'tools.view_image of the actual viewport screenshot', 'view_completed_before_trace_utc': viewed[filename], 'frozen_figure_sha256': expected, 'rendered_resource_matches_current_inventory': True, 'screenshot_sha256_verified_at': now})
assert len(view_records) == 31
# This file belongs to the recheck only. Initial pageviews remain untouched.
(ART / 'pageviews.jsonl').write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in view_records))

def unit_text(row):
    source = (ROOT / inventory[row['page_id']]['snapshot']).read_text()
    if row['page_id'] != 'glossary':
        return source
    parts = re.split(r'(?=^## )', source, flags=re.M)
    if row['section'] == '導言':
        return parts[0]
    return next(p for p in parts if p.startswith('## ' + row['section'] + ' '))

def quote(row):
    text = unit_text(row)
    body = re.sub(r'^#+[^\n]*\n', '', text, count=1).strip()
    return body.split('\n\n')[0]

actual_order = []
for index, row in enumerate(rows, 1):
    actual_order.append({'order': index, 'page_id': row['page_id'], 'section': row['section'], 'completed_read_utc': row['utc'], 'scope': row['scope'], 'source_sha256': row['source_sha256'], 'unit_sha256': sha(unit_text(row).encode()), 'figures_sha256': row['figures_sha256']})

pages = []
for page in changed:
    entry = inventory[page]
    row = by_page[page][0]
    screenshots = [v for v in view_records if v['page_id'] == page]
    pages.append({'page_id': page, 'source': entry['source'], 'selector': entry['selector'], 'snapshot': entry['snapshot'], 'source_sha256': entry['source_sha256'], 'figures_sha256': entry['figures_sha256'], 'read_completed_utc': row['utc'], 'verdict': row['verdict'], 'reason': row['current_understanding'], 'previous_teaching_actually_read': row['previous_actually_read'], 'new_question_and_need': row['new_question_and_need'], 'material_model_vocabulary_evaluation_switch': row['switching'], 'original_issue_ids': row.get('original_issue_ids', []), 'resolution': row.get('resolution'), 'current_doubts_preserved': row['current_doubts'], 'visual_review': {'status': 'actual_desktop_and_mobile_screenshots_viewed' if screenshots else 'no_authored_figure; source_text_checked; browser_layout_not_checked', 'viewports': [{'width': 1280, 'height': 900}, {'width': 390, 'height': 844}] if screenshots else [], 'screenshots': [v['screenshot_path'] for v in screenshots], 'requires_unseen_material_to_understand': False}})

context = []
for page, page_rows in by_page.items():
    if page in changed:
        continue
    entry = inventory[page]
    context.append({'page_id': page, 'source_sha256': entry['source_sha256'], 'figures_sha256': entry['figures_sha256'], 'scope': [r['scope'] for r in page_rows], 'read_sections': [r['section'] for r in page_rows], 'read_completed_utc': [r['utc'] for r in page_rows], 'verdict': 'pass', 'reason': [r['current_understanding'] for r in page_rows], 'current_doubts_preserved': [r['current_doubts'] for r in page_rows]})

boundaries = []
for previous, row in zip(rows, rows[1:]):
    gap = row.get('scope_gap_before', [])
    boundaries.append({'from': {'page_id': previous['page_id'], 'section': previous['section']}, 'to': {'page_id': row['page_id'], 'section': row['section']}, 'completed_read_utc': row['utc'], 'scope': row['scope'], 'is_adjacent_in_current_assignment_route': not bool(gap), 'unreread_gap': gap, 'task_at_this_point': row['current_understanding'], 'previous_teaching_actually_read': row['previous_actually_read'], 'new_question_and_why_needed': row['new_question_and_need'], 'switching_statement': row['switching'], 'switching_is_clear': row['page_id'] != '10.3', 'switching_limit': 'scene()參數與同圖身份未解釋；形狀主線可跟，不阻斷理解' if row['page_id'] == '10.3' else None, 'source_quote_for_entry': quote(row), 'current_doubts_preserved': row['current_doubts'], 'issue_ids': row.get('original_issue_ids', []) + row.get('new_issue_ids', []), 'missing_bridge': ['同圖呼叫參數關係可明說'] if row['page_id'] == '10.3' else [], 'verdict': row['verdict'], 'source_sha256': row['source_sha256'], 'figures_sha256': row['figures_sha256']})

dispositions = {
 'C06-01': {'status': 'resolved', 'actual_reread_pages': ['7.11', '7.12', '7.13', '7.17'], 'current_quote': '最後用同一組留出題比較，成績見[7.17](#7.17)；下一節[7.12](#7.12)先用四格材料說明題目家族的切分規則。', 'reader_reason': '7.11當下已分清下一節任務與延後結果。7.17真讀到同45題文字模型微調前1/10、後7/10、直接SFT5/10、多250步；未用外部報告補目的地。', 'deferred_question_preserved_at': by_page['7.11'][0]['utc'], 'deferred_question_resolved_at': by_page['7.17'][0]['utc']},
 'C06-02': {'status': 'resolved', 'actual_reread_pages': ['7.13', '7.14', '7.15'], 'current_quote': '我們已從[7.13的直接SFT屬性基模](#7.13)各複製一份，做這個比較。', 'reader_reason': '7.13先真讀到sft/model.pt從零900步test5/10、不是pretrain-sft.pt。7.14改指7.13並重述起點900步；7.15也明說用7.13保留基模。'},
 'C06-03': {'status': 'resolved', 'actual_reread_pages': ['7.4', '7.5'], 'current_quote': 'print("問題位置logits梯度", logits.grad[0, 2].norm().item())', 'reader_reason': '正文Q索引2/ID89與兩個程式梯度選取一致；沒有從問題格暗換成BOS格。'},
 'C06-04': {'status': 'resolved', 'actual_reread_pages': ['6.5', '9.10'], 'current_quote': '同批14段單音／答案不變，信心變大（新圖可見標題）', 'reader_reason': '兩新補圖在正文內311px手機寬度實看可讀條件、數值、分母、EOS或驗證選溫與作用範圍；各桌機長圖上下皆截取實看。未以放大原圖補可讀性。', 'viewed_screens': [pathlib.Path(v['screenshot_path']).name for v in view_records if v['page_id'] in ['6.5', '9.10']]},
 'C06-05': {'status': 'resolved', 'actual_reread_pages': ['9.1'], 'current_quote': '偏好標籤問哪個回答相對更好，安全標籤問各自是否通過規則，兩者不能互相代替。', 'reader_reason': '正文用欄位與例子交代相對/絕對。外部PKU來源、授權、裁短重查僅在補充一次，不再原樣整段貼正文。'},
 'C06-06': {'status': 'resolved_in_requested_locations', 'actual_reread_pages': ['7.10', '9.3', '9.4', '9.7'], 'current_quote': '應用也不能把引用文字生成的操作當作權限；這裡只回答顏色，不呼叫工具。', 'reader_reason': '相鄰整段重述已改成短提醒或新增資訊。有效數≠完整回答、真/假前提、可信permission與正式資料缺owner、人工target≠模型泛化仍保留。'},
 'C06-07': {'status': 'resolved_in_reread_locations', 'actual_reread_pages': ['6.1', '6.2', '6.6', '6.9', '7.5', '7.8', '7.13', '7.14', '7.15'] + [f'9.{i}' for i in range(1,10)], 'current_quote': 'BPE可登記整串特殊項，仍須另外決定普通內容相同拼寫怎麼編碼。', 'reader_reason': '本次回讀作者敘述已統一為繁體，表示/干擾等修正；真資料字串與實測輸出仍按原文讀。沒有聲稱已檢查全書。'},
 'C06-08': {'status': 'retained_optional', 'actual_reread_pages': ['glossary:G.4'], 'current_quote': '模型與介面規定的可用輸入範圍；能放入多少，與實際用好多少要分開看', 'reader_reason': 'current導航未改，初讀共同序列預算提醒仍保留optional；此次未重讀7.19/14.7，不把任一有限介面契約推成通則。'},
 'C06-09': {'status': 'retained_optional_not_reread_at_origin', 'actual_reread_pages': ['9.8'], 'current_quote': '接著複製[8.3的同一份加法基模](08.md#8.3)，訓練兩個版本。', 'reader_reason': '本次未重讀8.3或8.8。9.8依然足以理解其內部兩支同起點0/7，未知它與7.15身份關係仍不補造。原optional由root保留，不宣稱解開。'},
 'C06-10': {'status': 'resolved_at_original_location', 'actual_reread_pages': ['chapter-10', '10.1', '10.2', '10.3'], 'current_quote': 'image = scene("red", "square")[None]', 'reader_reason': '10.2直接沿用10.1同紅方塊，不需查scene預設。下一節10.3重新省參數形成另一个非阻斷optional，另列R06-01，不覆蓋本項初疑。'},
}
original_issues = []
for issue in initial['issues']:
    original_issues.append({'initial_issue_preserved': issue, 'current_disposition': dispositions[issue['id']]})

new_issue = {'id': 'R06-01', 'pages': ['10.3'], 'scope': 'context_after; first actual reading of 10.3 in this recheck', 'severity': '選讀改善', 'recommended_action': 'optional', 'location': '10.2→10.3程式素材交接', 'original_quote': 'patches = patchify(scene()[None], 4)', 'encountered_at_page': '10.3', 'encountered_trace_utc': by_page['10.3'][0]['utc'], 'then_understood': '上一頁已明示同黑底紅方塊scene(red,square)。這頁承接每塊48值，卻省參數；正文沒教scene()預設，不能自行聲稱圖色仍相同。48→8形狀推導不依色，仍能跟主線。', 'can_answer_main_question_from_then_read_content': True, 'missing_bridge': '省略呼叫參數與上一節同圖的關係未說；未阻斷位置數/特徵數的學習。', 'suggestion': '若意圖同圖，明用scene("red","square")；或說明改用獨立shape示範。', 'implementation_or_external_knowledge_used': False, 'after_read_author_response': {'status': 'parent acknowledged the optional and chose not to change source at this stage', 'effect_on_reader_record': '作者讀後處置與原理解分開；未以作者的實作確認改寫本疑問，未讀實作。'}}

initial_primary_ids = [p['page_id'] for p in initial['primary_pages']]
unreread_primary = [p for p in initial_primary_ids if p not in by_page]
figure_checks = []
for figure in sorted({v['figure_src'] for v in view_records}):
    same = [v for v in view_records if v['figure_src'] == figure]
    figure_checks.append({'figure_url': figure, 'figure_sha256': same[0]['rendered_resource_sha256'], 'live_file_and_browser_resource_match_current_inventory': True, 'viewed_desktop': any(v['viewport'] == {'width':1280,'height':900} for v in same), 'viewed_mobile': any(v['viewport'] == {'width':390,'height':844} for v in same), 'screenshots': [v['screenshot_path'] for v in same]})

report = {
 'schema_version': 2,
 'stage': 'round_3_continuity_current_recheck',
 'revision': 'revised-01',
 'assignment': '06_09',
 'status': 'completed',
 'reviewer_task': '/root/continuity_06_09',
 'fork_turns': 'none',
 'reviewer_background': '入門Python與基本高中數學；只用已讀正文理解，不用模型專家知識或實作補橋梁。',
 'independence': {'same_actual_initial_reviewer': True, 'other_reviewer_or_author_work_notes_read': False, 'implementation_read_or_run': False, 'external_explanations_read': False, 'subagents_spawned': False, 'training_or_gpu_used': False, 'parent_disposition_received_after_optional_record': '10.3讀後作者處置獨立標記，未用來修改原理解。'},
 'first_completed_read_at': rows[0]['utc'],
 'last_completed_read_at': rows[-1]['utc'],
 'report_written_at': now,
 'method_paths': ['docs/review-tools/continuity-reviewer-instructions.md', '.agents/skills/clear-tutorial/references/review-protocol.md'],
 'inventory_path': str(INVENTORY.relative_to(ROOT)),
 'inventory_sha256': sha(INVENTORY.read_bytes()),
 'initial_report_path': str(INITIAL_REPORT.relative_to(ROOT)),
 'initial_files_unchanged': [{'path': p, 'sha256': h, 'matches_initial_protection': True} for p,h in protection.items()],
 'trace_path': str(TRACE.relative_to(ROOT)),
 'trace_sha256': sha(TRACE.read_bytes()),
 'artifacts_directory': str(ART.relative_to(ROOT)),
 'actual_read_order': actual_order,
 'actual_read_scope_statement': '真回讀25個指定changed primary、17個必要未改primary及4個after-context，共46頁50節；glossary按導言/G.1/G.2/G.3/G.4逐節先記再讀。chapter6→7.18保原順序，再明示跳過未改7.19/chapter8/8.1–8.15，讀8.16→chapter9→9.1–9.10→glossary→chapter10/10.1/10.2/10.3。沒有假裝全部59主頁重讀。',
 'counts': {'changed_primary_pages': 25, 'changed_primary_pass': 25, 'changed_primary_revise': 0, 'unchanged_primary_pages_actually_reread': 17, 'after_context_pages_actually_read': 4, 'unique_pages_actually_reread': 46, 'completed_read_units': 50, 'boundaries': 49, 'necessary_figure_resources_viewed': 13, 'real_browser_screenshots_viewed': 31, 'new_optional_issues': 1},
 'overall_verdict': 'pass_with_retained_and_new_optionals',
 'overall_reason': '原需修的梯度對象、成績目的地、基模起點、手機補圖小字與9.1整段重複皆已實際回讀解開。原指定optional的重複與作者繁體也在回讀位置改善。glossary共同預算導航與8.3身份仍保留optional；新10.3scene()同圖身份未說為非阻斷optional。',
 'changed_primary_pages': pages,
 'necessary_previous_next_and_after_context': context,
 'boundaries': boundaries,
 'original_issue_dispositions': original_issues,
 'new_issues': [new_issue],
 'pageviews': view_records,
 'figure_integrity_checks': figure_checks,
 'source_integrity_checks': source_integrity,
 'evidence_categories': {'hand_calculation': ['6.5不同token總NLL與原文byte', '7.6人工候選loss均值', '7.14四候選錯標', '9.10四題兩箱與新增低信心答對ECE'], 'interface_or_data_demonstration_not_training': ['6.2/6.3 tokenizer合併與編碼', '6.6字面控制標記', '6.7切片解碼', '6.8窗口shift', '7.1–7.4資料/prompt/label', '7.5/7.11只求梯度未step', '7.7 PAD前向等價示範', '7.8人工分數索引', '7.9手寫停止序列', '9.1–9.7人工欄位/目標或判斷', '10.1–10.3合成圖與形狀/排序/Linear'], 'historical_measured_results_read_not_independently_reproduced': ['6.5兩tokenizer400更新共同raw-byte', '7.10UltraChat80次獨立模型', '7.13直接SFT900起點5/10', '7.14正標/四錯標300對照', '7.15/7.16 B-only/replay500', '7.17文字250+SFT900', '9.2/9.3/9.4/9.6/9.8行為兩版與凍結改問結果', '9.9/9.10圖片與單音分類校準'], 'planned_or_placeholder_not_results': ['7.15 measurements的None', '7.16 recipe只是預算', '9.7新措辭待模型測試', '9.8多輪情境提案'], 'actual_browser_observation': '31張正文viewport截圖經view_image實看；頁面的已核CPU輸出僅作頁面可見內容，不獨立重跑長實驗。'},
 'unchecked_scope': {'original_primary_not_reread_current': unreread_primary, 'original_before_context_not_reread_current': ['5.16', '5.17'], 'implementation_training_recipes_external_sources_or_results_not_read': True, 'historical_results_not_independently_reproduced': True, 'nonfigure_page_browser_layout_not_checked': [p for p in by_page if not inventory[p]['figures_sha256']], '10.4_and_later_after_context_not_read': True, 'retained_optional_identity_not_fabricated': '8.3與7.15起點關係未回核或補造；9.8只依正文已知其兩支同起點。'},
 'recording_clarifications': ['每節完成立即追加自己的UTC trace才讀下一節，原initial trace/report未改。', '7.11當下只解入口承諾，7.17目的地真讀到後另外保留解除位置，未倒填7.11已看成績。', '6.5/9.10 body unchanged不是未回讀；圖新SHA、正文嵌圖實看與讀者判斷均獨立保存。', '10.3是本次after-context首次真讀，不冒稱初讀59頁中讀過10.3。', '31張截圖初由capture記viewed:false，實際view_image後的完成證據來自每節trace；本recheck pageviews才標true，未改初讀檔案。', '只有本assignment recheck artifacts/report/trace被寫入，未改教材。'],
}
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
audit = {'utc': now, 'report_path': str(REPORT), 'report_sha256': sha(REPORT.read_bytes()), 'trace_path': str(TRACE), 'trace_sha256': sha(TRACE.read_bytes()), 'initial_files_unchanged': protection, 'changed_primary_pages': len(pages), 'read_units': len(rows), 'source_hash_matches': len(source_integrity), 'screenshots_viewed_and_hash_verified': len(view_records), 'figure_resources_verified': len(figure_checks)}
(ART / 'final-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(audit, ensure_ascii=False, indent=2))
