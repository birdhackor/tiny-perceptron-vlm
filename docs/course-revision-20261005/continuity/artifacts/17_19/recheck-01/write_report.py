import datetime
import hashlib
import json
from pathlib import Path

root = Path('/workspace/tiny-perceptron-vlm')
base = root / 'docs/course-revision-20261005/continuity'
artifact = base / 'artifacts/17_19/recheck-01'
old = json.loads((base / 'reports/17_19.json').read_text())
inventory = json.loads((base / 'revised-01/inventory.json').read_text())
pages = {p['page_id']: p for p in inventory['pages']}
traces = [json.loads(line) for line in (base / 'traces/17_19-recheck-01.jsonl').read_text().splitlines()]
pageviews = [json.loads(line) for line in (artifact / 'pageviews.jsonl').read_text().splitlines()]
initial_hashes = json.loads((artifact / 'initial-preservation.json').read_text())
initial_actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [base / 'reports/17_19.json', base / 'traces/17_19.jsonl']}

changed = ['17.9', '17.10', '19.2', '19.5', '19.7', '19.11', 'training', 'validation']
reasons = {
    '17.9': '實際回讀17.8→17.9→17.10→17.11，再跟到17.15；新「品質檢查與實報入口」承諾和17.15實際內容相符，C1715-01解決。',
    '17.10': '正文用「檔案」；無計時的單層儲存／浮點重建核對與既有L4量測範圍仍清楚，前後17.9轉換及17.11特徵誤差可接。',
    '19.2': '19.1→19.2→19.3實讀；新句明確expert之間共用零件，但輸入嵌入與輸出表各自儲存，無需等補充才消歧。',
    '19.5': '19.4→19.5→19.6實讀；「不是只靠」修正，人工示範／歷史模板／新任務規劃仍各有通知。',
    '19.7': '19.6→19.7→19.8實讀；「不是只差」修正，人工請求與真執行、模型第二段回答分明，真工具對而模型答錯仍保留。',
    '19.11': '19.10→19.11→19.12實讀；「四份階段／分支」與DPO非必經且推薦joint一致，推論檔／續訓工作檔／新成品待回填未混同。',
    'training': 'opening及T.1–T.11逐節整頁實讀；T.3列接字表及一／三／五字MLP四設定，T.10補固定joint命令及依賴／教師檔路徑，CT3-01與CT10-01解決。T.6暫留來源疑問保留於trace，由後讀T.10另澄清。',
    'validation': 'opening及全部六節逐節整頁實讀；第19章新任務工程／訓練／驗收尚未完成與已發布舊合成示範分开，再和第20章成熟權重路線分開，CV2-01解決。',
    '17.15': '實際補讀17.9連結目標全文，品質檢查及T.9／量化實報入口符合修後連結文字；本頁來源未改，由上游承諾修正解除C1715-01。',
}
issue_evidence = {
    'C1715-01': {
        'recheck_pages': ['17.9', '17.15', 'training'],
        'recheck_units': ['17.9', '17.15 link target', 'T.9'],
        'current_quote': '兩份packed檔重新載入後才評估，品質檢查與實報入口見17.15',
        'understanding': '現在預期品質核對方法和實報入口，而非17.15必給三版本完整成績。實讀17.15確有手設EOS排名反例、共同比較條件、T.9操作及量化實報入口。',
        'missing_bridge_after_recheck': None,
    },
    'CT3-01': {
        'recheck_pages': ['training'], 'recheck_units': ['T.3'],
        'current_quote': '固定比較包含兩類模型、四個設定：接字表，以及一字、三字、五字視窗的 MLP。',
        'understanding': '四個設定的名稱和兩類關係在正文操作入口直接交代，不會把before/after四次執行或兩個未命名架構當四種模型。',
        'missing_bridge_after_recheck': None,
    },
    'CT10-01': {
        'recheck_pages': ['training'], 'recheck_units': ['T.4', 'T.6', 'T.10', 'T.11'],
        'current_quote': '第一行是12.12的固定圖音聯合訓練，直接讀取 sft 與 encoders，不承接 vqa。這份 joint、T.6 手動保存的 checkpoints/joint.pt 和第19章的 capstone_joint 各有自己的配方，不能互換來源。',
        'current_command': '.venv/bin/python -m scripts.course_experiments.run --experiment joint --device cuda',
        'understanding': '固定sft先完成，再按T.6完成encoders/projector/vqa，補入joint命令後才跑multimodal_distillation。joint直接讀sft/encoders；工具不自動補依賴且第二行讀同course-v1根的vqa與joint各自model.pt和dataset.json。CPU手動joint與capstone_joint不可互換。',
        'missing_bridge_after_recheck': None,
        'scope_limit': '只判公開操作文本的來源／前置橋梁；不執行CUDA或訓練，不讀技術README或實作。12.12沿用本人真初讀的補讀理解，此輪沒有再讀該頁。',
    },
    'CV2-01': {
        'recheck_pages': ['validation', '19.1', '19.3', '19.6', '19.11', '19.12'],
        'recheck_units': ['小實驗、正式訓練與成品各有自己的範圍'],
        'current_quote': '這些是新成品的設計任務，工程、訓練與能力驗收尚未完成；已發布的舊合成示範只涵蓋簡單圖形、純音、固定句型與受限工具，其歷史成績不能代替新任務驗收。',
        'understanding': '第19章新商品／已知中文字卡／有限真人語音意圖計畫尚待工程與驗收，旧示範是歷史合成模型；第20章沿用成熟權重，是另一條路線，不統稱兩個已完成成品。',
        'missing_bridge_after_recheck': None,
    },
}

def unit(r):
    return r.get('unit', r.get('section'))

def understanding(r):
    return r.get('first_understanding', r.get('understanding'))

issues = []
for previous in old['issues']:
    evidence = issue_evidence[previous['id']]
    issues.append({
        'id': previous['id'], 'status': 'resolved', 'decision': 'pass',
        'original_quote': previous['quote'],
        'original_first_interpretation': previous['first_interpretation'],
        'original_missing_bridge': previous['missing_bridge'],
        'original_severity': previous['severity'],
        'original_suggestion': previous['suggestion'],
        'original_first_recorded_at': previous['first_recorded_at'],
        **evidence,
    })

actual_read_order = [{
    'sequence': i, 'page_id': r['page_id'], 'unit': unit(r),
    'recorded_at': r['recorded_at'], 'source_sha256': r['source_sha256'],
    'figures_sha256': r['figures_sha256'], 'source_snapshot': r['source_snapshot'],
} for i, r in enumerate(traces, 1)]
read_ids = list(dict.fromkeys(r['page_id'] for r in traces))
primary_pages = []
for original in old['primary_pages']:
    pid = original['page_id']; p = pages[pid]
    rows = [r for r in traces if r['page_id'] == pid]
    primary_pages.append({
        'page_id': pid, 'source': p['source'], 'source_snapshot': p['snapshot'],
        'source_sha256': p['source_sha256'], 'figures_sha256': p['figures_sha256'],
        'changed_in_revised_01': pid in changed,
        'reading_scope': 'actually_reread_in_recheck_01' if rows else 'carried_from_this_reviewers_true_initial_read_not_reread',
        'actual_units_reread': [unit(r) for r in rows],
        'actual_initial_units_read': original['actual_units_read'],
        'initial_source_sha256': original['source_sha256'],
        'initial_figures_sha256': original['figures_sha256'],
        'decision': 'pass' if rows or pid in reasons else original['decision'],
        'pass_or_revise_reason': reasons.get(pid, '必要上下文已實讀，逐節首次理解與通知句見本輪trace。' if rows else original['pass_or_revise_reason']),
    })
for p in primary_pages:
    if p['reading_scope'].startswith('carried_'):
        assert p['source_sha256'] == p['initial_source_sha256']
        assert p['figures_sha256'] == p['initial_figures_sha256']

boundary_records = [{
    'page_id': r['page_id'], 'unit': unit(r), 'recorded_at': r['recorded_at'],
    'prior_material': r['prior_material'], 'new_question': r['new_question'],
    'switch_notice': r['switch_notice'], 'first_understanding': understanding(r),
    'bridge_or_blocker': r.get('bridge_or_blocker', '無新問題，見當節understanding及issues。'),
    'decision': 'pass' if r['page_id'] == '17.9' else r['decision'],
    'decision_at_trace_time': r['decision'],
    'later_clarification': '後讀17.15符合承諾，C1715-01解決。' if r['page_id'] == '17.9' else ('後讀T.10明確固定joint命令／來源／檔路徑，CT10-01解決；保留T.6原暫疑。' if r['page_id'] == 'training' and unit(r) == 'T.6' else None),
    'source_sha256': r['source_sha256'], 'figures_sha256': r['figures_sha256'],
} for r in traces]

for p in pageviews:
    p['actually_viewed_with_view_image'] = True
    p['visual_evidence_scope'] = '實際正文URL與指定viewport之截圖；不是原圖放大、SVG字串或DOM尺寸推論。'

report = {
    'schema_version': 1, 'stage': 'continuity-recheck-01', 'assignment': '17_19',
    'reviewer_task': '/root/continuity_17_19', 'fork_turns': 'none',
    'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'completed', 'decision': 'pass',
    'same_actual_initial_reviewer': True,
    'reader_identity': '同一位完成48primary頁真初讀的第三輪讀者；只以高中基本數學／入門Python與本人實際已讀前文理解。本輪是指定變更回讀，並非新的盲讀。',
    'method_sources': old['method_sources'],
    'source_inventory': 'docs/course-revision-20261005/continuity/revised-01/inventory.json',
    'source_inventory_sha256': hashlib.sha256((base / 'revised-01/inventory.json').read_bytes()).hexdigest(),
    'source_freeze': 'revised-01/source-pages/<page_id>.md',
    'figure_freeze': '本輪inventory各頁figures_sha256；正文HTTP圖SHA另驗證，與inventory一致。',
    'initial_report': 'docs/course-revision-20261005/continuity/reports/17_19.json',
    'initial_trace': 'docs/course-revision-20261005/continuity/traces/17_19.jsonl',
    'initial_preservation': {'expected': initial_hashes, 'actual_sha256': initial_actual, 'byte_identical': initial_actual == {
        'docs/course-revision-20261005/continuity/reports/17_19.json': '44758e4e8dbf305047b2659de4d0a00dfc3d1aa66d71ffe9c05b2704973352d4',
        'docs/course-revision-20261005/continuity/traces/17_19.jsonl': 'd1eb4ed0d44ad9f40a34f91c907750cb0a99e1c198c39bb166c1c5dea53fc539'}},
    'trace_path': 'docs/course-revision-20261005/continuity/traces/17_19-recheck-01.jsonl',
    'artifacts_directory': 'docs/course-revision-20261005/continuity/artifacts/17_19/recheck-01/',
    'initial_primary_page_count': 48,
    'changed_primary_page_count': 8, 'changed_primary_page_ids': changed,
    'actual_reread_page_count': len(read_ids), 'actual_reread_page_ids': read_ids,
    'unit_count': len(actual_read_order), 'trace_record_count': len(traces),
    'whole_guide_page_units': {'training': 12, 'validation': 7},
    'scope_statement': '本輪只真回讀8個changed primary與必要上下文，合計18頁35單位；其餘30頁未回讀，沿用本人先前真初讀且已核對SHA未改。初讀48頁的紀錄保持原樣。不是宣稱48頁全部再讀。',
    'necessary_context': ['17.8–17.11', '19.1–19.3', '19.4–19.8', '19.10–19.12'],
    'supplemental_context': [{'page_id': '17.15', 'reason': '17.9新版品質檢查／實報入口連結必須真跟到目標全文確認。', 'actually_reread': True}],
    'previous_context_retained': [{'page_id': '12.12', 'actually_reread_this_round': False, 'basis': '本人初讀18.13引用時實際補讀12.12，原疑保留在初讀trace；本輪T.10公開正文已直接補命令及三份joint身份。'}],
    'actual_read_order': actual_read_order, 'primary_pages': primary_pages,
    'changed_primary_decisions': [{k: p[k] for k in ['page_id', 'source_sha256', 'figures_sha256', 'actual_units_reread', 'decision', 'pass_or_revise_reason']} for p in primary_pages if p['page_id'] in changed],
    'boundary_records': boundary_records, 'original_issue_resolutions': issues,
    'new_issues': [],
    'prevnext_and_limits': [
        {'scope': '17.8→17.9→17.10→17.11，另17.9→17.15', 'decision': 'pass', 'understanding': '打包無損而先量化有誤差→共同父檔轉換→檔案／還原／速度口徑→特徵誤差。品質檢查連結真到目標，不需外部技術報告補教材承諾。'},
        {'scope': '19.1→19.2→19.3', 'decision': 'pass', 'understanding': '新有限自訓整合設計→expert共用零件但兩表分開→素材家族／新原始ID切分。隨機架構例與歷史trained MoE不混。'},
        {'scope': '19.4→19.5→19.6→19.7→19.8', 'decision': 'pass', 'understanding': '父權重→示範行為→圖聲條件→真工具往返→偏好比較分支；局部示範／舊實測／新計畫各有通知。未再讀19.9，初讀理解保留。'},
        {'scope': '19.10→19.11→19.12', 'decision': 'pass', 'understanding': 'joint量化／DPO teacher學生→11份正式推論交付及另續訓角色→能力矩陣待填與舊實測分母；DPO不是推薦成品必經更新。'},
        {'scope': 'training opening→T.1–T.11', 'decision': 'pass', 'understanding': '整頁實讀，CPU自訓與固定配方不同。T.10补 fixed joint 操作來源後能接T.11可核對紀錄，沒有把T.6 CPU joint或19capstone拿來代用。'},
        {'scope': 'validation opening→全部六節', 'decision': 'pass', 'understanding': '驗證各層不同；新19任務待工程/訓練/驗收，舊19已發布合成示範與20成熟延伸分開。後文選版/雲端/重跑保持各自證據范围。'},
    ],
    'pageviews': pageviews,
    'figure_assessment': {
        'decision': 'pass', 'actual_figure_pages': ['17.8', '19.1', '19.6', '19.7'],
        'figure_count': 4, 'screenshot_count': len(pageviews),
        'desktop_viewport': {'width': 1280, 'height': 900}, 'mobile_viewport': {'width': 390, 'height': 844},
        'method': 'Chromium/Playwright真正文URL；viewport截圖，必要長圖另下半截；10份截圖全部實際view_image，19.7desktop bottom在其後單獨實看。',
        'judgments': {
            '17.8': '桌機／手機能讀高低半byte、合112與拆回次序；桌機長圖另截下半，不以圖放大替正文。',
            '19.1': '三入口接同MoE及工具返回可讀；圖標整合設計／神經权重自訓，不代表能力已驗收。',
            '19.6': '上入口／下出口／指定下框与答案可讀，是待驗收任務圖。',
            '19.7': '五框分模型首段請求、程序真算、回填與模型第二段；能辨真工具正確後模型仍可能答錯。',
        },
        'served_figure_hash_verification': json.loads((artifact / 'figure-hash-verification.json').read_text()),
        'missing_necessary_figure': False,
        'limits': '其他回讀頁沒有必要圖；未另驗無圖頁的全版排版。root所述HTML parity未當作本人圖易讀證據。',
    },
    'optional_improvements': [{
        'page_id': '17.10', 'severity': '可選字詞同步', 'decision': 'pass',
        'quote': '側欄仍顯示「17.10 文件變小，為什麼運算不一定更快？」',
        'first_understanding': '正文已改檔案，側欄文件仍能理解其指儲存檔。',
        'missing_bridge': None,
        'suggestion': '可同步網站導覽標題的「文件」為「檔案」。',
        'evidence': '實際17.8 desktop正文截圖側欄：17.8-desktop-0-main.png。',
    }],
    'unverified_scope': [
        '沒有回讀所有48primary；18頁真回讀、其餘30頁沿用本人真初讀且SHA未改。',
        '沒有把AI回讀當作真人高中生學習成效實測。',
        '沒有讀作者repair manifest／筆記、其他reviewer、技術實報、來源實作或外部解釋。',
        '沒有執行下載、GPU、訓練、全庫單元測試或估閱讀時間；CLI僅判公開操作文字前置是否明確。',
        '沒有重新驗證历史GPU能力／速度／續訓數值，正文有歷史範圍不等於本人重測。',
        '沒有驗收第19章新商品／12字卡／真人語音意圖成品；它在修後文字中仍為尚待工程、訓練及驗收。',
        '沒有重新打開W.1／W.7、20.8／20.12、12.12或外部資料與模型卡；只判連結標示是否承接正文問題。',
    ],
    'final_reader_restatement': '四個原理解橋梁問題已由公開正文解決，八個changed primary皆pass。新成品計畫、局部示範和既有合成实测可分辨；固定joint與三份不同joint身份、CLI前置和教師檔路徑已明確。只剩側欄字詞可選同步，不需教材再次改寫或重訓。',
    'rereview_required': False,
}
assert report['initial_preservation']['byte_identical']
assert len(traces) == 35 and len(read_ids) == 18
assert len(primary_pages) == 48
assert all(p['decision'] == 'pass' for p in report['changed_primary_decisions'])
(base / 'reports/17_19-recheck-01.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('wrote', base / 'reports/17_19-recheck-01.json')
