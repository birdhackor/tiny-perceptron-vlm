from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

repo = Path('/workspace/tiny-perceptron-vlm')
base = repo / 'docs/course-revision-20261005/continuity'
artifact = base / 'artifacts/10_12/recheck-01'
inventory_path = base / 'revised-01/inventory.json'
inventory = json.loads(inventory_path.read_text())
pages = {p['page_id']: p for p in inventory['pages']}
trace_path = base / 'traces/10_12-recheck-01.jsonl'
trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
order = ['9.8', '9.9', '9.10', 'chapter-10', '10.1', '10.2', '10.3']
assert [entry['page_id'] for entry in trace] == order
assert all(entry['decision'] == 'pass' for entry in trace)
by_id = {entry['page_id']: entry for entry in trace}

for entry in trace:
    page = pages[entry['page_id']]
    assert entry['source_sha256'] == page['source_sha256']
    assert entry['figures_sha256'] == page['figures_sha256']
    assert hashlib.sha256((repo / page['snapshot']).read_bytes()).hexdigest() == page['source_sha256']
    for figure, sha in page['figures_sha256'].items():
        assert hashlib.sha256((repo / figure).read_bytes()).hexdigest() == sha

raw_views = [json.loads(line) for line in (artifact / 'pageviews.jsonl').read_text().splitlines()]
assert len(raw_views) == 12
confirmed = {filename: entry for entry in trace for filename in entry.get('pageviews', [])}
assert set(confirmed) == {Path(v['screenshot']).name for v in raw_views}
pageviews = []
for view in raw_views:
    filename = Path(view['screenshot']).name
    assert hashlib.sha256(Path(view['screenshot']).read_bytes()).hexdigest() == view['screenshot_sha256']
    entry = confirmed[filename]
    assert entry['page_id'] == view['page_id']
    assert view['viewport'] in [{'width': 1280, 'height': 900}, {'width': 390, 'height': 844}]
    view = dict(view)
    view['viewed'] = True
    view['view_method'] = 'tools.view_image：逐張實看正文 viewport 截圖；未以 alt、SVG 文字、DOM 幾何或原圖放大代替。'
    view['view_confirmed_in_trace_at_utc'] = entry['timestamp_utc']
    view['confirmation_timestamp_meaning'] = '這是看完後追加該節紀錄的 UTC，沒有冒稱工具顯示的精確觀看瞬間。'
    pageviews.append(view)
(artifact / 'pageviews-final.json').write_text(json.dumps(pageviews, ensure_ascii=False, indent=2) + '\n')

primary_reread = ['chapter-10', '10.1', '10.2', '10.3']
not_reread = ([f'10.{n}' for n in range(4, 12)]
             + ['chapter-11'] + [f'11.{n}' for n in range(1, 19)]
             + ['chapter-12'] + [f'12.{n}' for n in range(1, 17)]
             + ['training-assets'])
assert len(not_reread) == 45
assert len(primary_reread) + len(not_reread) == 49

boundaries = [
    {'from': '9.8', 'to': '9.9', 'understanding': '9.8 的原樣模板高分不能外推改寫問題；9.9 再問候選機率看起來肯定是否真的可靠。', 'decision': 'pass', 'reason': '任務範圍限制先交代，再引入信心問題，不把手寫字串差異当成模型實測。'},
    {'from': '9.9', 'to': '9.10', 'understanding': '9.9 拉大 logits 可讓錯誤答案更肯定；9.10 以多個獨立問題按信心分箱，對照命中率。', 'decision': 'pass', 'reason': '四筆手算與追加低信心正確例完整列出資料、分箱與數值，校準的範圍可跟上。'},
    {'from': '9.10', 'to': 'chapter-10', 'understanding': '9.10 收束答案／信心／校準不能混為一談；第十章导讀明白改做紅色方塊的像素、patch、位置與文字介面。', 'decision': 'pass', 'reason': '新素材與新問題有通知；前章分類器歷史分數沒有變成新示範已會看圖的證據。'},
    {'from': 'chapter-10', 'to': '10.1', 'understanding': '導讀中的黑底紅方塊，在10.1 明確成為16×16、RGB三通道和batch一張。', 'decision': 'pass', 'reason': 'scene("red", "square")、中心／角落RGB與正文圖足以定位素材；未讀實作也可建立形狀。'},
    {'from': '10.1', 'to': '10.2', 'understanding': '本次10.2 明說沿用上一節16×16黑底紅色方塊，使用相同scene參數，切成16個各48值的patch。', 'decision': 'pass', 'reason': '不再需要猜scene()預設素材；patch位置0–15及切拼測試的證據限制仍明確。'},
    {'from': '10.2', 'to': '10.3', 'understanding': '10.2 切塊保留全部像素，節尾問壓短會否丟差異；10.3 以共享48→8權重及兩輸入撞成同值回答。', 'decision': 'pass', 'reason': '位置數16與特徵寬度8／16仍分開；隨機初始化與訓練能力沒有混稱。'}
]

changed = [
    {'page_id': '9.9', 'role': '初讀時實際讀過的context_before，本次改動頁', 'decision': 'pass',
     'reason': by_id['9.9']['reason'],
     'original_problem': {'id': '10_12-context-9.9-script-consistency', 'severity': 'optional', 'observed_fragments': ['颜色', '绿', '编'], 'initial_understanding_preserved': '初讀已理解softmax／温度不改錯誤候選排名；這是繁簡一致性建議，並非概念卡點。'},
     'resolution': '本次真回讀看到作者說明改用顏色、綠、編等繁體，原可選問題已解。資料字串仍按正文保留，沒有重寫初讀紀錄。',
     'necessary_previous': ['9.8'], 'necessary_next': ['9.10'], 'new_required_revision': False},
    {'page_id': '9.10', 'role': '初讀時實際讀過的context_before，本次改動頁', 'decision': 'pass',
     'reason': by_id['9.10']['reason'],
     'original_problem': {'severity': 'readability_observation_nonblocking', 'initial_understanding_preserved': '初讀已通過校準敘述，當時手機補充圖字偏小；沒有把這項觀察倒寫成初讀失敗。'},
     'resolution': '本次新音訊圖在390×844正文截图可直接讀8筆驗證／14筆測試、11/14命中、T=1與T=0.5、信心百分比、ECE與Brier；兩種viewport均逐張查看。圖仍清楚表達答案不變但信心變大，沒有把平均棒差當ECE。',
     'necessary_previous': ['9.8', '9.9'], 'necessary_next': ['chapter-10'], 'new_required_revision': False},
    {'page_id': '10.2', 'role': '49個primary中本次唯一改動頁', 'decision': 'pass',
     'reason': by_id['10.2']['reason'],
     'original_problem': {'severity': 'none_from_this_reviewer', 'initial_understanding_preserved': '我的初讀10.2為pass，未提出必修卡點；本次不把其他讀者的問題借記成自己的首次理解。'},
     'resolution': '實際確認開頭「沿用上一節的16×16黑底紅色方塊圖」以及image = scene("red", "square")[None]，明白通知同一素材。切塊、patch順序、切拼可還原不等於順序正確，以及下一節壓縮問題仍能跟上。',
     'necessary_previous': ['chapter-10', '10.1'], 'necessary_next': ['10.3'], 'new_required_revision': False}
]

report = {
    'schema_version': 1,
    'reviewer_task': '/root/continuity_10_12',
    'fork': 'none',
    'round': 'recheck-01',
    'assignment': '10_12',
    'same_actual_reviewer': True,
    'reviewer_background': '只以入門Python與高中數學理解正文；本次保留自己的真初讀記憶，不是假裝首次全書閱讀。',
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'decision': 'pass',
    'required_revision_count': 0,
    'issues': [],
    'inventory_path': str(inventory_path.relative_to(repo)),
    'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    'source_version': 'revised-01凍結正文與圖；只報告，未修改教材。',
    'method_paths': ['docs/review-tools/continuity-reviewer-instructions.md', '.agents/skills/clear-tutorial/references/review-protocol.md'],
    'actual_read_order': order,
    'actual_read_order_detail': [{'page_id': e['page_id'], 'scope': e['scope'], 'unit': e['unit'], 'timestamp_utc': e['timestamp_utc']} for e in trace],
    'newly_read_background': ['9.8'],
    'reread_original_context_before': ['9.9', '9.10'],
    'primary_actually_reread': primary_reread,
    'changed_primary_actually_reread': ['10.2'],
    'changed_scope_pages': changed,
    'pages': [{'page_id': e['page_id'], 'source_sha256': e['source_sha256'], 'figures_sha256': e['figures_sha256'], 'snapshot': e['snapshot'], 'decision': e['decision'], 'reason': e['reason'], 'understanding': e['understanding'], 'original_issue_resolution': e['original_issue_resolution']} for e in trace],
    'boundaries': boundaries,
    'trace_path': str(trace_path.relative_to(repo)),
    'trace_entry_count': len(trace),
    'trace_entries': trace,
    'pageviews': pageviews,
    'pageview_count': len(pageviews),
    'unique_figures_actually_seen': 4,
    'figure_viewports': [{'width': 1280, 'height': 900}, {'width': 390, 'height': 844}],
    'unchanged_primary_not_reread': [{'page_id': p, 'status': '沿用真初讀pass，非本次新閱讀／新驗證'} for p in not_reread],
    'unchanged_primary_not_reread_count': len(not_reread),
    'initial_review_preserved': {
        'report_path': 'docs/course-revision-20261005/continuity/reports/10_12.json',
        'trace_path': 'docs/course-revision-20261005/continuity/traces/10_12.jsonl',
        'changed': False,
        'basis': '初讀真完成49個primary，這次只回讀其中4頁；其餘45頁沿用初讀結論，沒有重讀或冒稱新全49頁閱讀。'
    },
    'not_rechecked': {
        'original_context_after': ['chapter-13', '13.1', '13.2'],
        'original_additional_background': ['7.3'],
        'initial_optional_note': '11.6的問答SFT縮寫範圍觀察未在這次回讀，不宣稱已解，也不以未讀全第七章補理解。',
        'other_pages': '沒有跟讀本次七頁的其他正文連結。'
    },
    'evidence_limits': [
        '本次不是重新盲讀全書；初讀理解／卡點未補寫、未改寫。',
        '9.8的字串比較與9.9／9.10的手算是教材示範；補充中的模型成績是正文標示的歷史實測，本次沒有跑模型或重新實測。',
        '10.1至10.3的像素、切塊與隨機共享權重是機制示範，沒有把切拼True或初始特徵當已學會看圖。',
        '圖SHA核對是來源指紋；正文可讀的判定來自真網站兩種viewport截圖及逐張view_image。',
        'root所做HTTP/local parity沒有作為我的閱讀證據；我保存自己的browser URL、viewport與截图SHA。',
        '未讀作者repair manifest、其他reader/reviews、技術報告、實作程式或外部模型知識。',
        '沒有GPU、訓練、估時、音訊播放或教材修改。'
    ]
}
out = base / 'reports/10_12-recheck-01.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'report': str(out), 'trace_entries': len(trace), 'read_pages': len(order), 'primary_reread': len(primary_reread), 'unchanged_primary_not_reread': len(not_reread), 'pageviews': len(pageviews), 'decision': report['decision']}, ensure_ascii=False))
