"""Assemble only this owner's preserved report and personally completed callback."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parents[4]
OLD = 'docs/course-revision-20261007-phase7/reviews/freeze-03/reports/technical/f/p7_technical_f-initial-20261007T053106.json'
MANIFEST = 'docs/course-revision-20261007-phase7/reviews/freeze-04/manifest.json'
TRACE = 'docs/course-revision-20261007-phase7/reviews/freeze-04/traces/technical/f/f6f5240dffca4aeca152b6afb8dca93d.jsonl'
A = 'docs/technical-reviews/artifacts/p7_technical_f/'
def read(p):
    return json.loads((R / p).read_text())
def sha(p):
    return hashlib.sha256((R / p).read_bytes()).hexdigest()

report = read(OLD)
initial = copy.deepcopy(report)
manifest = read(MANIFEST)
frozen = {p['page_id']: p for p in manifest['pages']}
rows = [json.loads(x) for x in (R / TRACE).read_text().splitlines()]
assert rows[-1]['event'] == 'complete' and len(rows) == 50
changed = {p['page_id'] for p in initial['pages'] if p['source_sha256'] != frozen[p['page_id']]['source_sha256']}
assert changed == {'20.8', '20.13', 'natural-v4-training'}
assert all(p['figures_sha256'] == frozen[p['page_id']]['figures_sha256'] for p in initial['pages'])
now = datetime.datetime.now(datetime.timezone.utc)
report['created_at'] = now.isoformat()
report['manifest_sha256'] = sha(MANIFEST)
report['verdict'] = 'pass'
report['trace_files'].append({'path': TRACE, 'sha256': sha(TRACE)})
report['predecessor_report'] = {'path': OLD, 'sha256': sha(OLD), 'verdict': initial['verdict'], 'preserved_unchanged': True}
report['initial_summary'] = initial['summary']
report['initial_completion'] = initial['completion']
report['checkpoint_count'] = 436
report['primary_checkpoint_count'] = 408
report['checkpoint_count_scope'] = {'initial': 388, 'initial_primary': 374, 'callback': 48, 'callback_primary': 34, 'unique_primary_pages': 54, 'counts_include_actual_repeated_callback_reading': True}
report['completion'] = {'session': rows[0]['session'], 'original_start_at': rows[0]['recorded_at'], 'actual_complete_at': rows[-1]['recorded_at'], 'complete_event_index': rows[-1]['event_index'], 'scope': 'Three changed primary pages and assigned19.10–19.12 fully re-read in original-owner callback; 51 byte-identical pages retain this same owner\'s initial full evidence and trace. Both immutable traces retained.'}
report['summary'] = '54頁完整技術報告：51個same-byte頁保留本人初輪核證據；20.8、20.13與natural-v4-training全頁及指定前文本人另存48個新checkpoint完成回查。三原必要問題均有實際修正與真回查，歷史數字、本人14/28判分、原失敗／更正／附帶metadata接觸及視覺與外部執行限制全部保留。'
report['same_byte_inheritance'] = [{'page_id': p['page_id'], 'source_sha256': p['source_sha256'], 'trace_file': p['trace_file'], 'basis': 'This owner\'s collected initial report; current frozen page and figure bytes equal initial page bytes.'} for p in initial['pages'] if p['page_id'] not in changed]
report['callback_scope'] = {'primary_page_ids': rows[0]['primary_page_ids'], 'context_page_ids': rows[0]['context_page_ids'], 'recheck_of': rows[0]['recheck_of'], 'failed_start_preserved': 'First start rejected because dispatch followup_receipt was {}; root retained old metadata and appended factual nonempty receipt text with original raw_result{}, then same scope start succeeded. No new body read before successful start. This was metadata correction, not scientific evidence or invented new call.', 'extra_execution': 'Only exact repaired Bash fence with temporary CPU Python alias and trainer stub: 11 synthetic progress cases. No real model/training/GPU/remote execution.', 'visual_scope': 'New actual tools.view_image receipts: 20.8 six selected views;20.13 seven;natural-v4-training eleven;context19.11 SVG640/360. Selected necessary tables and amended adjacent prose, not every page pixel or full horizontally overflowing command.', 'criteria_unchanged': True}

resolutions = {
 'F20.8-history-grading-scope': ('20.8', 16, '表前明示沿用本次固定逐題評分紀錄並提醒開放式語意判分可因評分者而異；綜合分數結論限定「依本次評分紀錄」。歷史16/28保留，本人14/28及自己的1039綜合分數略高108/75600也原樣保留，未推翻獨立EOS／語音門檻。全文兩段與桌機／手機表及前後限定句本人真回查。'),
 'F20.13-action-subset-scope': ('20.13', 20, '表內與結尾均明示動作／姿態／狀態3/11，並列站著／坐著、交通錐直立／倒下的實際混合題例。3/11及原分母／84題內子集範圍保持不變，吻合本人初輪實核原11題，未宣稱純動作或全42圖重新語意評分。全文四段與九列能力表桌機／手機真回查。'),
 'Fnatural-training-resume-guard': ('natural-v4-training', 48, '修後exact unit13 fence以 ) && \\ 條件連接；unit12明示檢查失敗不啟動訓練。本人逐段重讀全28段，實際CPU stub執行未修改fence：n0/1038/1039/1040/2076五例成功且未來檢查點正確；n2077/-1、缺檔、缺key、壞JSON、字串steps六例exit1均無trainer marker。只支持Bash控制及pending篩選契約，不支持真正續訓或GPU測量。初輪無防護的失敗／marker原樣保留。')
}
def resolve(issue):
    pid, event, explanation = resolutions[issue['id']]
    assert rows[event]['page_id'] == pid and issue['id'] in rows[event]['rechecks']
    issue['initial_status'] = issue['status']
    issue['status'] = 'resolved'
    issue['resolution'] = explanation
    issue['recheck'] = {'trace_file': TRACE, 'event_index': event}
    issue['resolved_at'] = rows[event]['recorded_at']
for issue in report['issues']:
    resolve(issue)

receipt_names = {'20.8': 'recheck04-page-20.8-view.json', '20.13': 'recheck04-page-20.13-view.json', 'natural-v4-training': 'recheck04-page-natural-v4-training-view.json'}
summaries = {
 '20.8': '歷史原逐題計數與token證據沿用本人初核；修後首次表前明示固定評分紀錄／評分者邊界，綜合排序也限定於本次紀錄。本人14/28與歷史16/28差異、1039獨立綜合值略高及EOS／語音門檻失敗全部保留；全頁兩段與六張新圖真回查完成。',
 '20.13': '原歷史逐題分數、EOS、OCR／CER及固定模型參數核證據保留；修後動作／姿態／狀態3/11標籤及站坐／交通錐題例對齊本人初核原混合子集，表和結尾一致，未改歷史數字。全頁四段與七張新圖回查；未本輪新判42張照片或執行ASR／模型。',
 'natural-v4-training': '原訓練資源、LoRA／題級損失、更新行数、選版公式與交付角色核證據保留。修後前置檢查用&&成功才train；本人exact fence CPU stub十一例核實五成功／六失敗不呼叫train。全頁28段及四表／修後鄰文十一張新圖回查；未執行真正模型續訓或新GPU測量。'
}
for page in report['pages']:
    pid = page['page_id']
    if pid not in changed:
        assert page == next(p for p in initial['pages'] if p['page_id'] == pid)
        continue
    original = next(p for p in initial['pages'] if p['page_id'] == pid)
    page['initial_page_record'] = copy.deepcopy(original)
    page['source_sha256'] = frozen[pid]['source_sha256']
    page['trace_file'] = TRACE
    notes = [row for row in rows[1:-1] if row['page_id'] == pid]
    page['checkpoint_event_indices'] = [row['event_index'] for row in notes]
    page['checkpoint_unit_indices'] = [row['unit_index'] for row in notes]
    page['verdict'] = 'pass'
    page['summary'] = summaries[pid]
    for issue in page['issues']:
        resolve(issue)
    receipt_path = A + receipt_names[pid]
    receipt = read(receipt_path)
    assert receipt['source_sha256'] == page['source_sha256']
    page['page_visual_check'] = {'status': 'verified', 'required': True, 'source_sha256': receipt['source_sha256'], 'details': receipt['observation'], 'observation': receipt['observation'], 'receipt': {'path': receipt_path, 'sha256': sha(receipt_path)}, 'artifacts': receipt['artifacts']}
    page['missing_visuals'] = receipt['observation']
    page['callback_resolution'] = {'issue_id': page['issues'][0]['id'], 'rechecks': [{'event_index': row['event_index'], 'unit_index': row['unit_index']} for row in notes if row['rechecks']], 'scope': resolutions[page['issues'][0]['id']][2]}

pages = {p['page_id']: p for p in report['pages']}
pages['20.8']['checks']['limitations'].update(status='pass', details='修後首次數字與排名句均限定本次固定評分紀錄，開放式判分者邊界清楚；歷史數值與本人獨立差異全部保留，EOS／voice獨立門檻不由primary抵消。')
pages['20.13']['claims'][2]['statement'] += ' 原動作／姿態／狀態子集3/11、關係子集21/29已含在84題內；修後命名及站坐／交通錐例子對齊原題。'
pages['20.13']['claims'][2]['verification']['denominators'].update(action_posture_state=11, relation=29)
pages['20.13']['checks']['factual_accuracy'].update(status='pass', details='修後能力表與結尾同用動作／姿態／狀態標籤，实际站坐／直立倒下題例符合原11題；歷史3/11及其餘數值未改。')
pages['20.13']['checks']['limitations'].update(status='pass', details='178回答非独立、歷史人工判分及未測範圍保留；混合動作／姿態／狀態子集修後命名清楚，沒有純動作泛化或本人全42圖新判分主張。')
p = pages['natural-v4-training']
old_failure = next(c for c in p['claims'] if c['id'] == 'resume-failure')
old_failure['location'] = 'Preserved original freeze-03 §5 unit13; historical pre-repair failure, not current repaired fence'
old_failure['scope'] += '。本claim保留原錯誤的歷史證據，current freeze-04正確條件連接另以resume-success-guard新claim核對。'
old_failure['verification']['details'] += ' 此為原錯誤歷史記錄，原artifact不可覆寫；當前修正見resume-success-guard。'
guard_path = A + 'recheck04-resume-exact-fence.json'
guard = read(guard_path)
assert guard['exit_code'] == 0
execution = json.loads(guard['stdout'])
assert len(execution['cases']) == 11 and sum(c['trainer_marker'] for c in execution['cases']) == 5
p['artifacts'].append({'id': 'repaired-guard', 'kind': 'execution', 'path': guard_path, 'sha256': sha(guard_path), 'description': 'Personally executed exact repaired current unit13 Bash fence, temporary CPU Python alias/trainer stub only; all eleven synthetic cases stdout/stderr/status/marker/fence SHA preserved.', 'command': json.dumps(guard['command']), 'result': 'exit0 audit; bash syntax0; five valid cases train stub marker true and correct future checkpoints; six failing prechecks Bash exit1 and marker false.', 'environment': {**guard['environment'], 'bash': execution['bash_version'], 'wrapper_python': guard['wrapper_environment']['python']}})
fence_path = A + 'recheck04-exact-resume-fence.sh'
assert sha(fence_path) == execution['fence_sha256']
p['artifacts'].append({'id': 'repaired-fence', 'kind': 'code', 'path': fence_path, 'sha256': sha(fence_path), 'description': 'Exact unchanged repaired fence personally extracted from then-current unit13 before CPU execution; no model or trainer invoked.'})
p['sources'].extend([{'id': 'repaired-resume-snippet', 'kind': 'repository_code', 'title': 'Personally read current repaired unit13 exact fence', 'verified': True, 'path': fence_path, 'sha256': sha(fence_path), 'version': 'freeze-04 SHA256:' + sha(fence_path), 'inspection_note': 'Personally read full unit12/13: assignment success connected with &&; exact full fence retained and actually executed in temporary CPU stub fixtures. This is current corrective source, separate from preserved original failed fence.'}, {'id': 'repaired-resume-execution', 'kind': 'execution', 'title': 'Owner exact repaired fence CPU stub outcomes', 'verified': True, 'artifact_id': 'repaired-guard'}])
p['claims'].append({'id': 'resume-success-guard', 'kind': 'software', 'statement': '修後pending檢查成功才執行train命令；五個0至2076範圍內合成步數得到未來1039／2077檢查點，完成／非法／讀檔與解析失敗六例均無train stub marker。', 'location': 'Current natural-v4-training §5 units12–13, full page reread through unit27', 'scope': 'Unmodified exact fence Bash5.2.37/projectPython3.13.5 CPU temporary trainer stub; synthetic JSON only. Does not prove actual adapter loading, resumed optimization, GPU training, remote workflow or model output.', 'status': 'verified', 'evidence': [{'source_id': 'repaired-resume-snippet', 'locator': 'assignment end ) && backslash; Python assert/read/print; --checkpoint-steps expansion', 'supports': 'Actual current source success-only control connection and future-step filter'}, {'source_id': 'repaired-resume-execution', 'locator': 'stdout.cases all11; bash_syntax_exit; fence_sha256', 'supports': 'Five success paths correct checkpoints; six failed prechecks exit1, trainer marker false'}], 'artifact_ids': ['repaired-guard', 'repaired-fence'], 'verification': {'method': 'executed', 'expected': 'Only successful prechecks invoke trainer stub; future steps omit completed archives.', 'observed': 'n0/1038→1039,2077; n1039/1040/2076→2077; n2077/-1/missing file/missing key/malformed JSON/string steps each exit1 and no marker.', 'details': 'Exact fence SHA479fb677386e3adb72187e6fee5936529409361fe451282d8be3970254312a71 identical across eleven cases. Relative Python alias points to actual project interpreter; scripts/natural_assistant.py only temporary marker-writing stub. No real weights/model/GPU/network/training touched.'}})
p['checks']['factual_accuracy'].update(status='pass', details='原方法與文件角色證據保留；修後前置檢查以&&條件連接，exact fence真CPU十一例驗明失敗不呼叫train stub。', claim_ids=['lora-and-validation', 'resume-success-guard'])
p['checks']['figure_consistency'].update(details='全頁正文重讀後本人真view十一張桌機／手機圖：四表完整資料列與修後&&鄰文可讀。長命令横向未全視覺展開，另以全文exact fence CPU核對。')
p['checks']['source_verification'].update(details='沿用本人親核原LoRA/PyTorch及實作／歷史raw fields；另親讀當前修後unit12/13，exact fence保存hash並以CPU stub真執行。', claim_ids=['lora-and-validation', 'historical-training-resources', 'resume-success-guard'])
p['checks']['limitations'].update(details='保留原help／parser／fixture／歷史GPU範圍及原guard失敗；修後僅新增exact fence合成CPU控制測試，未真正續訓或重跑推論／遠端發布。', claim_ids=['cli-and-checkpoint-contract', 'resume-failure', 'resume-success-guard'])
report['execution_scope']['callback'] = report['callback_scope']['extra_execution']
report['corrections_and_failed_probes'].append({'callback': report['callback_scope']['failed_start_preserved']})
out = 'docs/course-revision-20261007-phase7/reviews/freeze-04/reports/technical/f/p7_technical_f-recheck-' + now.strftime('%Y%m%dT%H%M%S') + '.json'
(R / out).parent.mkdir(parents=True, exist_ok=True)
with (R / out).open('x') as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'path': out, 'sha256': sha(out), 'verdict': report['verdict'], 'pages': len(report['pages']), 'callback_checkpoint_count': 48}, ensure_ascii=False))
