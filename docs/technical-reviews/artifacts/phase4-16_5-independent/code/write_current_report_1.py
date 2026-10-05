"""This same reviewer writes the complete current report after the actual recheck."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
REL=OUT.relative_to(ROOT).as_posix()
TASK='/root/phase4_factual_coordinator/factual_16_5'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((OUT/'initial-review.json').read_text())
receipt_path=OUT/'recheck-timing-scope-1.json'
receipt=json.loads(receipt_path.read_text())
assert report['reviewer_task']==receipt['reviewer_task']==TASK
assert receipt['verdict']=='pass' and receipt['unresolved_substantive_issues']==[]
assert sha(OUT/'input/section-recheck-1.md')==receipt['source_sha256']
report['source_sha256']=receipt['source_sha256']
report['verdict']='pass'
command='.venv/bin/python docs/technical-reviews/artifacts/phase4-16_5-independent/code/recheck_timing_scope_1.py > docs/technical-reviews/artifacts/phase4-16_5-independent/execution/recheck-timing-scope-1-stdout.txt 2> docs/technical-reviews/artifacts/phase4-16_5-independent/execution/recheck-timing-scope-1-stderr.txt'
env={'python':'3.13.5','device':'cpu','work':'原始資料與AST檢查；未訓練或生成模型成績'}
paths=[
 ('recheck_timing_scope_1','recheck-timing-scope-1.json','execution','本人重新讀完整新版16.5、親讀immutable計時方法與原測量pointers、再次view原render圖；逐claim再判斷的正式receipt。'),
 ('recheck_timing_scope_1_stdout','execution/recheck-timing-scope-1-stdout.txt','execution','本人實際重算37分母及秒轉毫秒、核對原件SHA的命令stdout。'),
 ('recheck_timing_scope_1_stderr','execution/recheck-timing-scope-1-stderr.txt','source_snapshot','本次有界核查stderr，空檔；沒有失敗例外。'),
 ('current_section_recheck_1','input/section-recheck-1.md','source_snapshot','本人本次實際完整重讀的新版16.5原始UTF-8 bytes。'),
 ('current_chapter_recheck_1','input/chapter-16-recheck-1.md','source_snapshot','本次複查實際full chapter snapshot；不是替其他小節審閱，原initial frozen snapshot另存。'),
 ('initial_review_history','initial-review.json','source_snapshot','本人首次revise完整報告的原始永久歷史；不是本次claim的科學來源。'),
 ('initial_review_record','initial-review-record.json','source_snapshot','本人initial報告SHA與checker exit1原始紀錄。'),
 ('recheck_timing_scope_1_code','code/recheck_timing_scope_1.py','code','本次原measurement/AST/分母查核的實際有界腳本。'),
 ('write_current_report_1_code','code/write_current_report_1.py','code','本人依實際recheck結果重建完整current report的腳本；不是科學concept來源。')]
for identifier,relative,kind,description in paths:
    p=OUT/relative
    a={'id':identifier,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'kind':kind,'description':description}
    if kind=='execution':
        a.update(command=command,result='exit0；ACTUAL_RECHECK_PASSED。40−3=37；PAD10.798640999997389ms、packing11.19073399999948ms；新文字的每個計時步驟均吻合immutable方法。',environment=env)
    report['artifacts'].append(a)
report['sources'].append({'id':'recheck_execution_1','kind':'execution','title':'本人對新完整16.5的實際科學來源與範圍複查',
                          'verified':True,'artifact_id':'recheck_timing_scope_1'})
for c in report['claims']:
    c['artifact_ids'].append('recheck_timing_scope_1')
    c['current_recheck']={'receipt_artifact_id':'recheck_timing_scope_1','locator':'/other_claims_personally_reassessed/'+c['id'],
                          'finding':receipt['other_claims_personally_reassessed'][c['id']]}
    if c['id']=='timing_scope':
        c['status']='verified'
        c['statement']='略去每路徑前3次暖身計時，餘37次模型更新中位數PAD10.799ms、packing11.191ms；計時含forward、loss、backward、clip、optimizer.step及末尾GPU同步，不含抽樣、裝填與清空梯度。'
        c['scope']='只支持原L4 seed42這次warm model update區間中位數比較；40次參數更新及845目標全部保留，沒有把前3次更新丟棄。不是端到端流程計時或通用速度結論。'
        c['verification']['expected']='37個暖身後模型更新時長，精確列計時內外步驟；中位數為10.799ms、11.191ms。'
        c['verification']['observed']='本人重算40−3=37；原scalar秒×1000並四捨五入吻合，全文的新描述逐步匹配原method。'
        c['verification']['details']='親讀immutable _sync40-44及_packing_updates731-751,773-787，核對原warm_step_median_seconds fields。沒有原per-step latency vector，未補造全40次中位數；新版已限定37次warm update區間。'
        c['evidence'].append({'source_id':'recheck_execution_1','locator':'/scientific_source_inspection and /changed_text_personally_read',
            'supports':'本人实际读新版并回到原科学数据与方法，核37样本分母、秒到毫秒以及包含/排除步骤；不以writer或coordinator宣告作为方法证据。'})
for issue in report['issues']:
    assert issue['id']=='timing_scope_missing'
    issue['status']='resolved'
    issue['resolution']='本人重新讀完整新版16.5，親核raw /results/packing/actual_updates/{padded,packed}以及immutable _packing_updates734-751,786；現文已說排除前3次暖身、餘37次、模型更新內部計時及排除抽樣/裝填/清空梯度，完全吻合。實際receipt artifact recheck_timing_scope_1；initial revise與原問題保留initial-review.json。'
    issue['history']={'initial_status':'open','initial_report_sha256':sha(OUT/'initial-review.json'),
                     'initial_source_sha256':report['frozen_input']['sha256'],
                     'initial_section_sha256':receipt['history']['original_source_sha256'],
                     'current_section_sha256':receipt['source_sha256'],
                     'recheck_receipt_artifact_id':'recheck_timing_scope_1'}
    # The full frozen chapter hash is explicitly named rather than used as section hash.
    issue['history']['initial_frozen_full_chapter_sha256']=issue['history'].pop('initial_source_sha256')
details={
 'factual_accuracy':'本人完整重讀新16.5並逐claim複查；唯一initial計時範圍問題已由原measurement/immutable方法親核解決，其餘機制/API/原結果轉寫仍成立。',
 'numeric_verification':'原8/4/4、14、845、40與15題exact分母及數值四捨五入已核；本次親算40−3=37，10.798640999997389ms→10.799，11.19073399999948ms→11.191，正文現明記計時區間。',
 'figure_consistency':'原SVG bytes與初審一致；本人再次view已實render PNG。Query列/Key欄、A/B位置及16格0/1均與新版正文/fence相符；未驗響應式網頁。',
 'source_verification':'本次本人回查原GPU測量具名pointers、hash匹配immutable計時原碼及_sync；原權威摘錄/方法/dataset版本均未變。正式新增receipt recheck_timing_scope_1，非coordinator writer證據。',
 'limitations':'新版已準確限定37次warm model update計時；原始單seed/小題組、assistant-text任務與SFT不同、CPU示範零更新及無per-step latency vector的真實scope皆保留。'}
for name,check in report['checks'].items():
    check['status']='pass'; check['details']=details[name]
report['revision_history']=[
 {'stage':'initial independent review','verdict':'revise','source_sha256':receipt['history']['original_source_sha256'],
  'report_snapshot':REL+'/initial-review.json','report_sha256':sha(OUT/'initial-review.json'),
  'issue':'timing_scope_missing'},
 {'stage':'same-reviewer actual follow-up','verdict':'pass','source_sha256':receipt['source_sha256'],
  'receipt_artifact_id':'recheck_timing_scope_1','receipt_path':receipt_path.relative_to(ROOT).as_posix(),
  'receipt_sha256':sha(receipt_path),'scope':'重新讀完整新版；親核原計時科學來源/37分母，逐claim再判斷，並再view未變原render圖；無GPU重訓/模型重評。'}]
report['current_recheck_receipt_artifact_id']='recheck_timing_scope_1'
target=ROOT/'docs/technical-reviews/16.5.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
new=json.loads(target.read_text())
assert new['reviewer_task']==TASK and new['verdict']=='pass'
assert new['source_sha256']==receipt['source_sha256']
assert all(c['status']=='verified' for c in new['claims'])
print('OWN_CURRENT_REPORT_WRITTEN_AND_IDENTITY_ASSERTED')
print('REPORT_SHA256',sha(target))
print('SOURCE_SHA256',new['source_sha256'])
print('FORMAL_RECEIPT_ID',new['current_recheck_receipt_artifact_id'])
print('FORMAL_RECEIPT_PATH',receipt_path.relative_to(ROOT).as_posix())
print('FORMAL_RECEIPT_SHA256',sha(receipt_path))
