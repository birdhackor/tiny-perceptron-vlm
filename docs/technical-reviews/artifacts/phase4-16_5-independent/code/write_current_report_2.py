"""Same owner assembles the complete current report from actual reinspection."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parents[1]
REL=OUT.relative_to(ROOT).as_posix()
TASK='/root/phase4_factual_coordinator/factual_16_5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior_path=OUT/'prior-current-review-1-opaque.json'
report=json.loads(prior_path.read_text())
receipt_path=OUT/'reinspection-16_5-2.json'
receipt=json.loads(receipt_path.read_text())
assert receipt['reviewer_task']==report['reviewer_task']==TASK
assert receipt['verdict']=='pass' and receipt['unresolved_substantive_issues']==[]
assert receipt['changed_text']['exact_only_change'] and receipt['changed_text']['changed_substantive_claims']==[]
assert sha(OUT/'input/section-recheck-2.md')==receipt['source_sha256']
spec=importlib.util.spec_from_file_location('facts',ROOT/'docs/review-tools/section_facts.py')
facts=importlib.util.module_from_spec(spec);spec.loader.exec_module(facts)
actual,_,_=facts.original_section(ROOT/'course/chapters/16.md','16.5')
assert hashlib.sha256(actual).hexdigest()==receipt['source_sha256']
report['source_sha256']=receipt['source_sha256']
report['verdict']='pass'
command='.venv/bin/python docs/technical-reviews/artifacts/phase4-16_5-independent/code/reinspection_16_5_2.py > docs/technical-reviews/artifacts/phase4-16_5-independent/execution/reinspection-16_5-2-stdout.txt 2> docs/technical-reviews/artifacts/phase4-16_5-independent/execution/reinspection-16_5-2-stderr.txt'
for identifier,path,kind,description in [
 ('reinspection_16_5_2','reinspection-16_5-2.json','execution','同一原owner完整重讀目前16.5、只核字形變動與支持含義、精確核既有code/figure/來源/ctx後復用自身真證據的正式receipt。'),
 ('reinspection_16_5_2_stdout','execution/reinspection-16_5-2-stdout.txt','execution','本次實際bytes/原source指紋、必要ctx與唯一字形變動查核stdout。'),
 ('reinspection_16_5_2_stderr','execution/reinspection-16_5-2-stderr.txt','source_snapshot','本次核查stderr空檔。'),
 ('reinspection_16_5_2_source','input/section-recheck-2.md','source_snapshot','本人本次完整讀取的目前16.5原始UTF-8 bytes，無正規化換行。'),
 ('prior_current_review_1_opaque','prior-current-review-1-opaque.json','source_snapshot','own前一完整報告opaque原樣備份，含當时issues及所有proof引用；不是科學來源。'),
 ('reinspection_16_5_2_code','code/reinspection_16_5_2.py','code','本人本次實際有界檢查腳本，無模型訓練/重評。'),
 ('write_current_report_2_code','code/write_current_report_2.py','code','本人依實際reinspection組成完整current report的腳本；不作concept科學來源。')]:
 p=OUT/path
 a={'id':identifier,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'kind':kind,'description':description}
 if kind=='execution':
  a.update(command=command,result='exit0；REINSPECTION_COMPLETE_NO_NEW_TECHNICAL_CHANGE。本人完整讀現節，只「谁→誰」一字變動；全部必要原source與ctx精確指紋吻合；原科學claim支持範圍未改。',environment=receipt['environment'])
 report['artifacts'].append(a)
report['sources'].append({'id':'reinspection_execution_2','kind':'execution','title':'本人當前完整16.5文字修訂的實際複查與原證據復用核對',
                         'verified':True,'artifact_id':'reinspection_16_5_2'})
for c in report['claims']:
 c['artifact_ids'].append('reinspection_16_5_2')
 c['current_reinspection']={'receipt_artifact_id':'reinspection_16_5_2',
   'source_sha256':receipt['source_sha256'],'scope':'本人完整重讀現節；無此claim的技術變動，按receipt中的精確原件/必要ctx/SVG指紋復用本人既有真實原來源及執行證據。'}
 if c['id'] in ('packing_independence','mask_api'):
  c['current_reinspection']['scope']='本人完整重讀現節並核「谁→誰」不改誰可讀誰的意思；重讀current attention_mask10-18，保留因果與segment交集及圖表軸的原科學支持。'
for issue in report['issues']:
 assert issue['status']=='resolved'
 issue['current_reinspection_confirmation']={'receipt_artifact_id':'reinspection_16_5_2',
   'finding':'本人現節完整重讀，前3暖身/餘37次及timing包含排除步驟仍原樣存在，原timing問題沒有回退；原issue history及前判定保留。'}
details={
 'factual_accuracy':'同一原owner重讀目前完整16.5，確認唯一字形修訂「谁→誰」未改技術意思；重核原helper契約及原件指紋，明示復用本人既有科學來源與真實CPU/原measurement核查。',
 'numeric_verification':'本輪無新數字/單位/分母；原8/4/4、14、845、40與37暖身後timing、15保留題轉寫均未變；本人完整重讀并核原measurement/方法/永久proof精確SHA後復用此前真計算，不虛報重跑。',
 'figure_consistency':'本輪SVG、原render PNG、fence及圖說精確bytes未變；復用本人先前Inkscape實render與view證據。只有誰字正文變動，沒有新圖形/畫面主張；未聲稱本次browser或root預覽檢查。',
 'source_verification':'本人重讀最新factual方法、完整现節及current attention_mask10-18；必要ctx3.6/3.7/7.3/7.7/5.10、原methods/raw/dataset及primary snapshots精確SHA核對。官方原版本/locator复用本人已親核來源，不讀別人判斷或作者修訂摘要。',
 'limitations':'現節仍明確限原L4一次實測、warm模型更新37分母及計時區間；assistant-text與SFT目標、CPU零更新、保留題組品質範圍均未變。只有文字字形變動，無新技術未知；不擴查16.4或代審intro。'}
for key,c in report['checks'].items():c['status']='pass';c['details']=details[key]
report['revision_history'].append({'stage':'same-owner current typography reinspection','verdict':'pass',
 'source_sha256':receipt['source_sha256'],'prior_report_opaque_path':prior_path.relative_to(ROOT).as_posix(),
 'prior_report_sha256':sha(prior_path),'receipt_artifact_id':'reinspection_16_5_2',
 'receipt_path':receipt_path.relative_to(ROOT).as_posix(),'receipt_sha256':sha(receipt_path),
 'scope':'本人完整讀目前16.5、唯一谁→誰字形修訂，親核當前helper與未變科學來源/必要ctx/figure精確指紋；復用自身原真證據，沒有重fetch或完整CPU/GPU/模型成績重跑。'})
report['current_recheck_receipt_artifact_id']='reinspection_16_5_2'
report['intro_reinspection_applicability']=receipt['intro']
report['pending_dependencies']=[]
target=ROOT/'docs/technical-reviews/16.5.json'
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
new=json.loads(target.read_text())
assert new['reviewer_task']==TASK and new['source_sha256']==receipt['source_sha256']
assert new['verdict']=='pass' and all(c['status']=='verified' for c in new['claims'])
print('OWN_CURRENT_REPORT_WRITTEN_AND_IDENTITY_ASSERTED')
print('REPORT_SHA256',sha(target))
print('SOURCE_SHA256',new['source_sha256'])
print('FORMAL_RECEIPT_ID',new['current_recheck_receipt_artifact_id'])
print('FORMAL_RECEIPT_PATH',receipt_path.relative_to(ROOT).as_posix())
print('FORMAL_RECEIPT_SHA256',sha(receipt_path))
print('FIGURE_SHA256',new['figure_sha256'])
print('INTRO_NOT_APPLICABLE_NON_FIRST_OWNER')
print('PRIOR_OPAQUE_SHA256',sha(prior_path))
