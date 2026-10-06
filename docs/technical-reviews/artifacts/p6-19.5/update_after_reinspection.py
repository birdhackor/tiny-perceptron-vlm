from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
prefix=str(BASE.relative_to(ROOT))+'/'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial_path=BASE/'initial-report-revise.json'
report=json.loads(initial_path.read_text())
assert report['reviewer_task']=='/root/p6_fact_19_5' and report['verdict']=='revise'
reinspection=json.loads((BASE/'reinspection-output.json').read_text())
assert reinspection['current_full_section_read']
assert reinspection['new_model_generations']==reinspection['training_steps_run']==reinspection['heldout_rows_rescored']==0
for family in ['artifact_hash_checks','repository_source_hash_checks','original_raw_hash_checks','fixed_data_hash_checks']:
    assert all(item['unchanged'] for item in reinspection[family])
source=(ROOT/'course/chapters/19.md').read_bytes()
start=source.index(b'## 19.5 ');end=source.index(b'## 19.6 ',start)
assert source[start:end]==(BASE/'current-section.md').read_bytes()
assert sha(BASE/'current-section.md')==reinspection['current_section_sha256']

initial_checker=BASE/'initial-checker-output.txt'
if not initial_checker.exists():initial_checker.write_bytes((BASE/'checker-output.txt').read_bytes())

def artifact(identifier,name,kind,description,command=None,result=None):
    a={'id':identifier,'path':prefix+name,'sha256':sha(BASE/name),'kind':kind,'description':description}
    if command is not None:
        a.update(command=command,result=result,environment={'python':'3.13.5','device':'cpu; standard-library SHA/JSON/text verification','platform':'Linux-6.18.44-x86_64-with-glibc2.41'})
    report['artifacts'].append(a)

artifact('a_initial_report','initial-report-revise.json','source_snapshot','同一位reviewer實際初次revise完整報告；只作修訂歷史，不作教材主張的權威來源。')
artifact('a_initial_checker','initial-checker-output.txt','source_snapshot','初次revise checker的真實輸出；保留未解決問題記錄，不冒充初次pass。')
artifact('a_current_section','current-section.md','source_snapshot','原reviewer實際重新閱讀的目前完整19.5原始bytes；canonical source_sha256對應此版本。')
artifact('a_current_context','reinspection-dependency-19.3.md','source_snapshot','複查時完整重新閱讀的必要19.3前文；唯一變動為OCR家族措辭，本節文字/工具規則未變。')
artifact('a_reinspection_code','reinspect.py','code','原reviewer實際执行的修订及证据SHA连续性检查程式。')
command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.5/reinspect.py > docs/technical-reviews/artifacts/p6-19.5/reinspection-stdout.json'
artifact('a_reinspection','reinspection-output.json','execution','修訂句、目前小節指紋、原39證據/9程式/9原始檔/12資料檔SHA及既有單次生成範圍的真實複查。',command,'exit 0；19.5只有所要求句子變更；舊證據均未變；目前句子只述內容與格式對應；0新生成/0訓練/0留出重評。')
artifact('a_reinspection_stdout','reinspection-stdout.json','execution','同一實際複查命令stdout，記錄目前及原始指紋與觀察支持範圍。',command,'exit 0；既有原始回答符合所列App問題及先前兩點要求，原issue已以收窄措辭解決。')
report['sources'].append({'id':'s_reinspection','kind':'execution','title':'Original reviewer actual 19.5 wording and evidence reinspection','verified':True,'artifact_id':'a_reinspection'})

c2=next(c for c in report['claims'] if c['id']=='c2')
c2['statement']='既有公開生成的內容對應當前App問題，也符合先前要求的兩點格式。'
c2['status']='verified'
c2['evidence']=[
    {'source_id':'s_verify','locator':'verification-output.txt public_cpu_existing_run_verification；原始actual_generations/0 prompt_messages/raw_output','supports':'核實既有四則公開輸入與其單一生成：回答包含App排查建議，並以兩行編號呈現。'},
    {'source_id':'s_reinspection','locator':'reinspection-output.json /section_change、/current_section_sha256、/observed_answer、/support','supports':'完整重讀目前19.5並核對精確單句修訂；目前句子只斷言已觀察的內容/格式對應，不再斷言內部因果作用。'},
    {'source_id':'s_inference','locator':'InferenceAssistant.reply lines277–313','supports':'先前要求與當前問題都確實在公開模型輸入；這是生成流程來源核實，不充作因果隔離證明。'},
]
c2['artifact_ids']=['a_verify','a_cpu_result','a_messages','a_reinspection','a_current_section']
c2['scope']='只支持這一個既有validation成功生成同時符合內容與兩點格式要求；沒有推論內部線索因果、一般記憶能力或所有新問法成功。'
c2['verification']={'method':'executed','expected':'目前完整19.5只將原因果句換成觀察對應；既有原始prompt/output可核實該對應。','observed':'exact one-sentence replacement；原結果是App檢查/重啟/客服建議，兩行以1.及2.編號，33個生成token含EOS；證據SHA均未變。','details':'原reviewer完整重新讀目前小節及變動前文，逐檔核對全部舊artifact、repository sources、原始CPU/SFT檔及12份固定資料指紋，直接復查指定result pointers。未追加神經生成或重新評分。','denominators':{'existing_selected_prompt':'1','historical_generation':'1','generated_tokens_including_EOS':'33','new_model_generations':'0','heldout_rows_rescored':'0'}}
c14=next(c for c in report['claims'] if c['id']=='c14')
c14['scope']='本節限制聲明及已修訂的觀察性措辭一致；有限示範／軟體編碼與一次成功均不外推為普遍誠實或可靠能力。'

original_issue=report['issues'][0]
original_issue['status']='resolved'
original_issue['resolution']='原reviewer已完整重讀目前19.5；原句精確改為「這個回答的內容對應當前 App 問題，也符合先前要求的兩點格式。」19.5其餘bytes未變，先前證據指紋完整核對未變，單例可支持此觀察對應。未追加模型因果/能力實驗；複查證據見a_reinspection。'
original_issue['resolved_statement']=c2['statement']
original_issue['resolved_source_sha256']=reinspection['current_section_sha256']
report['review_history']=[
    {'stage':'initial_independent_review','verdict':'revise','source_sha256':report['source_sha256'],'report_artifact_id':'a_initial_report','report_sha256':sha(initial_path),'issue_ids':['i1'],'unresolved_claim_ids':['c2']},
    {'stage':'original_reviewer_actual_reinspection','reviewer_task':'/root/p6_fact_19_5','verdict':'pass','source_sha256':reinspection['current_section_sha256'],'execution_artifact_id':'a_reinspection','resolved_issue_ids':['i1'],'full_current_section_read':True,'prior_evidence_hashes_unchanged':True,'changed_dependency':'19.3 only OCR family wording; read and irrelevant to this section text/tool claim; 19.4 unchanged','new_model_generations':0,'heldout_rows_rescored':0,'training_steps_run':0},
]
report['source_sha256']=reinspection['current_section_sha256']
report['verdict']='pass'
report['read_scope']['current_lesson']='19.5 full current original UTF-8; actual reread after exact one-sentence repair; a_current_section'
report['read_scope']['necessary_context']='19.3 current full manuscript reread after dependency SHA differed (only OCR family wording changed), 19.4 complete SHA-identical to initial read; no prior/other reader or technical conclusions read'
report['read_scope']['current_section_artifact']='a_current_section'
report['read_scope']['actual_reinspection_execution_artifact']='a_reinspection'
report['checks']['factual_accuracy'].update(status='pass',details='原完整核對仍適用；19.5唯一因果過強句已精確改為可由既有原始prompt/output驗證的內容與格式對應，原reviewer實際複查後確認i1解決。')
report['checks']['limitations'].update(status='pass',details='目前句子只寫單例內容及格式對應，保留不外推新問法/任意可靠程度限制；所有先前證據與必要程式指紋重核未變。未執行訓練、GPU、付費、新prompt生成、heldout evaluator、上傳或commit。')
report['checks']['source_verification']['details']+=' 複查逐項核對全部舊artifact、9份repository source、原始CPU/SFT及12資料檔SHA，均未變，方復用原獨立核對。'
report['checks']['source_verification']['claim_ids']=list(dict.fromkeys(report['checks']['source_verification']['claim_ids']+['c2']))
assert all(c['status']=='verified' for c in report['claims'])
assert all(i['status']=='resolved' for i in report['issues'])
(ROOT/'docs/technical-reviews/19.5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':report['verdict'],'source_sha256':report['source_sha256'],'reviewer_task':report['reviewer_task'],'initial_report_sha256':sha(initial_path),'resolved_issues':['i1'],'artifacts':len(report['artifacts'])},ensure_ascii=False,indent=2))
