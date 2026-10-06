from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
CALLBACK=BASE/'evaluator-dependency-callback'
prefix=str(BASE.relative_to(ROOT))+'/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((CALLBACK/'prior-report.json').read_text())
out=json.loads((CALLBACK/'output.json').read_text())
assert r['reviewer_task']==out['reviewer_task']=='/root/p6_fact_19_5'
assert r['verdict']=='pass' and out['section_bytes_unchanged']
assert out['new_model_generations']==out['training_steps_run']==out['heldout_model_or_record_scoring']==0
assert sha(CALLBACK/'current-evaluate.py')==sha(ROOT/'scripts/selftrained/evaluate.py')==out['current_evaluator_sha256']
assert all(x['AST_identical'] for x in out['cited_contracts_AST_unchanged'].values())
assert all(x['unchanged'] for x in json.loads((CALLBACK/'original-raw-sha-check.json').read_text()))

command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.5/dependency_callback.py > docs/technical-reviews/artifacts/p6-19.5/dependency-callback-stdout.json 2>&1'
def artifact(i,name,kind,description,execution=False):
    a={'id':i,'path':prefix+name,'sha256':sha(BASE/name),'kind':kind,'description':description}
    if execution:a.update(command=command,result='exit0；只有兩個source變更；c13全部引用契約AST相同；POSIX/Windows-path fixture、實際Linux r+b fsync及合成評分正反例通過。沒有模型、訓練、留出評測或新生成。',environment={'python':'3.13.5','device':'cpu; standard-library/disposable software fixtures','platform':'Linux-6.18.44-x86_64-with-glibc2.41','Windows_scope':'PureWindowsPath simulation only; Windows syscall not executed'})
    r['artifacts'].append(a)
artifact('a_dep_prior_report','evaluator-dependency-callback/prior-report.json','source_snapshot','source-only callback前的本人完整pass報告原bytes，保留舊evaluator SHA與真實歷史。')
artifact('a_dep_current_evaluate','evaluator-dependency-callback/current-evaluate.py','code','親讀的新版evaluator完整原碼，保存當次SHA；不覆寫先前原碼快照。')
artifact('a_dep_diff','evaluator-dependency-callback/evaluator-exact-diff.patch','source_snapshot','與本人原先核對版本的精確兩處diff，非作者修正摘要。')
artifact('a_dep_code','dependency_callback.py','code','AST擷取真實函式及r+b/fsync語句的有界一次性軟體fixture程式；不載入course model。')
artifact('a_dep_execution','evaluator-dependency-callback/output.json','execution','本人source-only callback的全SHA、AST連續性及一次性fixture實際驗證結果。',True)
artifact('a_dep_stdout','dependency-callback-stdout.json','execution','同一實際source-only callback命令stdout。',True)
artifact('a_dep_original_raw','evaluator-dependency-callback/original-raw-sha-check.json','source_snapshot','另核對9個原始CPU/SFT/manifest/message檔與既存永久副本完整SHA，均未變。')
s=next(s for s in r['sources'] if s['id']=='s_evaluate')
assert s['sha256']==out['previous_evaluator_sha256']
s['sha256']=out['current_evaluator_sha256']
s['version']='source-dependency callback inspected working tree; current SHA '+out['current_evaluator_sha256']
s['inspection_note']+=' 追加本人source-only callback：先確認filesystem/tool可用，AST定位并讀實際兩處改動 code_fingerprints309–315、main830–851及EvaluationJournal145–187；再親讀c13的format/score225–306、Generator.generate395–457、intent分層495–570。新版diff只有POSIX keys與收據r+b fsync，所有c13引用函式AST相同；真實函式/語句在一次性軟體fixture執行。未讀reader/其他technical/authornotes。'
r['sources'].append({'id':'s_dep_execution','kind':'execution','title':'19.5 evaluator source-only dependency callback','verified':True,'artifact_id':'a_dep_execution'})
c=next(c for c in r['claims'] if c['id']=='c13')
c['evidence'].append({'source_id':'s_dep_execution','locator':'evaluator-dependency-callback/output.json /cited_contracts_AST_unchanged、/synthetic_score_checks、/receipt_open_mode_captured、/Windows_path_fixture_current_keys','supports':'新版score_reply/format_pass/intent分層/EOS記錄AST未變；真實scoring helpers對合成正確、錯答與context耗盡reply仍按契約評分。兩個作業系統可攜性修正不改教材所引用的有限評分行為。'})
c['artifact_ids']=list(dict.fromkeys(c['artifact_ids']+['a_dep_execution','a_dep_current_evaluate']))
c['verification']['details']+=' Source-only callback又以新版AST擷取helper執行合成fixture：semantic/format正例true、錯答與context耗盡false；無course records/model output重評。Windows-path keys改為POSIX而摘要不變，實際Linux r+b fsync收據bytes不變。'
c['scope']+=' 本次Windows-path fixture不是Windows OS syscall驗證，不能由本節核對宣稱所有Windows環境均可執行。'
r['review_history'].append({'stage':'source_dependency_only_callback','reviewer_task':'/root/p6_fact_19_5','verdict':'pass','source_sha256':r['source_sha256'],'prior_report_artifact_id':'a_dep_prior_report','prior_report_sha256':out['prior_report_sha256'],'previous_evaluator_sha256':out['previous_evaluator_sha256'],'current_evaluator_sha256':out['current_evaluator_sha256'],'execution_artifact_id':'a_dep_execution','actual_tool_and_filesystem_access_confirmed':True,'cited_claim_ids':['c13'],'cited_contract_AST_unchanged':True,'authored_section_unchanged':True,'disposable_software_fixtures_executed':True,'new_model_generations':0,'training_steps_run':0,'heldout_model_or_record_scoring':0})
r['read_scope']['source_dependency_callback']='親讀新版evaluator兩處改動及c13引用契約，實際filesystem/tool確認與software fixture驗證；a_dep_execution。19.5原bytes未變。'
r['checks']['source_verification']['details']+=' Source-only callback親讀新evaluator，確認只有POSIX key/r+b fsync修正；c13引用契約AST未變并以disposable fixtures驗證，來源SHA更新為本次親核對版本。'
r['checks']['factual_accuracy']['details']+=' evaluator依賴更新另由原reviewer實際source-only callback驗證，c13仍成立。'
r['checks']['limitations']['details']+=' source-only callback僅執行合成軟體及filesystem fixture，Windows測量範圍明列為PureWindowsPath，未執行Windows系統呼叫。'
assert all(c['status']=='verified' for c in r['claims'])
(ROOT/'docs/technical-reviews/19.5.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':r['verdict'],'section_sha256':r['source_sha256'],'evaluator_sha256':s['sha256'],'prior_report_sha256':out['prior_report_sha256'],'cited_claim':'c13'},ensure_ascii=False,indent=2))
