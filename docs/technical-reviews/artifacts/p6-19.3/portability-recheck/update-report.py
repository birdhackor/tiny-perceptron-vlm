from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
REL=OUT.relative_to(ROOT).as_posix()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report=json.loads((OUT/'before-report.json').read_bytes())
assert report['verdict']=='pass' and report['reviewer_task']=='/root/p6_fact_19_3'
output=(OUT/'check-output.txt').read_text()
assert '"verdict": "pass"' in output and '"training_test_or_model_generation_executed": false' in output
prior_source=next(s for s in report['sources'] if s['id']=='s_eval')
previous_sha=prior_source['sha256'];current_sha=sha(ROOT/'scripts/selftrained/evaluate.py')
assert sha(OUT/'previous-evaluate.py')==previous_sha and sha(OUT/'current-evaluate.py')==current_sha
def artifact(identifier,name,kind,description,**extra):
 report['artifacts'].append({'id':identifier,'kind':kind,'path':f'{REL}/{name}','sha256':sha(OUT/name),'description':description,**extra})
artifact('a_portability_prior_report','before-report.json','source_snapshot','同一原审阅者在此callback前的19.3 pass报告原bytes，保留此前eval source SHA及初次revise/字卡修复完整历史。')
artifact('a_eval_previous','previous-evaluate.py','source_snapshot','Git b3a6684a取得的原evaluate.py完整bytes；SHA828b7cbb…与此前本审阅者正式来源SHA精确相同，不来自旧作者摘要/他节审阅。')
artifact('a_eval_current','current-evaluate.py','source_snapshot','callback实际读取的当前evaluate.py完整bytes，SHA99feca5e…；只改code_fingerprints路径序列化和pre-marker receipt open mode。')
artifact('a_eval_diff','evaluate-diff.patch','code','真实previous/current源bytes产生的unified diff：仅as_posix与rb→r+b两行。')
artifact('a_portability_code','check.py','code','本callback真实CPU查核代码；只执行code_fingerprints及benign receipt fsync AST，不调用evaluate.main、model或journal。')
artifact('a_portability','check-output.txt','execution','当前evaluate路径/fsync有限CPU查核及被19.3依赖的旧代码契约AST不变核对。',command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.3/portability-recheck/check.py > docs/technical-reviews/artifacts/p6-19.3/portability-recheck/check-output.txt 2>&1',result='exit 0；精确两行delta；31 fingerprint keys保持POSIX格式并追踪当前eval SHA；PureWindowsPath/PosixPath的as_posix一致；当前原fsync AST在Linux完成且fixture bytes SHA不变。旧Journal/protocol validation/Generator/summarize/scoring/voice continuation AST相同，原输入与续聊CPU证据可复用。课程节、图与其他正式代码源SHA未变。未验证Windows原生fsync，未做训练/test/模型生成。',environment={'python':'3.13.5','platform':'Linux-6.18.44-x86_64-with-glibc2.41','device':'cpu','model_training_generation_or_heldout_eval':'none'})
artifact('a_portability_receipt','fixture-receipt.json','source_snapshot','仅为CPU文件系统helper创建的benign receipt，fsync前后bytes相同；不是实际evaluation journal/protocol或模型输出。')
report['sources'].append({'id':'s_eval_previous','kind':'repository_code','title':'Previously inspected evaluate.py source retained for original measured-result provenance','path':f'{REL}/previous-evaluate.py','sha256':previous_sha,'version':'Git b3a6684a1af47f1086c1b9e56ed090ff6699220a original evaluate.py; matches previous reviewer SHA exactly','verified':True,'inspection_note':'原审阅者此前亲读的源bytes保存供原raw measured-record provenance；本callback把它与当前源逐bytes/AST比对，只发现路径/fsync两行。未声称原始heldout记录属于新code SHA下重新执行。'})
report['sources'].append({'id':'s_portability','kind':'execution','title':'19.3 original reviewer callback on current evaluate.py filesystem portability changes','verified':True,'artifact_id':'a_portability'})
prior_source['sha256']=current_sha
prior_source['version']='2026-10-06 current working-tree filesystem-portability implementation; exact source SHA99feca5ec576f3cf27cd799ffcce4c72039941938f3cf4b33aa9cac09bdfbdb6'
prior_source['inspection_note']+=' 本人filesystem callback：精确比较与此前同SHA的Git原bytes；L313 str(relative_path)→as_posix，L839 receipt rb→r+b。全部其余源bytes相同；Journal/protocol_contents/validate_protocol/Generator/summarize/score_reply及voice续聊branch AST相同。实际CPU运行原code_fingerprints及独立benign receipt原fsync AST，原材料/续聊契约证据在未变分支上复用；没有重训、test、模型generation，也没有声称Windows原生运行已验证。'
for identifier in ['c3','c16','c17']:
 claim=next(c for c in report['claims'] if c['id']==identifier)
 claim['evidence'].append({'source_id':'s_portability','locator':'check-output exact_source_delta, reused_original_contract_evidence, path_checks, receipt_fsync, scope_and_result','supports':'当前evaluate.py只有两行操作性变化；19.3相关protocol/generation/scoring/voice-continuation契约未变。文件系统helper真实CPU核对不产生新模型结果。'})
 claim['artifact_ids'].extend(['a_portability','a_eval_diff'])
 if identifier=='c3':
  claim['scope']+=' 本次callback未运行当前heldout test。原raw heldout记录仍来自旧eval SHA828b7cbb…；code_sha指纹仍绑定精确源版本，当前evaluate SHA变化不会被本次修订绕过或重建旧frozen protocol。'
 if identifier=='c16':
  claim['scope']+=' 当前源voice continuation/generation AST与此前实际CPU契约检查版本相同；callback只复用此未变证据，没有生成真实第一答或第二答。'
report['checks']['source_verification']['details']+=' 另由原审阅者完成当前evaluate.py portability callback：当前源SHA99feca5e…、原源SHA828b7cbb…与精确两行diff均保存；相关模型/来源/续聊契约AST未变，有限路径/fsync helper真实CPU执行通过。Windows原生runtime与当前heldout模型结果未重测，原raw记录保留原版本 provenance。'
report['checks']['limitations']['details']+=' portability callback仅验证Linux fsync fixture及PureWindowsPath序列化，不扩称Windows原生运行或新模型评测已完成。'
report['portability_reinspection']={'reviewer_task':'/root/p6_fact_19_3','date':'2026-10-06','filesystem_available_and_actually_read':True,'previous_evaluate_sha256':previous_sha,'current_evaluate_sha256':current_sha,'section_unchanged_sha256':report['source_sha256'],'exact_changes':['relative paths use as_posix() in code_fingerprints','receipt opened r+b before os.fsync in main'],'reused_evidence_basis':'All other evaluate source bytes and relevant AST contracts identical, course/figure/other cited code sources unchanged. Old measured outputs remain tied to original code version.','actual_execution':'Bounded CPU path serialization/current code_fingerprints and receipt fsync fixture only. No model generation/training/test.','artifact_id':'a_portability','before_report_artifact':'a_portability_prior_report','verdict':'pass'}
(ROOT/'docs/technical-reviews/19.3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'lesson':'19.3','verdict':report['verdict'],'source_sha256':report['source_sha256'],'current_eval_sha256':current_sha,'callback_evidence_saved':True},ensure_ascii=False))
