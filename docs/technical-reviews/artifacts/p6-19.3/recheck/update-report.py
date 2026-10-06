from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
ART=OUT.parent
REL=OUT.relative_to(ROOT).as_posix()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report=json.loads((OUT/'initial-19.3-report.json').read_bytes())
assert report['verdict']=='revise' and report['reviewer_task']=='/root/p6_fact_19_3'
assert '"verdict": "pass"' in (OUT/'recheck-output.txt').read_text()
report['source_sha256']=sha(OUT/'current-section-19.3.md')
report['verdict']='pass'
for a in report['artifacts']:
 if a['id']=='a_section':a['description']='初次revise审阅的未正规化19.3原文bytes，SHA为6dd32a81…；当前正式source_sha256对应a_recheck_section，不把旧frozen输入改成新版本。'
def artifact(identifier,name,kind,description,**extra):
 report['artifacts'].append({'id':identifier,'kind':kind,'path':f'{REL}/{name}','sha256':sha(OUT/name),'description':description,**extra})
artifact('a_initial_report','initial-19.3-report.json','source_snapshot','原审阅者自己的初次revise完整JSON原bytes备份，保留c8反例、当时措辞/判定/原节SHA与其他已核实主张。')
artifact('a_recheck_section','current-section-19.3.md','source_snapshot','完整重读的修订后19.3 UTF-8原bytes；当前正式source_sha256由本文件给出。')
artifact('a_recheck_frozen','current-frozen-19.md','source_snapshot','本次复查时整章原bytes frozen input；全章SHA只指此明确snapshot，其他小节变化不替代本节核对。')
artifact('a_recheck_code','recheck.py','code','原审阅者复查用有限CPU代码：正文唯一delta、全部旧证据/代码/图SHA、同一固定package、12记录身份及OCR增强父家族/ROI核对。')
artifact('a_recheck','recheck-output.txt','execution','真实复查输出，保留PNG碰撞，验证新限定来源家族的承诺；没有训练或模型generation/eval。',command='/workspace/tiny-perceptron-vlm/.venv/bin/python docs/technical-reviews/artifacts/p6-19.3/recheck/recheck.py > docs/technical-reviews/artifacts/p6-19.3/recheck/recheck-output.txt 2>&1',result='exit 0；完整当前19.3仅修字卡表一处；旧来源、artifacts、图全SHA相同，固定archive与12record身份相同；786原OCR train parent与3930增强children同家族/target/ROI，所有跨份group交集0；旧单字左PNG碰撞仍在且属于不同家族；复查pass。',environment={'python':'3.13.5','device':'cpu','model_training_generation_or_heldout_eval':'none'})
report['sources'].append({'id':'s_recheck','kind':'execution','title':'Original 19.3 reviewer reinspection of the source-family wording repair','verified':True,'artifact_id':'a_recheck'})
c8=next(c for c in report['claims'] if c['id']=='c8')
c8['statement']='字卡同一来源家族的区域问题及训练增强版本留在同一份。'
c8['location']='19.md L141「同一來源家族的區域問題及訓練增強版本留在同一份」'
c8['status']='verified'
c8['evidence'].append({'source_id':'s_recheck','locator':'recheck-output section_delta, family_contract_recheck, collision_preserved','supports':'完整重读当前19.3，实际复查所有3930增强children仍在786原parent训练家族，保留target/ROI；跨split groups零交集。原碰撞两记录不同source family，不与目前限定相矛盾。'})
c8['verification']={'method':'executed','expected':'同一来源家族及其区域/增强衍生留同份；不宣称所有独立单字render PNG像素零重複。','observed':'3930 augmented train children均有原train parent group且target/ROI相同；train/validation/test家族group交集0。两张原单字左PNG仍完全相同但source groups不同。','details':'原审阅者完整重读当前19.3后，执行recheck.py比对唯一正文delta、旧所有证据hash、固定package/12record identities及augmentation source family关系；没有产生新数据、模型答案或heldout评测。'}
c8['scope']='明确保证按生成来源家族/谱系留份，不保证所有独立生成的既知单字canvas像素全局唯一；原train/validation左PNG碰撞作为既有资料范围限制继续保留。2–4字组合字串隔离、12既知字与两字体训练曝光的范围没有变化。'
c8['artifact_ids'].extend(['a_initial_report','a_recheck_section','a_recheck'])
c9=next(c for c in report['claims'] if c['id']=='c9')
c9['scope']='只支持2–4字target字符串与来源谱系隔离；不能据image path不同推出所有单字canvas像素不同。c8的初次反例完整保留于a_initial_report及碰撞artifacts，当前文本已明确限定source-family承诺。'
issue=report['issues'][0]
issue['status']='resolved'
issue['resolution']='2026-10-06，原审阅者/root/p6_fact_19_3完整重读修订后的19.3。L141改为「2–4字組合按字串隔離；同一來源家族的區域問題及訓練增強版本留在同一份」。实际复查3930增强记录保留786原训练来源家族、target/ROI，各split家族交集0；措辞现在限制来源家族隔离，不再暗示all PNG/pixels零重複。12已知单字、Sans/Serif都已训练、显式ROI识别的限制保持。两条原左字记录/两相同PNG、最初revise报告与初次源SHA全部保留；数据、model/code/figure未变，旧证据hash核对后复用。'
for name in ['factual_accuracy','limitations']:report['checks'][name]['status']='pass'
report['checks']['factual_accuracy']['details']='21项主张均已核实；原c8完整canvas隔离过宽措辞由作者改为来源家族隔离，原审阅者完整复读当前节并CPU查核3930增强parent关系后确认修复。初次PNG碰撞与revise历史未删除。'
report['checks']['limitations']['details']='当前文本明确source-family切分，不再宣称所有单字PNG像素零重複；既知12字、已见Serif、给定ROI辨识、录音不等于人数、零group交集不等于模型使用材料的范围完整保留。原PNG碰撞仍记录为资料限制。'
report['checks']['source_verification']['details']+=' 修订后原审阅者核对所有旧代码/证据/figure SHA及固定package与12records身份均未变后复用；新增复查证据有完整真实命令与输出。'
report['reinspection']={'reviewer_task':'/root/p6_fact_19_3','date':'2026-10-06','full_current_section_read':True,'initial_verdict':'revise','initial_section_sha256':'6dd32a81acea2b64ecf848e15eed40f0f28e35dcc2e750c62e956abfaa6ae70b','current_section_sha256':report['source_sha256'],'issue_resolution_artifact':'a_recheck','data_code_model_figure_mutation':'none by reviewer; identical data/code/figure hashes checked','other_reviews_or_author_notes_read':'none','initial_review_preserved_artifact':'a_initial_report'}
(ROOT/'docs/technical-reviews/19.3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'lesson':'19.3','verdict':'pass','claims':len(report['claims']),'resolved_issue':'c8','source_sha256':report['source_sha256'],'prior_revize_evidence_retained':True},ensure_ascii=False))
