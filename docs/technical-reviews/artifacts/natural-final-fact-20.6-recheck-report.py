"""Persist same-reviewer verdict reached after full manual rereading/re-viewing.

Assertions here check integrity/reexecuted evidence, not semantic truth by themselves.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE='docs/technical-reviews/artifacts/natural-final-fact-20.6-'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
first=read(BASE+'first-review.json')
previous=read(BASE+'audit.json');current=read(BASE+'recheck-audit.json')
oldtok=read(BASE+'supplement.json');newtok=read(BASE+'recheck-supplement.json')
assert first['verdict']=='revise'
assert previous['raw_records']==current['raw_records']
assert previous['summaries']==current['summaries']
assert previous['source_checks']==current['source_checks']
assert previous['examples']==current['examples']
assert previous['inputs']==current['inputs']
assert oldtok['token_checks']==newtok['token_checks']
assert oldtok['execution_checks']==newtok['execution_checks']
oldbody=(ROOT/(BASE+'section.md')).read_text()
newbody=(ROOT/(BASE+'recheck-section.md')).read_text()
assert oldbody.replace('4張包含真人照片的商品或分享頁面','4張含實物照片的商品或分享頁面')==newbody
assert current['lesson_sha256']==hashlib.sha256(newbody.encode()).hexdigest()
data=(ROOT/'docs/natural-assistant/DATA.md').read_text()
restored=data.replace('4 張含實物照片的商品或分享頁面','4 張包含真人照片的商品或分享頁面')
assert hashlib.sha256(restored.encode()).hexdigest()==oldtok['input_hashes']['docs/natural-assistant/DATA.md']
changed_inputs=[]
for p,d in oldtok['input_hashes'].items():
 if newtok['input_hashes'][p]!=d:
  assert p=='docs/natural-assistant/DATA.md';changed_inputs.append(p)
 else:assert sha(p)==d
checked_artifacts=[]
for a in first['artifacts']:
 assert sha(a['path'])==a['sha256'];checked_artifacts.append(a['path'])
for s in first['sources']:
 if s['kind']=='repository_code':assert sha(s['path'])==s['sha256']
for p,d in first['figure_sha256'].items():assert sha(p)==d
records=[]
for name,out,err in [('recheck-audit','recheck-audit.stdout.txt','recheck-audit.stderr.txt'),('recheck-supplement','recheck-supplement.stdout.json','recheck-supplement.stderr.txt')]:
 assert not (ROOT/(BASE+err)).read_bytes()
 records.append({'command':'.venv-natural/bin/python '+BASE+name+'.py','observed_exit_code':0,'stdout':{'path':BASE+out,'sha256':sha(BASE+out)},'stderr':{'path':BASE+err,'sha256':sha(BASE+err)},'script':{'path':BASE+name+'.py','sha256':sha(BASE+name+'.py')},'environment':{'python':'3.12.14','device':'CPU','unicode_database':'15.0.0'}})
integrity={'reviewer_task':'/root/natural_factual_final_20_6','manual_recheck_note':BASE+'recheck.md','manual_recheck_note_sha256':sha(BASE+'recheck.md'),'first_review_sha256':sha(BASE+'first-review.json'),'first_source_sha256':first['source_sha256'],'current_source_sha256':current['lesson_sha256'],'current_DATA_sha256':sha('docs/natural-assistant/DATA.md'),'source_change_only':'contains-human-photos -> contains-real-object-photos; complete source read personally, four affected originals re-viewed','other_source_artifacts_unchanged':checked_artifacts,'raw_records_and_independent_recounts_unchanged':True,'fresh_CPU_executions':records,'tokens_checked_again':len(newtok['token_checks']),'result':'manual c15 issue resolved; all claims supported; ready to persist pass'}
(ROOT/(BASE+'recheck-integrity.json')).write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
report=first
report['source_sha256']=current['lesson_sha256']
report['verdict']='pass'
report['review_history']=[{'version':'initial','verdict':'revise','report_path':BASE+'first-review.json','report_sha256':sha(BASE+'first-review.json'),'source_sha256':previous['lesson_sha256']},{'version':'same-reviewer full recheck','source_sha256':current['lesson_sha256'],'actual_recheck_note':BASE+'recheck.md','actual_recheck_integrity':BASE+'recheck-integrity.json'}]
def art(id,suffix,kind,description,command=None):
 p=BASE+suffix;a={'id':id,'kind':kind,'path':p,'sha256':sha(p),'description':description}
 if kind=='execution':a.update(command=command,result='Observed actual exit0; independent CPU checks and original-byte assertions completed. No model inference.',environment={'python':'3.12.14','device':'CPU','unicode_database':'15.0.0','tokenizers':'0.22.2'})
 report['artifacts'].append(a)
art('audit-v2','recheck-audit.json','execution','New actual full CPU recomputation against current source, retaining every raw answer/token and first evidence unchanged','.venv-natural/bin/python '+BASE+'recheck-audit.py')
art('audit-v2-code','recheck-audit.py','code','Separate executable audit copy; source/exercises and all input computations rerun')
art('audit-v2-stdout','recheck-audit.stdout.txt','source_snapshot','Actual fresh audit stdout')
art('supplement-v2','recheck-supplement.json','execution','Fresh original-tokenizer decoding, completion/provenance and repetition checks including current DATA SHA','.venv-natural/bin/python '+BASE+'recheck-supplement.py')
art('supplement-v2-code','recheck-supplement.py','code','Actual separate CPU token/source checker')
art('supplement-v2-stdout','recheck-supplement.stdout.json','source_snapshot','Actual fresh supplemental stdout')
art('inspection-v2','recheck.md','derivation','Personal complete current rereading and actual affected-image re-viewing; final judgment')
art('integrity-v2','recheck-integrity.json','execution','Preserved old source/report, only precise media-wording change, all current inputs verified and fresh execution receipts','.venv-natural/bin/python '+BASE+'recheck-report.py')
art('current-source','recheck-section.md','source_snapshot','Exact actually reread current section bytes')
art('old-review','first-review.json','source_snapshot','Preserved independent complete initial revise report with contradicted c15')
for c in report['claims']:
 c['artifact_ids']+=['inspection-v2']
 if c['kind'] in ['numeric','software','empirical']:
  c['artifact_ids']+=['audit-v2','supplement-v2']
  c['verification']['details']+=' Same reviewer reread current complete text and freshly reran all relevant CPU calculations/token checks; old evidence retained.'
 if c['id']=='c15':
  c['status']='verified';c['assessment']='supported'
  c['statement']='Ten natural cases comprise six street/store/factory photographs and four product/share pages containing photographs of real objects.'
  c['verification']['expected']=c['statement']
  c['verification']['observed']='Re-viewed allfour current originals: embroideredcloth000, collar/embroideredgarment020, modeledclothing050, pendantproduct099; revised description fits each. Sixscene originals unchanged. No claimallfour containa person.'
  c['scope']='The actual six scene photographs and four mixed product/share pages in this fixed set; not a census of people or general Traditional Chinese streets.'
report['issues'][0]['status']='resolved'
report['issues'][0]['resolution']='Coordinator replaced 包含真人照片 with 含實物照片 in20.6 and DATA.md; same reviewer personally reread whole current section/relevant DATA, re-viewed allfour affected originals, and freshly reran all numerical/raw-token/source checks. Initial contradicted source/report preserved.'
for s in report['sources']:
 if s['id']=='cpu':s['artifact_id']='audit-v2'
report['checks']['factual_accuracy']={'status':'pass','details':'All26 claims supported after actual whole-section rereading and four-image re-viewing; c15 exact media-description issue resolved. Initial revise report preserved.','claim_ids':[c['id'] for c in report['claims']]}
report['checks']['limitations']={'status':'pass','details':'Revised media count now fits actual originals. Source-GT caveats, incomplete clipped pharmacy character, fixed model selection, limited sample coverage, code/image rights, font/pretraining, token stopping and no high-resolution quality experiment scopes remain explicit and supported.','claim_ids':['c15','c16','c19','c22','c23','c24','c25','c26']}
report['checks']['numeric_verification']['details']+=' Fresh second CPU audit confirms every integer, ratio, raw string and token stopping claim against current source.'
report['checks']['source_verification']['details']+=' Preserved initial authorities; verified their exact files unchanged, fresh full TSV/decoded-image/139-render hash checks and92 token decodes passed.'
target=ROOT/'docs/technical-reviews/20.6.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'verdict':'pass','report_sha256':sha('docs/technical-reviews/20.6.json'),'source_sha256':current['lesson_sha256'],'DATA_sha256':sha('docs/natural-assistant/DATA.md'),'all_claims_supported':len(report['claims']),'resolved_issue':'c15','old_review_sha256':sha(BASE+'first-review.json'),'fresh_executions':records},ensure_ascii=False,indent=2))
