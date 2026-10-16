"""Late informed continuity-phase technical source-link callback only."""
import copy
import difflib
import hashlib
import json
import platform
import sys
from datetime import UTC,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[6]
sys.path.insert(0,str(ROOT))
from scripts.reading_time import lesson_slices
from scripts.check_technical_reviews import _validate
ART=Path(__file__).parent
TASK='/root/backfill_tech_curriculum'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads((ROOT/p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
previous_path='docs/course-revision-20261016/reviews/technical-order-curriculum-callback-01.json'
previous=load(previous_path)
previous_sha=sha(ROOT/previous_path)
assert previous_sha=='59ad1ec11e65f5a8aa3473d90b18705ec8d9f2f68a3cb68627ee0406f1cbcbdb'
f18=load('docs/course-revision-20261016/freeze-18.json');f19=load('docs/course-revision-20261016/freeze-19.json')
p18={p['page_id']:p for p in f18['pages']};p19={p['page_id']:p for p in f19['pages']}
a=(ROOT/p18['curriculum']['snapshot']).read_text();b=(ROOT/p19['curriculum']['snapshot']).read_text()
old='課程索引按下表區分「主題初讀」與「按需回讀」'
new='課程索引依[閱讀指南](../course/README.md#R.2)區分「主題初讀」與「按需回讀」'
assert a.count(old)==1 and b==a.replace(old,new)
assert b==(ROOT/'docs/curriculum.md').read_text() and sha(ROOT/p19['curriculum']['snapshot'])==p19['curriculum']['source_sha256']
assert set(p18)==set(p19)
other_same=[]
for id in p19:
 if id!='curriculum':
  assert (ROOT/p18[id]['snapshot']).read_bytes()==(ROOT/p19[id]['snapshot']).read_bytes()
  other_same.append(id)
assert len(other_same)==349
ref='../course/README.md';target=(ROOT/'docs'/ref).resolve()
assert target==ROOT/'course/README.md'
guide=target.read_text();frozen_guide=(ROOT/p19['course']['snapshot']).read_text()
assert guide==frozen_guide and p18['course']['source_sha256']==p19['course']['source_sha256']
r2=lesson_slices(guide)['R.2']
assert r2.startswith('## R.2 ') and '第一次接觸語言模型' in r2 and '各主題也可以分開學' in r2 and '補一個必要步驟，再回到原問題' in r2
source_checks={}
for s in previous['sources']:
 if s['kind']=='repository_code':
  actual=sha(ROOT/s['path']);source_checks[s['id']]={'path':s['path'],'expected_sha256':s['sha256'],'actual_sha256':actual,'matches':actual==s['sha256']}
assert all(x['matches'] for x in source_checks.values())
artifact_checks={}
for x in previous['artifacts']:
 actual=sha(ROOT/x['path']);artifact_checks[x['id']]={'path':x['path'],'expected_sha256':x['sha256'],'actual_sha256':actual,'matches':actual==x['sha256']}
assert all(x['matches'] for x in artifact_checks.values())
assert p18['curriculum']['figures_sha256']==p19['curriculum']['figures_sha256']
for p,h in p19['curriculum']['figures_sha256'].items():assert sha(ROOT/p)==h
inspection={
 'reviewer_task':TASK,'current_context':'informed','collection_phase':'continuity','inspected_at':datetime.now(UTC).isoformat(),
 'scope':'Only the late source-link sentence, its neighboring paragraphs and the R.2 target were reread. The remainder of curriculum was compared byte-for-byte to the prior owner-reviewed page. This is not a new full technical, reader or continuity review.',
 'command':'.venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/curriculum/inspect-callback-02.py',
 'environment':{'python':platform.python_version(),'device':'cpu','operation':'read-only source/link/diff/hash checks; report archival'},
 'previous_report':{'path':previous_path,'sha256':previous_sha},
 'current_freeze':{'path':'docs/course-revision-20261016/freeze-19.json','sha256':sha(ROOT/'docs/course-revision-20261016/freeze-19.json')},
 'current_page':p19['curriculum'],'diff':list(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='freeze18',tofile='freeze19')),
 'single_exact_replacement':{'old':old,'new':new,'rest_byte_identical':True},
 'neighboring_text':'\n'.join(b.splitlines()[99:112]),
 'link_check':{'relative_target':ref,'resolved_repo_target':str(target.relative_to(ROOT)),'anchor':'R.2','anchor_heading':r2.splitlines()[0],'target_snapshot':p19['course']['snapshot'],'target_source_sha256':p19['course']['source_sha256'],'target_section_sha256':hashlib.sha256(r2.encode()).hexdigest(),'target_live_equals_frozen':True,'target_excerpt_first_read':'第一次接觸語言模型，先沿第 1–4 章閱讀','target_excerpt_topic_start':'各主題也可以分開學','target_excerpt_reread':'跳讀遇到陌生概念時，使用正文中的回讀連結與小節末尾的補充連結，補一個必要步驟，再回到原問題。'},
 'other_canonical_pages_byte_identical':len(other_same),'previous_repository_source_pins':source_checks,'previous_artifact_pins':artifact_checks,'unchanged_figure_map':True,
 'independent_interpretation':'The edited sentence now names an actual source rather than a nonexistent following table. R.2 supplies an initial foundational route, a topic-start table and explicit on-demand prerequisite rereading. This supports the sentence\'s routing distinction. The adjacent R.3 sequential-route pointer stays unchanged. No new method, numeric, empirical, model-capability or hardware claim is introduced.',
 'claim_disposition':{'count':40,'claim_statements_unchanged':True,'c15':'Existing route/software claim gets this additional narrow source-link evidence; chapter order evidence remains inherited from callback01.','remaining_claims':'Unchanged statements and previous evidence inherited; no earlier execution or full reading claimed again.'},
 'not_done':['training','formal test','GPU or broad benchmark','HF access or writes','Colab or notebook kernel rerun','new external source retrieval','new figure render','reading other reviewers reports'],
 'necessary_or_optional_issues':[]}
proof_path=ART/'callback-02-inspection.json';dump(proof_path,inspection)
r=copy.deepcopy(previous)
r['current_context']='informed';r['source_sha256']=p19['curriculum']['source_sha256'];r['figure_sha256']=p19['curriculum']['figures_sha256']
r['callback_history']=[{'path':previous_path,'sha256':previous_sha,'scope':'Previous informed order callback; preserved byte-for-byte.'}]
r['callback']={'kind':'late_source_pointer_informed_owner_callback','collection_phase':'continuity','reviewed_at':datetime.now(UTC).isoformat(),'freeze':inspection['current_freeze'],'original':previous['callback']['original'],'previous_callback':inspection['previous_report'],'current_source_snapshot':p19['curriculum']['snapshot'],'current_source_sha256':r['source_sha256'],'scope':inspection['scope'],'claim_disposition':inspection['claim_disposition'],'issues_or_optional_suggestions':[]}
r['scope_note']='Current freeze19 overall curriculum verdict is carried from this owner\'s original and callback01 evidence, after a late continuity-phase narrow source-pointer correction. reviewer_context=fresh records only the original independent review; current_context=informed applies now. This callback rereads only the changed sentence, neighbors and R.2 target; the remainder of curriculum is byte-identical to callback01. Neither prior report is rewritten.'
r['verification_boundary']='This callback executed source/link/diff/hash checks only. All40claimstatements are unchanged. Existing full technical checks, external reads, route validation, source/notebook synchronization, figure viewing and measured/model evidence remain explicitly inherited from original/callback01, not rerun. No training, formaltest, GPU/HF/Colab operation or broad benchmark was performed.'
for source in r['sources']:
 if source['id']=='sBody':source.update(path=p19['curriculum']['snapshot'],title=p19['curriculum']['snapshot'],sha256=r['source_sha256'],version='freeze19 canonical snapshot; late informed source-link callback2026-10-16',inspection_note='This callback reread the only changed sentence plus neighbors. R.2target actually read and resolved. Exact remainder-byte equality against callback01wholepage verified; earlier whole-page technical reading and all40statements inherited. No claim of another whole-page reread.')
 else:source['late_callback_provenance']='Source remains the previous owner-checked bytes/version; repository SHA rechecked in callback02. Original external source reads and executions were not repeated.'
for c in r['claims']:
 c['late_owner_callback']={'previous_report':previous_path,'previous_report_sha256':previous_sha,'disposition':'Unchanged statement and evidence inherited; no new full technical revalidation.'}
 if c['id']=='c15':
  c['artifact_ids'].extend(['aLateSource','aLateSourceCode'])
  c['evidence'].append({'source_id':'sLateSourceExec','locator':'callback-02-inspection.link_check and single_exact_replacement','supports':'Edited routing sentence resolves to existing current R.2, which provides initial/topic starts and explicit on-demand rereading; all other curriculum bytes unchanged.'})
  c['verification']['details']+=' Callback02adds only actual source-link/neighbor/R.2checks. Earlier order/config executions are inherited and were not run again.'
  c['late_owner_callback']['disposition']='Unchanged route claim, with additional narrow R.2source-pointer evidence.'
r['sources'].append({'id':'sLateSourceExec','kind':'execution','title':'Late curriculum source-pointer check','verified':True,'artifact_id':'aLateSource'})
r['artifacts'].extend([
 {'id':'aLateSource','kind':'execution','path':str(proof_path.relative_to(ROOT)),'sha256':sha(proof_path),'description':'Narrow changed-sentence/neighbor/R.2check, exact rest-byte equality, oldsource/artifactpins and unchangedfiguremap.','command':inspection['command'],'result':'Single exact source-pointer replacement; R.2anchor and routing meanings verified; remaining curriculum and349othercanonicalpages byte-identical; allprevioussource/artifactpinsmatch.','environment':{'python':platform.python_version(),'device':'cpu','operation':'read-only source/link/hash inspection'}},
 {'id':'aLateSourceCode','kind':'code','path':str(Path(__file__).relative_to(ROOT)),'sha256':sha(Path(__file__)),'description':'Exact late callback source-link and preservation inspection code.'}])
r['issues']=[]
r['checks']={
 'factual_accuracy':{'status':'pass','details':'All40statementspreservedfromthepriorownerreport. Thiscallbackonlychecks the edited routing pointer and neighboring context, with the rest of curriculum byte-identical. R.2actually supplies topic starts and on-demand rereading.','claim_ids':[c['id'] for c in r['claims']]},
 'numeric_verification':{'status':'pass','details':'No numeric or empirical text changed. Prior numerical verification and measured evidence are inherited with unchangedsource/artifactpins; no arithmetic or experiment rerun claimed.','claim_ids':previous['checks']['numeric_verification']['claim_ids']},
 'figure_consistency':{'status':'pass','details':'SameSVG and figuremap; earlier personal render/view evidence inherited. The source-pointer edit does not change the diagram or its interpretation. No new render performed.','claim_ids':['c31']},
 'source_verification':{'status':'pass','details':'Current sBody points to actually inspected freeze19snapshot. Relative link resolves to current course/README.md#R.2; target heading and intended routing content were read. All prior repositorysource/artifactpins match; remaining curriculum bytes are identical.','claim_ids':[c['id'] for c in r['claims']]},
 'limitations':{'status':'pass','details':'Late informed callback collected during continuity phase, not a replacement for earlier technical seal or a new blind/full review. Scope limited to source pointer/context/R.2andpreservationchecks. No capability, resource, denominator or model evidence strengthened; no training/formaltest/HFactivity.','claim_ids':previous['checks']['limitations']['claim_ids']}}
out=ROOT/'docs/course-revision-20261016/reviews/technical-order-curriculum-callback-02.json'
assert not out.exists(),'Refusing to overwrite a previous callback02.'
dump(out,r)
# Required actual disk reread before validation and sealing.
actual=json.loads(out.read_text())
errors=_validate(ROOT,ROOT/'docs/curriculum.md','curriculum',b,actual,{TASK:['curriculum']},set())
assert not errors,errors
assert sha(ROOT/previous_path)==previous_sha
original=previous['callback']['original'];assert sha(ROOT/original['path'])==original['sha256']
seal={'reviewer_task':TASK,'current_context':'informed','collection_phase':'continuity','completed_at':datetime.now(UTC).isoformat(),'format_validation_errors':errors,'report':{'path':str(out.relative_to(ROOT)),'sha256':sha(out)},'claims':len(actual['claims']),'sources':len(actual['sources']),'artifacts':len(actual['artifacts']),'source_sha256':r['source_sha256'],'previous_callback_unchanged':True,'original_unchanged':True,'scope':'Only late source-pointer, neighbor/R.2andrest-byte-equalitychecks; prior whole-page technical evidence inherited.'}
dump(ART/'callback-02-validation-seal.json',seal)
print(json.dumps(seal,ensure_ascii=False,indent=2))
