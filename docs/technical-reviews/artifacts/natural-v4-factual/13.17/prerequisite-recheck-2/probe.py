"""Own affected-prerequisite syntax audit; no model training, no figure rereading."""
from pathlib import Path
import hashlib
import io
import json
import platform
import re
from contextlib import redirect_stdout

ROOT=Path(__file__).resolve().parents[6]
ART=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
old=(ART/'prior-closure-current-prerequisite-13.16.md').read_bytes()
new=(ART/'current-13.16.md').read_bytes()
block=lambda b:re.search(rb'```python\n(.*?)\n```',b,re.S).group(1).decode()
assert block(old)==block(new)
predictions={'range_values':[0,1],'scores':[1,14],'chosen':1,'correct_and_number_only':False,
             'rule_exercise_scores':[1,0],'rule_exercise_chosen':0}
ns={};stdout=io.StringIO()
with redirect_stdout(stdout):exec(compile(block(new),'current13.16-prerequisite-recheck2','exec'),ns)
actual={'range_type':type(range(len(ns['answers']))).__name__,
        'range_values':list(range(len(ns['answers']))),'scores':ns['scores'],
        'lambda_results':[(lambda index:ns['scores'][index])(i) for i in range(len(ns['answers']))],
        'chosen':ns['chosen'],'chosen_text':ns['answers'][ns['chosen']],
        'correct_and_number_only':ns['answers'][ns['chosen']]==str(1+2)}
for key in ['range_values','scores','chosen','correct_and_number_only']:assert actual[key]==predictions[key]
assert actual['lambda_results']==actual['scores']
assert actual['chosen']==1 and actual['scores'][actual['chosen']]==14
exercise=block(new).replace('len(answer)','int(answer == str(1 + 2))')
ex={};exercise_stdout=io.StringIO()
with redirect_stdout(exercise_stdout):exec(compile(exercise,'13.16-rule-exercise-recheck2','exec'),ex)
assert ex['scores']==predictions['rule_exercise_scores'] and ex['chosen']==predictions['rule_exercise_chosen']
prior=json.loads((ART/'prior-current-pass.json').read_text())
original_evidence_integrity=json.loads((ART/'prior-evidence-integrity.json').read_text())
original=(ROOT/'docs/technical-reviews/artifacts/natural-v4-factual/13.17/original-section.md').read_bytes()
chapter=(ROOT/'course/chapters/13.md').read_bytes();heads=list(re.finditer(rb'^## ',chapter,re.M))
current_main=next(chapter[m.start():heads[i+1].start() if i+1<len(heads) else len(chapter)]
 for i,m in enumerate(heads) if chapter[m.start():].startswith(b'## 13.17 '))
assert current_main==original and sha(current_main)==prior['source_sha256']
for path,digest in prior['figure_sha256'].items():assert sha((ROOT/path).read_bytes())==digest
assert len(prior['claims'])==22 and all(c['status']=='verified' for c in prior['claims'])
audit=json.loads((ART/'prior-audit-result.json').read_text())
assert audit['independent_results']['test']['ppo']['total']==[12,18]
assert audit['independent_results']['test']['dpo']['total']==[12,18]
out={'reviewer_task':'/root/v4_review_coordinator/factual_v4_13_17',
  'method':'Complete new13.16 rawUTF8 read; exact diff against own prior; authority inspection plus short actual Python snippet/exercise execution.',
  'environment':{'python':platform.python_version(),'device':'cpu','training':'none'},
  'prior_pass_sha256':sha((ART/'prior-current-pass.json').read_bytes()),
  'prior_prerequisite13_16_sha256':sha(old),'current_prerequisite13_16_sha256':sha(new),
  'main_source_sha256':sha(current_main),'main_bytes_identical':True,
  'predictions_before_probe':predictions,'actual':actual,
  'exact_current_code_stdout':stdout.getvalue(),'exercise_scores':ex['scores'],'exercise_chosen':ex['chosen'],
  'exercise_stdout':exercise_stdout.getvalue(),'current_and_prior_python_block_identical':True,
  'all_original_code_and_record_fingerprints_unchanged':all(not s['changed'] for s in original_evidence_integrity['sources'] if s['path']!='course/chapters/13.md'),
  'all_22_main_claims_preserved':True,'closest_affected_main_claims':['c16','c18','c21'],
  'new_main_claims_needed':False,'remaining_issues':[],
  'figure_evidence':'All5 SVG bytes match own prior map. Reuse own preserved prior renders and personalview observations; no new render/view claimed.',
  'cpu_evidence':'Reuse own existing native CPU training, derivations, original record reconstruction and denominator evidence; no new model training.',
  'scope_judgment':'Added range/key/lambda sentences are accurate for the unchanged two-answer code. range provides indices0,1 as a range object, not an eager list; max key selects original index1 using scores1/14; lambda is a one-expression function. The unchanged counterexample and task table still support13.17 loss-vs-task-result limits. No contradictory/unresolved13.17 fact introduced.'}
(ART/'probe-result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
