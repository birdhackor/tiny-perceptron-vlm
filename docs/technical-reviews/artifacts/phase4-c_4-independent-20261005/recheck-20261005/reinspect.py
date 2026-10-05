"""Fresh bounded checks of the revised scope, using unchanged original methods/data."""
from pathlib import Path
import ast,hashlib,json,sys,shutil,re

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=HERE/'section.md'
assert sha(source)=='0fd438244f9d23795f0dce18bbb8bef428317f6a1db764109f0d1dca0f7bfec2'
code=ROOT/'scripts/course_experiments/applications.py'
raw=ROOT/'docs/course-experiments/results/reasoning.json'
fig=ROOT/'course/figures/rewrite-C-candidate-choice.svg'
for a,b in [(code,BASE/'frozen-input/scripts/course_experiments/applications.py'),(raw,BASE/'frozen-input/docs/course-experiments/results/reasoning.json'),(fig,BASE/'frozen-input/course/figures/rewrite-C-candidate-choice.svg')]:assert a.read_bytes()==b.read_bytes()
tree=ast.parse(code.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_verify_reasoning')
ns={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),str(code),'exec'),ns)
row={'a':2,'b':2,'c':0,'truth':4}
bad={'generated':'2+2=3;3+0=4;answer=4','invalid_special_tokens':[]}
good={'generated':'2+2=4;4+0=4;answer=4','invalid_special_tokens':[]}
sets=[]
for name,cs in [('final-correct-but-equations-wrong',[bad]),('same-bad-plus-fully-valid',[bad,good]),('fully-valid-only',[good])]:
 checked=[ns['_verify_reasoning'](c,row,'steps') for c in cs]
 cov=any(c['final_correct'] for c in checked)
 answer=next((c['final_answer'] for c in checked if c['fully_verified']),None)
 sets.append({'name':name,'candidates':[c['generated'] for c in cs],'coverage':cov,'verifier_answer':answer,'verifier_correct':answer==4,'fully_verified_flags':[c['fully_verified'] for c in checked]})
assert sets[0]['coverage'] and not sets[0]['verifier_correct']
assert sets[1]['coverage'] and sets[1]['verifier_correct']
assert sets[2]['coverage'] and sets[2]['verifier_correct']
recompute=json.loads((HERE/'recompute.stdout.json').read_text())
assert len(recompute['budgets'])==8
for b in recompute['budgets']:
 assert b['covered']==b['verified_correct']
 assert b['covered_but_no_fully_verified']==0
 assert b['questions']==24
src=ROOT/'outputs/reviewer-tools/phase4-c_4-recheck-20261005'
for name in ['extraction.json','fence-1.py','stdout.txt','stderr.txt','environment.json','execution.json']:
 shutil.copyfile(src/name,HERE/('current-'+name))
assert (HERE/'current-fence-1.py').read_bytes()==(BASE/'fence-1.py').read_bytes()
assert (HERE/'current-stdout.txt').read_text()=='集合包含正解 True\n多數決 3 答對 False\n驗證器 4 答對 True\n'
assert sha(BASE/'initial-report.json')=='81bb0f16f3a422e8e5dc8decaa6e8aa306ebf3edb2829413f5864322ef823025'
out={'reviewer_task':'/root/phase4_factual_coordinator/factual_c_4','source_sha256':sha(source),'reviewer_scope':'Personally read complete revised C.4; exact raw-byte diff only changes supplemental verifier paragraph. C.3/code/raw/figure/papers unchanged and hash-checked where used; original diagram was rendered/viewed in initial review, no new diagram change.','environment':{'python':sys.version,'device':'CPU; no model loaded'},'bounded_original_method_variants':sets,'recomputed_original_budgets':recompute['budgets'],'recheck_commands':[{'command':".venv/bin/python docs/review-tools/section_facts.py 'course/chapters/0C.md#C.4' --output outputs/reviewer-tools/phase4-c_4-recheck-20261005 --execute --timeout 60",'actual_exit_code':0},{'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-c_4-independent-20261005/recompute.py > docs/technical-reviews/artifacts/phase4-c_4-independent-20261005/recheck-20261005/recompute.stdout.json 2> docs/technical-reviews/artifacts/phase4-c_4-independent-20261005/recheck-20261005/recompute.stderr.txt','actual_exit_code':0}],
 'history':{'initial_report_path':str(BASE/'initial-report.json'),'initial_report_sha256':sha(BASE/'initial-report.json'),'initial_source_sha256':sha(BASE/'section.md'),'original_issue':'Unqualified implication from fully_verified selection to final_correct coverage equality','unchanged_original_evidence':{'code_sha256':sha(code),'raw_sha256':sha(raw),'figure_sha256':sha(fig)}},
 'resolution':'New text explicitly separates final-answer coverage from complete equation verification, allows wrong-step/correct-final candidate to count in coverage yet fail verifier, and confines equality to all eight observed groups that each contain a fully valid candidate whenever covered. Independent replay of 720 recorded candidates and fresh bounded bad-only/mixed-valid sets support this exact scope.','outcome':'Resolved; full current C.4 supported within stated narrow scope. No training, new model scores, additional papers or changed figure.'}
print(json.dumps(out,ensure_ascii=False,indent=2))
