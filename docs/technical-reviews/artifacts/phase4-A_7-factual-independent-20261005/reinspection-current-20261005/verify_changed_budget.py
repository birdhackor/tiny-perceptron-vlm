"""Only the changed A.7 input-over-capacity arithmetic and unchanged evidence hashes."""
import ast
import hashlib
import json
import platform
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=HERE.parent
EXPECTED='32397500c886a31262e690ab72dd5ba34e5fc2936e5dbac9bebb9547d23bb58d'
digest=lambda raw:hashlib.sha256(raw).hexdigest()
prior=json.loads((HERE/'prior-report.opaque.json').read_bytes())
assert prior['reviewer_task']=='/root/phase4_factual_coordinator/factual_a_7'
raw=(HERE/'current-section.md').read_bytes()
assert digest(raw)==EXPECTED
fence=raw.split(b'```python\n',1)[1].split(b'```',1)[0]
assert digest(fence)==digest((BASE/'execution/fence-1.py').read_bytes())
tree=ast.parse(fence.decode())
budget_node=next(n for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='budget' for t in n.targets))
budget=ast.literal_eval(budget_node.value)
limit_node=next(n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='limit' for t in n.targets))
C=ast.literal_eval(limit_node.value)
P_before=sum(v for k,v in budget.items() if k!='answer')
P_after=P_before+30
G=budget['answer']
assert (P_before,P_after,G,C)==(75,105,25,100)
assert P_after+0-C==5
assert all(P_after+g>C for g in range(26))
# Mathematical monotonicity: for every G>=0, P+G >= P > C.
minimum_input_removed_with_G0=max(0,P_after-C)
minimum_input_removed_with_G25=max(0,P_after+G-C)
assert minimum_input_removed_with_G0==5
assert minimum_input_removed_with_G25==30
assert P_after-30+G==C
assert P_after-29+G==C+1
assert P_after-5==C
artifact_reuse=[]
for a in prior['artifacts']:
    actual=digest((ROOT/a['path']).read_bytes())
    assert actual==a['sha256'],a['id']
    artifact_reuse.append({'id':a['id'],'path':a['path'],'sha256':actual,'hash_match':True,
                          'reuse_scope':'Own prior evidence; not claimed to be newly executed/rendered.'})
live_checks=[]
for live,frozen in [('scripts/course_experiments/applications.py','inputs/scripts/course_experiments/applications.py'),
                    ('scripts/course_experiments/common.py','inputs/scripts/course_experiments/common.py'),
                    ('tiny_perceptron/data.py','inputs/tiny_perceptron/data.py'),
                    ('tiny_perceptron/model.py','inputs/tiny_perceptron/model.py'),
                    ('tiny_perceptron/tokenization.py','inputs/tiny_perceptron/tokenization.py'),
                    ('docs/course-experiments/results/rag.json','inputs/docs/course-experiments/results/rag.json')]:
    actual=digest((ROOT/live).read_bytes()); saved=digest((BASE/frozen).read_bytes())
    assert actual==saved,live
    live_checks.append({'live_path':live,'own_frozen_path':(BASE/frozen).relative_to(ROOT).as_posix(),
                        'sha256':actual,'unchanged':True})
result={'reviewer_task':prior['reviewer_task'],'source_sha256':EXPECTED,
 'method':'AST literal values from current unchanged fence; independent integer arithmetic; no model execution',
 'prior_report':{'path':(HERE/'prior-report.opaque.json').relative_to(ROOT).as_posix(),
                 'sha256':digest((HERE/'prior-report.opaque.json').read_bytes()),
                 'source_sha256':prior['source_sha256'],'prior_verdict':prior['verdict']},
 'fence_sha256':digest(fence),'arithmetic':{'C':C,'P_before':P_before,'P_after_history_plus30':P_after,
 'reserve_G':G,'input_excess_even_without_answer':5,'minimum_input_removal_without_answer':5,
 'minimum_input_removal_preserving_G25':30,'removal29_with_G25_total':101,
 'removal30_with_G25_total':100,'tested_nonnegative_reserves_0_through25':26,
 'all_26_infeasible_without_input_removal':True,'proof_for_all_nonnegative_G':'P+G>=P=105>C=100'},
 'same_fence':True,'prior_artifact_hash_checks':artifact_reuse,'live_code_and_raw_hash_checks':live_checks,
 'environment':{'python':sys.version,'platform':platform.platform(),'device':'cpu','framework':'Python standard library only'},
 'new_original_raw_leaves_read':[],
 'scope':'New claim: input105 cannot be repaired by reducing answer reservation alone, and preserving25 requires30 input deletions. Unchanged 84-sample measurements/primary-source evidence reused after exact hash verification; no broad rerun or download.'}
(HERE/'changed-budget-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'source_sha256':EXPECTED,'arithmetic':result['arithmetic'],
 'prior_artifacts_verified':len(artifact_reuse),'live_code_raw_files_verified':len(live_checks),
 'environment':result['environment']},ensure_ascii=False,indent=2))
