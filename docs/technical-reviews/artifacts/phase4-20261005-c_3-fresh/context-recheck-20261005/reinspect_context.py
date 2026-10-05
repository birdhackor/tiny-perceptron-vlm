"""Narrow dependency reinspection: C.4 changed context of own C.3 claims only."""
import hashlib,json,re,sys
from pathlib import Path
import torch
BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
TASK='/root/phase4_factual_coordinator/factual_c_3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial=json.loads((OUT/'initial-pass/C.3.json').read_text())
assert initial['reviewer_task']==TASK
assert sha(OUT/'initial-pass/C.3.json')=='7c97ab87e9827f92b55265565761898de2850fe8227c29751321ffc54fb33305'
assert sha(OUT/'C.3.md')==initial['source_sha256']
old_manifest=json.loads((OUT/'initial-pass/manifest.json').read_text())
for entry in old_manifest['files']:
 assert sha(entry['path'])==entry['sha256']
raw_path=BASE/'originals/docs/course-experiments/results/reasoning.json'
raw=json.loads(raw_path.read_text())
counts=[]
# Inspect only original candidate final/strong criteria fields and their inputs;
# re-score the existing finite sets to check the new context's actual support.
for mode in ['direct','steps']:
 for budget in raw['results']['comparison'][mode]['budgets']:
  covered=strong_covered=selected_correct=0
  final_valid_strong_invalid=0
  for row in budget['samples']:
   truth=row['a']+row['b']+row['c'];assert truth==row['truth']
   finals=[];strong=[];answers=[]
   for candidate in row['candidates']:
    ids=candidate['generated_ids']; payload=ids[:-1] if ids and ids[-1]==2 else ids
    text=bytes(i-8 for i in payload if i>=8).decode('utf-8',errors='replace').strip()
    clean=not any(i<8 for i in payload)
    answer=None;full=False
    if mode=='direct':
     if clean and re.fullmatch(r'-?[0-9]+',text):answer=int(text)
     full=answer==truth
    else:
     tail=re.search(r';answer=(-?[0-9]+)$',text) if clean else None
     answer=int(tail[1]) if tail else None
     m=re.fullmatch(r'(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);(-?[0-9]+)\+(-?[0-9]+)=(-?[0-9]+);answer=(-?[0-9]+)',text) if clean else None
     if m:
      a,b,sub,previous,c,total,final=map(int,m.groups())
      full=(a+b==sub and previous+c==total and previous==sub and total==final and (a,b,c)==(row['a'],row['b'],row['c']) and final==truth)
    final_correct=answer==truth
    assert final_correct==candidate['final_correct'] and full==candidate['fully_verified']
    final_valid_strong_invalid+=int(final_correct and not full)
    finals.append(final_correct);strong.append(full);answers.append(answer)
   hit=any(finals);strong_hit=any(strong)
   chosen=next((answer for answer,valid in zip(answers,strong) if valid),None)
   covered+=hit;strong_covered+=strong_hit;selected_correct+=chosen==truth
   assert not strong_hit or hit
   assert not hit or strong_hit
  assert covered==budget['oracle_coverage']['numerator']
  assert selected_correct==budget['verifier_accuracy']['numerator']
  assert covered==strong_covered==selected_correct
  counts.append({'branch':mode,'k':budget['candidate_count'],'questions':24,'final_answer_coverage':covered,'fully_verified_set_coverage':strong_covered,'first_fully_verified_selection_correct':selected_correct,'recorded_final_correct_but_not_fully_verified_candidates':final_valid_strong_invalid})
receipt={
 'reviewer_task':TASK,'recheck_stage':'own C.3 dependency reinspection after neighboring C.4 paragraph update','verdict':'pass','unresolved_issues':[],
 'initial_history':{'report':str(OUT/'initial-pass/C.3.json'),'report_sha256':sha(OUT/'initial-pass/C.3.json'),'proof_manifest':str(OUT/'initial-pass/manifest.json'),'proof_manifest_sha256':sha(OUT/'initial-pass/manifest.json'),'all_initial_proof_files_hash_checked_unchanged':True,'initial_whole_input':str(BASE/'originals/course/chapters/0C.md'),'initial_whole_sha256':sha(BASE/'originals/course/chapters/0C.md'),'frozen_input_only':True},
 'actual_read_scope':{'current_text':['C.3 complete raw section','C.4 complete raw section; comparison against own initially frozen C.4 section'],'changed_context':'Only C.4 paragraph beginning 精確驗證器能直接計算 changed; self-generated exact diff retained','necessary_current_code':['scripts/course_experiments/applications.py::_verify_reasoning lines707–747','scripts/course_experiments/applications.py::_reasoning_samples lines764–800'],'raw_pointers':['/results/comparison/{direct,steps}/budgets/*/{candidate_count,oracle_coverage,verifier_accuracy}','/results/comparison/{direct,steps}/budgets/*/samples/*/{a,b,c,truth,candidates/{generated_ids,final_correct,fully_verified}}'],'own_prior_evidence':['Own initial PASS report, scope, completion and proof manifest; original source bytes/previous independent raw checks retained'],'other_reviews_read':False,'instructional_figure_reinspection':'None; C.3 has no referenced figure. C.4 figure content/code unchanged and not necessary to changed paragraph dependency; no new whole-C.4 verdict or figure verification asserted.'},
 'current_context':{'whole_snapshot':str(OUT/'current-whole-frozen-input.md'),'whole_sha256':sha(OUT/'current-whole-frozen-input.md'),'meaning':'Current whole Markdown saved as a new frozen input; no patch to previous whole SHA','C.3':{'path':str(OUT/'C.3.md'),'sha256':sha(OUT/'C.3.md'),'unchanged':True},'C.4':{'path':str(OUT/'C.4.md'),'sha256':sha(OUT/'C.4.md')},'change_diff':str(OUT/'C.4-change.diff')},
 'independent_support':{'original_raw_sha256':sha(raw_path),'eight_existing_set_counts':counts,'mechanism':'final_correct requires only parsed final integer==truth; fully_verified additionally requires equations, linkage and original operands. The equality of verifier selection and oracle coverage is an observed property of these eight stored budgets, not guaranteed by candidate coverage alone. No train or model-generation operation occurred.'},
 'claim_impact':{'oracle-table-and-denominators':'No change: its initial scope already says correct final answer, not fully verified steps or deployed-selection accuracy. New neighboring distinction is consistent; original C.3 numerators/denominators/source bytes unchanged.','finite-coverage-scope':'No change: its initial claim explicitly distinguishes set membership from selected output and never asserts universal verifier==coverage. Current C.4 now limits that equality to existing eight-budget records; independent narrow raw reinspection confirms this finite support.','other_six_claims':'No dependency on changed C.4 paragraph: iid/math/literal-fence/pass@k/training and sampling conditions depend on unchanged own bytes, primary sources and original code/raw evidence.'},
 'scope_limit':'This is own C.3 context reinspection, not a replacement or approval of C.4 review. No old/other reviewer report read; no unrelated original fence/CPU/GPU/model generation/train rerun; no source/figure/other report edits.',
 'command':'.venv/bin/python docs/technical-reviews/artifacts/phase4-20261005-c_3-fresh/context-recheck-20261005/reinspect_context.py',
 'environment':{'python':sys.version,'torch':str(torch.__version__),'device':'cpu','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available())},'actual_exit_code':0
}
(OUT/'context-reinspection.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'reviewer_task':TASK,'initial_proof_hashes_unchanged':True,'C.3_unchanged':True,'C.4_sha256':sha(OUT/'C.4.md'),'counts':counts,'claim_impact':'none; changed context agrees with existing own C.3 final-answer coverage scope','verdict':'pass','unresolved_issues':0},ensure_ascii=False))
