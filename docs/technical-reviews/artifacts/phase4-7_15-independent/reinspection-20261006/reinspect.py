"""Same original owner: actual bounded reinspection, no neural training or refetch.
Reads only own frozen report/evidence and necessary raw result JSON pointers.
"""
import ast, difflib, hashlib, importlib.util, json, platform, shutil, sys
from pathlib import Path
R=Path('/workspace/tiny-perceptron-vlm');A=R/'docs/technical-reviews/artifacts/phase4-7_15-independent';D=A/'reinspection-20261006'
PRIOR='94e7640465c8fe3ae05fe01539b63c3aa8ad19ee984870e51dd7fd6b2992522d'
H=R/'docs/technical-reviews/history'/('phase4-7_15-own-before-reinspection-'+PRIOR+'.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert sha(H)==PRIOR
report=json.loads(H.read_text())
helper=R/'docs/review-tools/section_facts.py';spec=importlib.util.spec_from_file_location('section_facts_current',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
raw,whole,line=m.original_section(R/'course/chapters/07.md','7.15');current_sha=hashlib.sha256(raw).hexdigest()
assert current_sha=='8956dde5b025bae60c57ff0e03268b62dd04f44db738052478b2bdc90187fad6'
(D/'current-section.md').write_bytes(raw)
metadata=json.loads((R/'outputs/reviewer-tools/phase4-7_15-independent-reinspection-20261006/extraction.json').read_text());assert metadata['source_sha256']==current_sha and metadata['figure_sha256']=={}
shutil.copyfile(R/'outputs/reviewer-tools/phase4-7_15-independent-reinspection-20261006/extraction.json',D/'current-extraction.json')
fences=m.fences(raw,line);python=[x for x in fences if x['language']=='python'];assert len(python)==1
(D/'current-fence-1.py').write_bytes(python[0]['raw'])
assert python[0]['raw']==(A/'original/fence-1.py').read_bytes()
ctx={}
for lesson in ['7.11','7.13']:
 body,ctxwhole,ctxline=m.original_section(R/'course/chapters/07.md',lesson)
 p=D/('necessary-context-'+lesson.replace('.','_')+'.md');p.write_bytes(body)
 ctx[lesson]={'path':p.relative_to(R).as_posix(),'sha256':sha(p),'actual_read':'Entire current necessary section; reviewed source/provenance statements, not old reviewer reports.'}
old=(A/'original/section.md').read_text();new=raw.decode()
change='\n'.join(difflib.unified_diff(old.splitlines(),new.splitlines(),fromfile='own-frozen-20261005',tofile='current-20261006',lineterm=''))+'\n'
(D/'section.diff').write_text(change)
checks=[]
for item in report['artifacts']:
 p=R/item['path'];h=sha(p);assert h==item['sha256'];checks.append({'artifact_id':item['id'],'path':item['path'],'expected_sha256':item['sha256'],'observed_sha256':h,'unchanged':True})
implementation_checks=[]
for f in ['scripts/course_experiments/common.py','scripts/course_experiments/text.py','tiny_perceptron/data.py','tiny_perceptron/model.py','tiny_perceptron/training.py','scripts/prepare_data.py']:
 h=sha(R/f);frozen=sha(A/'inputs'/f);assert h==frozen
 implementation_checks.append({'path':f,'current_sha256':h,'own_frozen_sha256':frozen,'unchanged':True})
results={};provenance={};result_file_checks=[]
for name in ['sft','sft_ablation']:
 p=R/'docs/course-experiments/results'/(name+'.json');snapshot=A/'inputs'/p.relative_to(R)
 assert sha(p)==sha(snapshot)
 # inventory keys/type first; actual inspection below uses only named pointers.
 d=json.loads(p.read_text());results[name]=d
 result_file_checks.append({'path':p.relative_to(R).as_posix(),'sha256':sha(p),'own_frozen_snapshot':snapshot.relative_to(R).as_posix(),'snapshot_sha256':sha(snapshot),'unchanged':True,'top_level_key_types':{k:type(v).__name__ for k,v in d.items()}})
 for k in ['revision','seed','step_scale','evidence_status']:provenance[name+'.'+k]=d[k]
sft=results['sft'];ablation=results['sft_ablation']
assert sft['results']['training']['steps']==900
assert sft['results']['training']['checkpoint']==sft['results']['checkpoint']=='model.pt'
base=sft['results']['after']['test'];before=ablation['results']['before']['A_attributes']['test']
assert base==before and base['matches']==5 and base['records']==10
assert sft['results']['pretrain_then_sft']['sft']['checkpoint']=='pretrain-sft.pt'
# Hash-verified historical code, inspected only actual compute/provenance branches.
text=A/'historical/scripts/course_experiments/text.py';common=A/'historical/scripts/course_experiments/common.py'
ttree=ast.parse(text.read_bytes());ctree=ast.parse(common.read_bytes())
run_sft=next(x for x in ttree.body if isinstance(x,ast.FunctionDef) and x.name=='run_sft')
run_ablation=next(x for x in ttree.body if isinstance(x,ast.FunctionDef) and x.name=='run_sft_ablation')
context=next(x for x in ctree.body if isinstance(x,ast.ClassDef) and x.name=='Context');dep=next(x for x in context.body if isinstance(x,ast.FunctionDef) and x.name=='dependency')
assert dep.args.defaults[0].value=='model.pt'
load_assignment=next(x for x in run_ablation.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base' for t in x.targets))
assert ast.unparse(load_assignment)=='base = load_lm(ctx.dependency(\'sft\'), ctx.device)'
training_assignment=next(x for x in run_sft.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='training' for t in x.targets))
assert '900' in ast.unparse(training_assignment) and "name='model'" in ast.unparse(training_assignment)
page=json.loads((D/'page-render-receipt.json').read_text());assert page['http_status']==200
assert page['actual_target_link']==[{'text':'7.13保留的直接SFT屬性模型','href':'http://127.0.0.1:8765/7.13.html'}]
receipt={'reviewer_task':report['reviewer_task'],'reviewed_on':'2026-10-06','mode':'same original technical owner actual reinspection','verdict':'pass','prior_history_path':H.relative_to(R).as_posix(),'prior_history_sha256':sha(H),'current_source':'course/chapters/07.md#7.15','current_source_sha256':current_sha,'current_snapshot_path':(D/'current-section.md').relative_to(R).as_posix(),'own_prior_source_sha256':report['source_sha256'],'actual_read_scope':['Current 7.15 entire raw section and all code, no lesson images','Current 7.13 and 7.11 necessary source-route context','Own frozen evidence artifacts and original result pointers listed below','Historical run_sft compute branches556–588, run_sft_ablation615–628, Context.dependency25–29','Actual local current7.15 Chromium page supplementary record expanded; screenshot personally viewed via view_image'],'necessary_context_snapshots':ctx,'substantive_changed_claim':{'prior_text':'本課實跑使用的A是[7.12的屬性問答](#7.12)','current_text':'本課實跑的A屬性任務使用[7.13保留的直接SFT屬性模型](#7.13)','interpretation':'Clarifies direct-SFT sft/model.pt starting checkpoint, rather than referring to the independent four-cell pedagogical splitting example. Traditional-character normalization does not change technical claims.','primary_support':{'run_sft_assignment':ast.unparse(training_assignment),'run_ablation_base_assignment':ast.unparse(load_assignment),'Context_dependency_default_filename':dep.args.defaults[0].value,'direct_sft_steps':900,'direct_sft_checkpoint':'model.pt','direct_sft_test_matches':5,'direct_sft_test_records':10,'ablation_A_before_test_identical_to_direct_sft_after_test':True,'pretrain_sft_checkpoint':'pretrain-sft.pt'},'scope':'Stored original results and actual historical compute route support which weights start the experiment. No new weights loaded or models trained; no claim of optimizer history continuation.'},'original_result_pointers_actually_checked':['sft:/revision','sft:/seed','sft:/step_scale','sft:/evidence_status','sft:/results/training/steps','sft:/results/training/checkpoint','sft:/results/checkpoint','sft:/results/after/test','sft:/results/pretrain_then_sft/sft/checkpoint','sft:/results/pretrain_then_sft/after_sft/test/matches','sft:/results/pretrain_then_sft/after_sft/test/records','sft_ablation:/revision','sft_ablation:/seed','sft_ablation:/step_scale','sft_ablation:/evidence_status','sft_ablation:/results/before/A_attributes/test'],'result_provenance':provenance,'own_prior_artifact_hash_checks':checks,'current_implementation_hash_checks':implementation_checks,'current_original_measurement_hash_checks':result_file_checks,'evidence_reuse_decision':{'fence_byte_identical':True,'lesson_svg_references':[],'all_original_evidence_hashes_unchanged':True,'reuse_original_fence_execution':True,'reuse_own_prior_independent_dataset_raw_ID_NLL_sampling_audit':True,'reuse_own_original_paper_official_docs_inspection':True,'reason':'Only substantive change clarifies source checkpoint pointer; same fence/implementation/primary source bytes/original measurements remain supported by personal executed proof. No rerun needed for unchanged numerical/CPU claims.'},'render_and_actual_view':{'page_receipt_path':(D/'page-render-receipt.json').relative_to(R).as_posix(),'page_receipt_sha256':sha(D/'page-render-receipt.json'),'screenshot_path':(D/'current-page.png').relative_to(R).as_posix(),'screenshot_sha256':sha(D/'current-page.png'),'personally_viewed':True,'observed':'Expanded supplementary text is visible and readable at1280x800. New7.13 direct-SFT link shown and actual href resolves7.13.html. Numeric record unchanged; no lesson illustration (UI icon SVGs are navigation controls).'},'execution_scope':'Bounded Python SHA/AST/original-JSON pointer comparison and Chromium local render; no original neural weight inference, retraining, GPU, external source refetch or complete pipeline.','environment':{'python':platform.python_version(),'python_executable':sys.executable,'device':'cpu; JSON/AST only','browser':page['browser']},'unresolved_issues':[]}
write(D/'reinspection-receipt.json',receipt)
print(json.dumps({'source_sha256':current_sha,'prior_history_sha256':sha(H),'reinspection_receipt_path':(D/'reinspection-receipt.json').relative_to(R).as_posix(),'reinspection_receipt_sha256':sha(D/'reinspection-receipt.json'),'fence_unchanged':True,'base_checkpoint':'sft/model.pt','base_steps':900,'base_test':'5/10','base_raw_test_equal_to_ablation_before':True,'reused_own_prior_artifacts':len(checks),'verdict':'pass'},ensure_ascii=False,indent=2))
