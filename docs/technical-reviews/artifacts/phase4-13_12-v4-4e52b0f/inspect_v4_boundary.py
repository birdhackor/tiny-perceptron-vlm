"""Original-owner V4 source and method boundary verification, no training."""
from pathlib import Path
from datetime import datetime, UTC
import ast, hashlib, importlib.util, json, shutil, sys
import torch

R=Path(__file__).resolve().parents[4];D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
P=R/'docs/technical-reviews/artifacts/phase4-13_12-independent'
prior=R/'docs/technical-reviews/artifacts/phase4-13_12-reference-scope-20261006'
assert sha(D/'prior-revise-opaque.json')=='8f44ad28bd8d64913afed5b8e932a7617af6f2a5f6cec7ff8c7ef36f10da43e3'
history=json.loads((D/'prior-history-opaque-manifest.json').read_bytes())
for n,v in history['files'].items():assert sha(R/n)==v,(n,'old history changed')
spec=importlib.util.spec_from_file_location('sf',R/'docs/review-tools/section_facts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
raw,whole,line=m.original_section(R/'course/chapters/13.md','13.12')
assert m.digest(raw)=='4e52b0f5e6afa038e0202979e85de86831e39746eca746a3fc8deb8b78fc4ffc'
assert raw==(D/'13.12-v4-current.md').read_bytes()
fence=m.fences(raw,line)[0]['raw'];assert fence==(P/'frozen-input/fence-1.py').read_bytes();(D/'current-unchanged-fence.py').write_bytes(fence)
expected='reference對照PPO開始時的策略（通常已完成示範微調），整段PPO期間可以固定。'
text=raw.decode();assert expected in text
assert '[InstructGPT的三步訓練流程與公式2](https://arxiv.org/pdf/2203.02155v1)' in text
method=R/'scripts/course_experiments/posttraining.py';assert method.read_bytes()==(P/'frozen-input/scripts/course_experiments/posttraining.py').read_bytes()
tree=ast.parse(method.read_bytes());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_posttraining')
def assigned(name):return next(x for x in f.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
reference=assigned('reference');ppo=assigned('ppo')
sft=next(x for x in f.body if isinstance(x,ast.For) and any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='sft_steps' for n in ast.walk(x.iter)))
assert sft.end_lineno<reference.lineno<ppo.lineno
assert 'deepcopy(policy)' in ast.unparse(reference.value) and 'requires_grad_(False)' in ast.unparse(reference.value)
helper=R/'tiny_perceptron/posttraining.py';assert helper.read_bytes()==(P/'frozen-input/tiny_perceptron/posttraining.py').read_bytes()
for name in ['ppo-1707.06347v2.pdf','instructgpt-2203.02155v1.pdf']:
 assert sha(P/name)==history['files'][(P/name).relative_to(R).as_posix()]
oldmeasure=json.loads((P/'bounded-cpu.result.json').read_bytes())
pointers=['ratio','exercise_ratio','wrong_denominator_ratio','ratio_times_advantage','old_log_gradient','new_log_gradient','old_distribution_batch_action','new_distribution_batch_action','per_context_probability_sums','saved_clone_retained_history','small_log_ratio','training_executed','updates_performed']
assert oldmeasure['ratio_times_advantage'][0]==1.2 and oldmeasure['training_executed'] is False
result={'current_source_sha256':m.digest(raw),'current_fence_sha256':m.digest(fence),'new_exact_reference_scope':expected,
 'new_citation_url':'https://arxiv.org/pdf/2203.02155v1','original_method_sha256':sha(method),'helper_sha256':sha(helper),
 'stage_order_lines':{'SFT':[sft.lineno,sft.end_lineno],'reference':reference.lineno,'PPO':ppo.lineno},
 'unchanged_prior_arithmetic_pointers_reopened':{'/'+n:oldmeasure[n] for n in pointers},
 'original_numerical_execution_reused':True,'new_numeric_or_training_execution':False,'old_history_files_checked':len(history['files']),
 'prior_context_13_4_unchanged':(D/'13.4-v4-current.md').read_bytes()==(prior/'13.4-current-context.md').read_bytes(),
 'prior_context_13_15_unchanged':(D/'13.15-v4-current.md').read_bytes()==(prior/'13.15-current-context.md').read_bytes(),
 'prior_context_7_17_unchanged':(D/'7.17-current-necessary-context.md').read_bytes()==(prior/'7.17-current-necessary-context.md').read_bytes(),
 'new_context_svg_sha256':sha(R/'course/figures/rewrite-13-model-roles.svg')}
environment={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'device':'CPU/static inspection','cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'executed_on':datetime.now(UTC).isoformat()}
(D/'v4-inspection.result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(D/'v4-inspection.environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2)+'\n')
shutil.copyfile(R/'course/figures/rewrite-13-model-roles.svg',D/'current-context-model-roles.svg')
for source in ['docs/review-tools/factual-reviewer-instructions.md','scripts/check_technical_reviews.py','.agents/skills/clear-tutorial/references/review-protocol.md']:
 target=D/'method-input'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/source,target)
print(json.dumps(result,ensure_ascii=False,indent=2))
