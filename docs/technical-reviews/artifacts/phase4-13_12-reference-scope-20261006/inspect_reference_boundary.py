"""Static original-contract inspection only: no model initialization or training."""
from pathlib import Path
from datetime import datetime, UTC
import ast, hashlib, json, shutil, sys
import torch

R=Path(__file__).resolve().parents[4]
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=R/'scripts/course_experiments/posttraining.py'
tree=ast.parse(p.read_bytes())
f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_posttraining')
def assigned(target):
 return next(x for x in f.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in x.targets))
initial=assigned('policy');reference=assigned('reference');ppo=assigned('ppo')
sft=next(x for x in f.body if isinstance(x,ast.For) and isinstance(x.iter,ast.Call) and any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='sft_steps' for n in ast.walk(x.iter)))
rollout=next(x for x in f.body if isinstance(x,ast.For) and isinstance(x.iter,ast.Call) and any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value=='ppo_rollout_batches' for n in ast.walk(x.iter)))
assert initial.lineno<sft.lineno<=sft.end_lineno<reference.lineno<ppo.lineno<rollout.lineno
assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_update' for n in ast.walk(sft))
assert 'deepcopy(policy)' in ast.unparse(reference.value)
assert 'requires_grad_(False)' in ast.unparse(reference.value)
assert 'deepcopy(reference)' in ast.unparse(ppo.value)
for old_name in ['old_log_probs','old_selected']:
 old=next(x for x in ast.walk(rollout) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==old_name for t in x.targets))
 assert old.lineno>rollout.lineno
before=R/'docs/technical-reviews/artifacts/phase4-13_12-independent/frozen-input/scripts/course_experiments/posttraining.py'
assert p.read_bytes()==before.read_bytes()
result={'method_source':p.relative_to(R).as_posix(),'method_sha256':sha(p),
 'initial_policy_line':initial.lineno,'supervised_update_loop_lines':[sft.lineno,sft.end_lineno],
 'frozen_reference_line':reference.lineno,'ppo_initial_copy_line':ppo.lineno,
 'rollout_loop_line':rollout.lineno,'stage_order':'initial policy -> supervised demonstration updates -> frozen reference copy -> PPO initial copy -> rollout old snapshots',
 'reference_assignment':ast.get_source_segment(p.read_text(),reference),
 'initial_reference_is_after_sft':True,'current_original_method_matches_prior':True,
 'training_executed':False,'model_initialized':False}
out={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),
     'torch_git_version':str(torch.version.git_version),'device':'CPU/static AST inspection',
     'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),
     'executed_on':datetime.now(UTC).isoformat()}
(D/'boundary-inspection.result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(D/'boundary-inspection.environment.json').write_text(json.dumps(out,indent=2)+'\n')
target=D/'original-method/scripts/course_experiments/posttraining.py';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(target)==sha(p)
print(json.dumps(result,ensure_ascii=False,indent=2))

# Quote locators are computed from the actual current UTF-8 section snapshots.
for name,needle in [('13.12-current-context.md','reference對照整個後訓練起點'),('13.4-current-context.md','完成示範微調的起點'),('7.17-current-necessary-context.md','SFT是其中一種'),('13.15-current-context.md','監督式示範訓練')]:
 q=D/name;matches=[(i+1,line) for i,line in enumerate(q.read_text().splitlines()) if needle in line]
 assert matches;print(name,'SHA',sha(q),'section-relative quote lines',matches)
