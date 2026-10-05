import json,subprocess,sys,ast
from pathlib import Path
from types import SimpleNamespace
from scripts.course_experiments.run import experiment_spec
from scripts.course_experiments.posttraining import run_posttraining
A=Path(__file__).resolve().parents[1]
s=experiment_spec('posttraining'); fields={k:s[k] for k in ['id','module','function','assets']}
assert fields=={'id':'posttraining','module':'posttraining','function':'run_posttraining','assets':[]}
errors=[]
for ctx in [SimpleNamespace(device='cuda',step_scale=1.0),SimpleNamespace(device='cpu',step_scale=0.5)]:
 try:run_posttraining(ctx)
 except ValueError as e:errors.append(str(e))
 else:raise AssertionError('guard failed')
assert len(errors)==2
proc=subprocess.run([sys.executable,'-m','scripts.course_experiments.run','--help'],capture_output=True,text=True,timeout=15)
assert proc.returncode==0 and '--experiment' in proc.stdout and '--device {cpu,cuda}' in proc.stdout
(A/'execution/recipe-help.stdout').write_text(proc.stdout); (A/'execution/recipe-help.stderr').write_text(proc.stderr)
result={'experiment_dispatch':fields,'early_contract_rejections':errors,'help_exit_code':proc.returncode,'whole_recipe_executed':False,'bounded_substitute':'original section fence one collect/update plus feature variant; full training and checkpoint evaluation not run','read_implementation':['scripts/course_experiments/run.py:17-23,58-123,166-198','scripts/course_experiments/posttraining.py:256-389,414-451'],'dpo_branch_present':True}
(A/'execution/recipe-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False,indent=2))
