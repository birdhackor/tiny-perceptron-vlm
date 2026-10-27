import json,sys,subprocess,platform,hashlib
from pathlib import Path
root=Path('/workspace/tiny-perceptron-vlm'); out=root/'docs/technical-reviews/artifacts/tokenizer-20261027/readme-final'; scratch=Path('/workspace/work/tokenizer-dynamic-20261027/readme-final')
py=root/'.venv/bin/python'
commands=[['scripts/check_env.py'],['scripts/prepare_data.py','--kind','toy-text','--output',str(scratch/'generated')],['scripts/train.py','--task','text','--data',str(scratch/'generated/toy-text/train.jsonl'),'--device','cpu','--output',str(scratch/'must-not-exist.pt')],['scripts/fetch_training_assets.py','--list'],['scripts/fetch_course_models.py','--list'],['-m','ipykernel','--version'],['-m','jupyterlab','--version']]
results=[]
for args in commands:
 p=subprocess.run([str(py),*args],cwd=root,capture_output=True,text=True)
 results.append({'command':[str(py),*args],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 assert p.returncode==0,(args,p.stderr)
r=json.loads(results[2]['stdout'].splitlines()[-1]); assert r['mode']=='dry-run-no-weight-update'; h=r['history'][0]; assert h['loss']>0 and h['grad_norm']>0
assert not (scratch/'must-not-exist.pt').exists()
import torch
receipt={'created_at':__import__('datetime').datetime.now(__import__('datetime').UTC).isoformat(),'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu','platform':platform.platform()},'results':results,'checks':{'no_weight_file_created':True,'loss':h['loss'],'grad_norm':h['grad_norm'],'mode':r['mode']},'scope':'README minimum forward/backward only; no parameter update, training, data/model download, GPU, MPS or Colab execution.'}
(out/'cpu-original-01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':receipt['checks'],'commands':len(results),'environment':receipt['environment']},ensure_ascii=False))
