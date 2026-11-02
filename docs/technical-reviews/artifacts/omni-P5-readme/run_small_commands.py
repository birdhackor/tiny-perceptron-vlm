"""Reviewer-owned small CPU command receipts; never loads a final or teacher model."""
import datetime, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
ROOT=Path('/workspace/work/omni-book-integrated-v1')
OUT=ROOT/'docs/technical-reviews/artifacts/omni-P5-readme'
SCRATCH=Path('/workspace/work/omni-P5-readme-independent-raw/commands')
SCRATCH.mkdir(parents=True,exist_ok=True)
(SCRATCH/'scripts').symlink_to(ROOT/'scripts') if not (SCRATCH/'scripts').exists() else None
PYTHON='/workspace/tiny-perceptron-vlm/.venv/bin/python'
env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),UV_PROJECT_ENVIRONMENT='/workspace/tiny-perceptron-vlm/.venv',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
commands=[
 ('environment',[PYTHON,'scripts/check_env.py'],ROOT),
 ('toy-data',[PYTHON,'scripts/prepare_data.py','--kind','toy-text'],SCRATCH),
 ('toy-dry-run',[PYTHON,'scripts/train.py','--task','text','--data','data/generated/toy-text/train.jsonl'],SCRATCH),
 ('asset-list',[PYTHON,'scripts/fetch_training_assets.py','--list'],ROOT),
 ('model-list',[PYTHON,'scripts/fetch_course_models.py','--list'],ROOT),
 ('notebook-source-check',[PYTHON,'scripts/build_course.py','--check'],ROOT),
 ('kernel-toy-3.6',[PYTHON,'scripts/check_notebooks.py','--mode','kernel','--lesson','3.6','--output',str(SCRATCH/'kernel')],ROOT),
 ('uv-cpu-resolution',['uv','sync','--frozen','--extra','cpu','--group','notebook','--dry-run','--offline'],ROOT),
 ('ipykernel-prefix',[PYTHON,'-m','ipykernel','install','--prefix',str(SCRATCH/'kernel-prefix'),'--name','readme-independent','--display-name','README independent check'],ROOT),
 ('jupyterlab-cli',[PYTHON,'-m','jupyterlab','--version'],ROOT),
 ('pytest-cli',[PYTHON,'-m','pytest','--version'],ROOT),
 ('ruff-cli',[str(Path(PYTHON).with_name('ruff')),'--version'],ROOT),
 ('uv-cli',['uv','--version'],ROOT),
 ('git-cli',['git','--version'],ROOT),
]
rows=[]
for name,cmd,cwd in commands:
 began=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  p=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True,timeout=45)
  row={'id':name,'command':cmd,'cwd':str(cwd),'started_at':began,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired as e:row={'id':name,'command':cmd,'cwd':str(cwd),'started_at':began,'timeout_seconds':45,'stdout':str(e.stdout),'stderr':str(e.stderr)}
 rows.append(row)
 print(json.dumps({'command_id':name,'returncode':row.get('returncode'),'stdout_tail':row.get('stdout','')[-550:],'stderr_tail':row.get('stderr','')[-550:]},ensure_ascii=False),flush=True)
result={'reviewer_task':'/root/omni_p5_technical_readme','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{'runner_python':platform.python_version(),'command_python':PYTHON,'device':'CPU','platform':platform.platform(),'torch':'2.14.1+cpu','thread_limits':'2'},'scope':'Artificial toy forward/backward only. Zero --train, optimizer update, final/teacher neural model load/forward, mature model downloads, Modal, workflows or publication. ipykernel uses an isolated prefix instead of mutating the shared sys-prefix. uv is an offline dry-run resolution, not a new installation.','commands':rows,'toy_checkpoint_files':[str(p) for p in SCRATCH.rglob('*.pt')],'kernel_spec':json.loads((SCRATCH/'kernel-prefix/share/jupyter/kernels/readme-independent/kernel.json').read_text())}
with (OUT/'small-command-receipts-v1.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
