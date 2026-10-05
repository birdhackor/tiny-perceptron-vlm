import json,os,sys,subprocess,datetime,hashlib,shutil
from pathlib import Path
A=Path(__file__).resolve().parent;ROOT=A.parents[3]
commands=[('extract-original',[str(ROOT/'.venv/bin/python'),'docs/review-tools/section_facts.py','course/chapters/07.md#7.19','--output','/tmp/phase4-7_19-receipted']),('cpu-probe',[str(ROOT/'.venv/bin/python'),str(A/'probe.py')]),('render-640',['inkscape','course/figures/new-7.19-sequence-budget.svg','--export-type=png','--export-filename='+str(A/'figure-640.png'),'--export-width=640']),('render-390',['inkscape','course/figures/new-7.19-sequence-budget.svg','--export-type=png','--export-filename='+str(A/'figure-390.png'),'--export-width=390']),('inkscape-version',['inkscape','--version'])]
overrides={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':''}
env=os.environ.copy();env.update(overrides)
receipts=[]
for name,argv in commands:
 start=datetime.datetime.now(datetime.UTC).isoformat()
 r=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,timeout=30,check=False)
 (A/(name+'.stdout.txt')).write_bytes(r.stdout);(A/(name+'.stderr.txt')).write_bytes(r.stderr)
 record={'id':name,'command_argv':argv,'cwd':str(ROOT),'environment_overrides':overrides,'started_at':start,'finished_at':datetime.datetime.now(datetime.UTC).isoformat(),'exit_code':r.returncode,'stdout':name+'.stdout.txt','stderr':name+'.stderr.txt','stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()};receipts.append(record)
 if r.returncode:raise RuntimeError(name+' failed; inspect stored stdout/stderr')
 if name=='extract-original':
  for path in Path('/tmp/phase4-7_19-receipted').rglob('*'):
   if path.is_file():
    target=A/'original'/path.relative_to('/tmp/phase4-7_19-receipted');assert target.read_bytes()==path.read_bytes()
(A/'command-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'commands':[(r['id'],r['exit_code']) for r in receipts]},ensure_ascii=False,indent=2))
