"""Run this review's original fence and bounded checks with preserved provenance."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import torch

base=Path(__file__).resolve().parents[1]
root=Path('/workspace/tiny-perceptron-vlm')
env=os.environ.copy()
offline={'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
env.update(offline)
environment={'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),'torch_git_version':str(torch.version.git_version),'cuda_build':str(torch.version.cuda),'cuda_available':str(torch.cuda.is_available()),'device':'cpu','platform':platform.platform(),'cwd':str(root),'offline_environment':offline}
assert torch.version.cuda is None and not torch.cuda.is_available()
records=[]
for name,path in [('original_fence',base/'frozen/fence-1.py'),('variants',base/'execution/check_variants.py'),('scope',base/'execution/check_scope.py')]:
    command=[sys.executable,str(path)]
    started=time.perf_counter()
    result=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,timeout=30,check=False)
    (base/'execution'/f'{name}.stdout.txt').write_text(result.stdout)
    (base/'execution'/f'{name}.stderr.txt').write_text(result.stderr)
    record={'name':name,'command_argv':command,'command':shlex.join(command),'cwd':str(root),'timeout_seconds':30,'elapsed_seconds':time.perf_counter()-started,'exit_code':result.returncode,'code_path':path.relative_to(root).as_posix(),'code_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'stdout_path':(base/'execution'/f'{name}.stdout.txt').relative_to(root).as_posix(),'stdout_sha256':hashlib.sha256(result.stdout.encode()).hexdigest(),'stderr_path':(base/'execution'/f'{name}.stderr.txt').relative_to(root).as_posix(),'stderr_sha256':hashlib.sha256(result.stderr.encode()).hexdigest()}
    records.append(record)
    print(name,'exit_code',result.returncode)
    print(result.stdout)
    if result.stderr:print(result.stderr)
(base/'execution/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
(base/'execution/runs.json').write_text(json.dumps({'environment':environment,'runs':records},indent=2)+'\n')
if any(r['exit_code'] for r in records):raise SystemExit(1)
