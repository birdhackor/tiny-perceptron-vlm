import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-5_12-independent'
script=OUT/'code/inspect_cpu.py'
cmd=[str(ROOT/'.venv/bin/python'),str(script)]
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
start=time.perf_counter()
with (OUT/'inspection.stdout.txt').open('wb') as stdout,(OUT/'inspection.stderr.txt').open('wb') as stderr:
 result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=45,check=False)
receipt={'command_argv':cmd,'command':shlex.join(cmd),'cwd':str(ROOT),'timeout_seconds':45,
 'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-start,
 'executed_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
 'environment_file':'cpu-environment.json','stdout':'inspection.stdout.txt','stderr':'inspection.stderr.txt'}
(OUT/'inspection-execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
raise SystemExit(result.returncode)
