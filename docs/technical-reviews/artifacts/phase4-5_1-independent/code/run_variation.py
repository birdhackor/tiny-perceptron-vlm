"""Execute the exact 5.1 fence with only y replaced by all ignore labels."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT=Path('/workspace/tiny-perceptron-vlm')
OUT=ROOT/'docs/technical-reviews/artifacts/phase4-5_1-independent'
raw=(OUT/'original/fence-1.py').read_bytes()
old=b'y = torch.tensor([[2, 3, 4]])'
new=b'y = torch.tensor([[-100, -100, -100]])'
assert raw.count(old)==1
variation=raw.replace(old,new)
path=OUT/'code/all-ignore-variation.py'
path.write_bytes(variation)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT))
command=[str(ROOT/'.venv/bin/python'),str(path)]
start=time.perf_counter()
with (OUT/'variation.stdout.txt').open('wb') as stdout,(OUT/'variation.stderr.txt').open('wb') as stderr:
    completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=60,check=False)
elapsed=time.perf_counter()-start
stdout=(OUT/'variation.stdout.txt').read_text()
stderr=(OUT/'variation.stderr.txt').read_text()
assert completed.returncode==1 and not stdout
assert 'ValueError: 所有 labels 都被忽略：没有可學習的答案' in stderr
assert 'before = masked_loss' in stderr
import torch
receipt=dict(command_argv=command,command=' '.join(command),cwd=str(ROOT),timeout_seconds=60,exit_code=completed.returncode,elapsed_seconds=elapsed,expected_failure=True,expected_error='ValueError: 所有 labels 都被忽略：没有可學習的答案',observed_error=stderr.splitlines()[-1],failure_boundary='first masked_loss before loop; no backward, optimizer.step, success print or after<before assertion reached',single_change=dict(original=old.decode(),replacement=new.decode()),original_fence_sha256=hashlib.sha256(raw).hexdigest(),variation_sha256=hashlib.sha256(variation).hexdigest(),python=platform.python_version(),torch=str(torch.__version__),device='cpu',cuda_build=str(torch.version.cuda),offline_environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS']})
(OUT/'variation-execution.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
