"""同環境、隔離HEAD版本比較；不更改教材或任何原判。"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
CACHE=Path('/workspace/work/reference/omni-technical-validation-v1')
BASE=CACHE/'baseline-head'
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
if BASE.exists():raise SystemExit('Refuse to overwrite baseline checkout')
env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
clone=subprocess.run(['git','clone','--shared','--no-checkout',str(ROOT),str(BASE)],env=env,capture_output=True,text=True,check=True)
subprocess.run(['git','checkout','--detach',revision],cwd=BASE,env=env,capture_output=True,text=True,check=True)
env.update(PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],PYTHONPATH=str(BASE))
results=[]
for label,argv in [('reader',['python','scripts/check_course_reviews.py']),('technical',['python','scripts/check_technical_reviews.py']),('all',['python','docs/review-tools/check_review_round.py']),('pytest',['pytest','-ra'])]:
    p=CACHE/'command-output'/f'baseline-{label}.log';start=time.monotonic()
    with p.open('wb') as out:
        try:r=subprocess.run(argv,cwd=BASE,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=110);code=r.returncode;timeout=False
        except subprocess.TimeoutExpired:code=None;timeout=True
    raw=p.read_bytes();item=dict(label=label,command=shlex.join(argv),cwd=str(BASE),returncode=code,timed_out=timeout,seconds=round(time.monotonic()-start,3),output_path=str(p),output_sha256=hashlib.sha256(raw).hexdigest(),output_excerpt=raw.decode(errors='replace')[-1600:]);results.append(item);print(json.dumps({k:item[k] for k in ['label','returncode','timed_out','seconds']}),flush=True)
result=dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),command='uv run --extra cpu python '+str(Path(__file__).relative_to(ROOT)),environment=dict(python=sys.version,device='CPU',python_executable=sys.executable,import_root=str(BASE)),baseline_revision=revision,baseline_checkout=str(BASE),baseline_scope='Exact HEAD committed version, no working-tree edits; cloned locally with Git LFS smudge skipped. Same installed Python and pytest. No global configuration or commit created.',commands=results)
Path(__file__).with_name('baseline-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
