"""實跑validation頁當版命令；長掃描有界，不把逾時記成pass。"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
CACHE = Path('/workspace/work/reference/omni-technical-validation-v1/command-output')
CACHE.mkdir(parents=True, exist_ok=True)
PAGE = ROOT / 'docs/course-revision-20261029-omni/reviews/wave-01/source-freeze/sources/validation.md'
BLOCK = re.search(r'```bash\n(.*?)```', PAGE.read_text(), re.S).group(1)

def run(label, command, timeout):
    started = datetime.datetime.now(datetime.UTC).isoformat()
    before = time.monotonic()
    output_path = CACHE / (label + '.log')
    with output_path.open('wb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'',
                                   'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    raw = output_path.read_bytes()
    text = raw.decode(errors='replace')
    return dict(label=label, command=shlex.join(command), started_at=started,
                elapsed_seconds=round(time.monotonic()-before,3), timeout_seconds=timeout,
                timed_out=timed_out, returncode=code, output_path=str(output_path),
                output_bytes=len(raw), output_sha256=hashlib.sha256(raw).hexdigest(),
                output_excerpt=text[-2200:])

if __name__=='__main__':
    rows=[]
    for i,line in enumerate(BLOCK.strip().splitlines()):
        timeout = 120 if i == 6 else (35 if i == 5 else 70)
        result=run(f'page-{i+1}', ['uv','run','--extra','cpu',*shlex.split(line)], timeout)
        rows.append(result)
        print(json.dumps({k:result[k] for k in ['label','returncode','timed_out','elapsed_seconds']},ensure_ascii=False),flush=True)
    rows.append(run('focused-kernel-1.1', ['uv','run','--extra','cpu','python','scripts/check_notebooks.py','--mode','kernel','--workers','1','--lesson','1.1','--output',str(CACHE/'focused-notebook')],35))
    rows.append(run('shell-exit-status', ['bash','-c','false; echo $?; true; echo $?'],5))
    for cmd in [['git','--version'],['git','status','--short','--untracked-files=no'],['git','add','-h'],['git','commit','-h'],['git','config','-h']]:
        rows.append(run('git-'+cmd[1].lstrip('-'),cmd,10))
    value = dict(created_at=datetime.datetime.now(datetime.UTC).isoformat(),
                 command='uv run --extra cpu python '+str(Path(__file__).relative_to(ROOT)),
                 environment=dict(python=sys.version,platform=platform.platform(),device='CPU only',uv='extra cpu'),
                 source_sha256=hashlib.sha256(PAGE.read_bytes()).hexdigest(),
                 scope='Exact embedded bash command block run with bounded CPU time. Focused one-notebook rerun checks completion JSON separately. No formal training or Git commit created.',
                 results=rows)
    Path(__file__).with_name('page-command-results.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
