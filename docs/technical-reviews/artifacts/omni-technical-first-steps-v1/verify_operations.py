"""Bounded operational checks; no clone, sync mutation, training or cloud runtime."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path('/workspace/work/omni-technical-first-steps')
records = []
def run(args, cwd=ROOT):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=25)
    row = {'command': args, 'cwd': str(cwd), 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    records.append(row)
    return p
run(['uv', '--version'])
run(['git', '--version'])
run(['git', 'lfs', 'version'])
run(['uv', 'sync', '--frozen', '--extra', 'cpu', '--group', 'notebook', '--dry-run'])
run([str(ROOT/'.venv/bin/python'), 'scripts/check_env.py'])
run([sys.executable, '-m', 'ipykernel', 'install', '--prefix', str(CACHE/'isolated-kernel'),
     '--name', 'tiny-perceptron', '--display-name', 'Tiny Perceptron'])
run([sys.executable, '-m', 'jupyterlab', '--version'])
run([sys.executable, 'scripts/train_simple.py', '--model', 'bigram', '--seed', '42', '--device', 'cpu'])
run(['git', 'check-ignore', '--no-index', 'data/x', 'checkpoints/x', 'outputs/x', '.venv/x'])

args=[sys.executable, '-m', 'jupyterlab', 'notebooks', '--no-browser', '--ServerApp.allow_root=True',
      '--ServerApp.ip=127.0.0.1', '--ServerApp.port=0', '--IdentityProvider.token=',
      '--ServerApp.open_browser=False']
env={**os.environ, 'JUPYTER_RUNTIME_DIR': str(CACHE/'jupyter-runtime'),
     'JUPYTER_CONFIG_DIR': str(CACHE/'jupyter-config'), 'IPYTHONDIR': str(CACHE/'ipython')}
p=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
try:
    stdout,stderr=p.communicate(timeout=4)
    stopped_by_reviewer=False
except subprocess.TimeoutExpired:
    p.terminate()
    stdout,stderr=p.communicate(timeout=5)
    stopped_by_reviewer=True
records.append({'command':args,'cwd':str(ROOT),'exit_code':p.returncode,'stdout':stdout,'stderr':stderr,
                'terminated_after_4_seconds_by_reviewer':stopped_by_reviewer,
                'started_server': 'Jupyter Server' in stderr and 'is running at:' in stderr})

sys.path.insert(0,str(ROOT))
from scripts.export_course import section_footer
index=json.loads((ROOT/'course/lesson-index.json').read_bytes())
item=next(x for x in index if x['id']=='1.1')
notebook=json.loads((ROOT/item['notebook']).read_bytes())
footer=section_footer(item,notebook,'frozen-review-placeholder',True)
site_path=ROOT/'outputs/omni-review-site-current/build-info.json'
site_info=json.loads(site_path.read_bytes()) if site_path.exists() else {}
result={'created_at':datetime.now(timezone.utc).isoformat(),
        'command':'uv run --extra cpu python docs/technical-reviews/artifacts/omni-technical-first-steps-v1/verify_operations.py',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','platform':platform.platform()},
        'commands':records, 'footer_generated':footer,
        'notebook_first_code_cell':next(c for c in notebook['cells'] if c['cell_type']=='code'),
        'index_notebook_count':len(index),'missing_index_notebooks':[x['notebook'] for x in index if not (ROOT/x['notebook']).exists()],
        'site_build_info_path':str(site_path),'site_build_info_sha256':hashlib.sha256(site_path.read_bytes()).hexdigest() if site_path.exists() else None,
        'site_build_info':site_info,
        'limits':'Existing Linux environment only; sync is dry-run; kernel installation uses an isolated prefix rather than modifying shared sys-prefix. Server terminated after bounded startup. No Windows, macOS, Google account/Colab, fresh full clone or entire book execution performed.'}
target=OUT/'operations-execution-v1.json'
with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                  'commands':[{k:v for k,v in r.items() if k in ['command','exit_code','started_server']} for r in records],
                  'missing_index_notebooks':result['missing_index_notebooks'],'limits':result['limits']},ensure_ascii=False))
