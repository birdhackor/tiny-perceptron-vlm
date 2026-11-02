"""Independent README command replay; mutation commands run only in an external copy."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import time
from urllib.request import urlopen

ROOT = Path('/workspace/tiny-perceptron-vlm')
WORK = Path('/workspace/work/omni-technical-readme-v1')
COPY = WORK / 'sandbox'
OUT = ROOT / 'docs/technical-reviews/artifacts/omni-technical-readme-v1'
README = ROOT / 'docs/course-revision-20261029-omni/reviews/source-freeze-reader-fix04/sources/readme.md'
raw = README.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '0f500c1286a34f120413c3de0965da251343e871e8b8cc9c6770f194ae10bd94'
blocks = re.findall(r'```bash\n(.*?)```', raw.decode(), re.S)
env = dict(os.environ)
env.update(UV_PROJECT_ENVIRONMENT=str(WORK / 'venv'), UV_NO_SYNC='1',
           PYTHONPATH=str(COPY), HF_HOME=str(WORK/'hf-cache'),
           JUPYTER_CONFIG_DIR=str(WORK/'jupyter-config'), JUPYTER_RUNTIME_DIR=str(WORK/'jupyter-runtime'))
(WORK/'raw').mkdir(exist_ok=True)
(WORK/'jupyter-config').mkdir(exist_ok=True)
(WORK/'jupyter-runtime').mkdir(exist_ok=True)
(WORK/'jupyter-config/jupyter_server_config.py').write_text(
    "c.ServerApp.allow_root=True\nc.ServerApp.open_browser=False\nc.ServerApp.ip='127.0.0.1'\n"
    "c.ServerApp.port=8291\nc.IdentityProvider.token=''\n")
environment = {'python':platform.python_version(), 'torch':importlib.metadata.version('torch'),
               'device':'cpu', 'os':platform.platform(), 'uv':subprocess.check_output(['uv','--version'],text=True).strip()}
records=[]
def run(command, timeout=180, keep=True):
    i=len(records)+1
    start=datetime.now(timezone.utc).isoformat()
    wrapped='uv run --extra cpu '+command
    if command.startswith(('uv ','source ')):
        wrapped=command
    # Activate the private environment for non-Python CLI names. All command strings are literal README lines.
    shell='source .venv/bin/activate\n'+wrapped
    began=time.monotonic()
    try:
        p=subprocess.run(['bash','-c',shell],cwd=COPY,env=env,capture_output=True,text=True,timeout=timeout)
        output=p.stdout+p.stderr;code=p.returncode
    except subprocess.TimeoutExpired as e:
        output=((e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout or '')+((e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr or '')
        code='timeout'
    path=WORK/'raw'/f'{i:02d}.txt';path.write_text(output)
    r={'command':command,'actual_invocation':wrapped,'cwd':str(COPY),'started_at':start,
       'seconds':round(time.monotonic()-began,3),'exit_code':code,'raw_path':str(path),
       'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_bytes':path.stat().st_size}
    if keep and len(output)<14000:r['output']=output
    else:r['output_tail']=output[-3500:]
    records.append(r)
    (OUT/'commands.json').write_text(json.dumps({'environment':environment,'readme_sha256':hashlib.sha256(raw).hexdigest(),
      'scope':'Frozen README command text; CPU only; maintenance mutations confined to disposable copy; no training/GPU workflows.',
      'records':records},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'command':command,'exit_code':code,'seconds':r['seconds'],'tail':output[-250:]},ensure_ascii=False),flush=True)
    return p if code!='timeout' else None

# README installation after obtaining the source. Public clone is replayed separately.
run(blocks[0].splitlines()[2])
run(blocks[0].splitlines()[4])
run(blocks[0].splitlines()[5])
labcmd=blocks[0].splitlines()[6]
log=WORK/'raw/jupyter.txt'
with log.open('w') as stream:
    p=subprocess.Popen(['bash','-c','source .venv/bin/activate\n'+labcmd],cwd=COPY,env=env,stdout=stream,stderr=stream)
    status=None
    for _ in range(50):
        try:
            with urlopen('http://127.0.0.1:8291/api/status',timeout=1) as r: status=json.load(r)
            break
        except Exception: time.sleep(.2)
    p.terminate()
    try:p.wait(10)
    except subprocess.TimeoutExpired:p.kill();p.wait()
records.append({'command':labcmd,'status_api':status,'raw_path':str(log),'raw_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
                'root_specific_configuration':'localhost only, allow_root, no browser, no token; private config path; no remote Colab session'})

for command in blocks[1].strip().splitlines():run(command)
run('python scripts/fetch_training_assets.py --list')
for command in blocks[2].strip().splitlines():run(command)

# Site dependency is separate in README; add it before maintenance tools that may call the exporter.
run('uv sync --frozen --extra cpu --group notebook --group site')
for command in blocks[3].strip().splitlines():run(command,timeout=240,keep=not command.startswith(('pytest','python scripts/check_','python docs/review-tools')))

summary=json.loads((OUT/'commands.json').read_text())
summary['records']=records
summary['completed_at']=datetime.now(timezone.utc).isoformat()
(OUT/'commands.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
