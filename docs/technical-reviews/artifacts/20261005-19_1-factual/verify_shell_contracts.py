import hashlib,json,platform,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
commands=[['uv','--version'],['uv','sync','--frozen','--extra','cpu','--dry-run','--offline'],['git','ls-remote','https://github.com/birdhackor/tiny-perceptron-vlm.git','HEAD']]
records=[]
for i,argv in enumerate(commands):
 r=subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=30)
 (OUT/f'shell-{i}-stdout.txt').write_text(r.stdout);(OUT/f'shell-{i}-stderr.txt').write_text(r.stderr)
 assert r.returncode==0,(argv,r.stderr)
 records.append({'argv':argv,'exit_code':r.returncode,'stdout_file':f'shell-{i}-stdout.txt','stderr_file':f'shell-{i}-stderr.txt','stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr.encode()).hexdigest()})
assert 'Would use project environment at: .venv' in (OUT/'shell-1-stderr.txt').read_text()
(OUT/'shell-contracts.json').write_text(json.dumps({'environment':{'python':platform.python_version(),'uv':(OUT/'shell-0-stdout.txt').read_text().strip(),'device':'cpu-only dry-run; no install or model execution'},'commands':records,'scope':'Git read-only remote exists; uv frozen CPU extra resolves in dry-run only. Installation/uninstallation/download plan is not executed.'},ensure_ascii=False,indent=2)+'\n')
print('Git read-only HEAD and uv CPU frozen offline dry-run passed. No install/download/uninstall took place.')
