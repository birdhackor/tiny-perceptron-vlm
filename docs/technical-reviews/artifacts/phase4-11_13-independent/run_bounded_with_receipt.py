from pathlib import Path
from datetime import UTC,datetime
import subprocess,os,json,hashlib,sys
r=Path(__file__).resolve().parent
repo=r.parents[3]
command=[str(repo/'.venv/bin/python'),str(r/'check_bounded.py')]
updates={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
start=datetime.now(UTC).isoformat()
with (r/'bounded-stdout.txt').open('wb') as stdout,(r/'bounded-stderr.txt').open('wb') as stderr:
 result=subprocess.run(command,cwd=repo,env={**os.environ,**updates},stdout=stdout,stderr=stderr,timeout=60,check=False)
receipt={'started_at':start,'finished_at':datetime.now(UTC).isoformat(),'command_argv':command,'cwd':str(repo),'explicit_environment_overrides':updates,'timeout_seconds':60,'exit_code':result.returncode,'files':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ('check_bounded.py','bounded-stdout.txt','bounded-stderr.txt','bounded-results.json')}}
(r/'bounded-execution-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
sys.exit(result.returncode)
