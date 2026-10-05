from pathlib import Path
import subprocess,time,json,hashlib,os,shlex
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
command=[str(ROOT/'.venv/bin/python'),str(OUT/'check_variants.py')]
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','MPLBACKEND':'Agg'}
t=time.perf_counter()
with (OUT/'variant-stdout.txt').open('wb') as stdout,(OUT/'variant-stderr.txt').open('wb') as stderr:
 result=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=45,check=False)
receipt={'command':shlex.join(command),'command_argv':command,'cwd':str(ROOT),'timeout_seconds':45,'exit_code':result.returncode,'elapsed_seconds':time.perf_counter()-t,'device':'cpu','code_sha256':hashlib.sha256((OUT/'check_variants.py').read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256((OUT/'variant-stdout.txt').read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256((OUT/'variant-stderr.txt').read_bytes()).hexdigest(),'prior_attempt':'The initial harness counted two fill=none overlay highlights as pixels and failed its len(rects)==256 assertion. Preserved original harness and traceback; corrected only the reviewer filter to count filled pixels, plus explicit independent overlay position checks.'}
(OUT/'variant-execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
print((OUT/'variant-stdout.txt').read_text())
raise SystemExit(result.returncode)
