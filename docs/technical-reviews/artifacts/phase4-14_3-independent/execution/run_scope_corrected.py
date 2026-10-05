from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys,time
base=Path(__file__).resolve().parents[1]
root=Path('/workspace/tiny-perceptron-vlm')
p=base/'execution/check_scope.py'
env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
command=[sys.executable,str(p)];started=time.perf_counter()
r=subprocess.run(command,cwd=root,env=env,text=True,capture_output=True,timeout=30,check=False)
for label,value in [('stdout',r.stdout),('stderr',r.stderr)]:
 (base/'execution'/f'corrected-scope.{label}.txt').write_text(value)
record={'command_argv':command,'command':shlex.join(command),'exit_code':r.returncode,'cwd':str(root),'timeout_seconds':30,'elapsed_seconds':time.perf_counter()-started,'code_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr.encode()).hexdigest(),'environment_file':'initial-environment.json','prior_failure':'initial-scope.stderr.txt; initial current architecture full-file SHA check failed; recorded Git revision source was retrieved and matched before correction'}
(base/'execution/corrected-scope-run.json').write_text(json.dumps(record,indent=2)+'\n')
print(r.stdout)
if r.stderr:print(r.stderr)
raise SystemExit(r.returncode)
