import os,sys,pathlib,subprocess,json,hashlib,platform,time
root=pathlib.Path('.').resolve();out=root/'docs/technical-reviews/artifacts/phase4-17_5-independent'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(root))
results=[]
for label,script in [('original','fence-1.py'),('variants','cpu-variants.py')]:
 command=[str(root/'.venv/bin/python'),str(out/script)]
 start=time.perf_counter();run=subprocess.run(command,env=env,cwd=root,capture_output=True,timeout=45)
 for stream,data in [('stdout',run.stdout),('stderr',run.stderr)]: (out/f'{label}.{stream}.txt').write_bytes(data)
 results.append({'label':label,'command_argv':command,'cwd':str(root),'timeout_seconds':45,'exit_code':run.returncode,'elapsed_seconds':time.perf_counter()-start,'script_sha256':hashlib.sha256((out/script).read_bytes()).hexdigest(),'stdout_sha256':hashlib.sha256(run.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(run.stderr).hexdigest(),'env_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','MPLBACKEND','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH']}})
 print(label,'exit',run.returncode); print(run.stdout.decode());print(run.stderr.decode())
(out/'execution.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
assert all(x['exit_code']==0 for x in results)
