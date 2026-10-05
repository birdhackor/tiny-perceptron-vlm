"""Capture actual bounded verification return codes and logs."""
import hashlib,json,os,shlex,subprocess,sys,time
from datetime import datetime,UTC
from pathlib import Path
out=Path(__file__).resolve().parent;root=out.parents[3]
settings={'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','MPLBACKEND':'Agg'}
records=[]
selected=[('render','render_section.py')] if sys.argv[1:]==['render'] else [('bounded','bounded_checks.py'),('render','render_section.py')]
for label,name in selected:
 argv=[sys.executable,str(out/name)]; started=time.monotonic()
 with (out/(label+'-stdout.txt')).open('wb') as stdout,(out/(label+'-stderr.txt')).open('wb') as stderr:
  process=subprocess.run(argv,cwd=root,env={**os.environ,**settings},stdout=stdout,stderr=stderr,timeout=40,check=False)
 record={'label':label,'command_argv':argv,'command':shlex.join(argv),'cwd':str(root),'executed':True,'exit_code':process.returncode,'timeout_seconds':40,'elapsed_seconds':time.monotonic()-started,'checked_at':datetime.now(UTC).isoformat(),'environment_overrides':settings,'artifacts':[{ 'path':f'{label}-{stream}.txt','sha256':hashlib.sha256((out/f'{label}-{stream}.txt').read_bytes()).hexdigest()} for stream in ('stdout','stderr')]}
 records.append(record)
 print(json.dumps(record,ensure_ascii=False))
 if process.returncode:break
(out/('render-success-command-receipt.json' if sys.argv[1:]==['render'] else 'bounded-command-receipts.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
if any(r['exit_code'] for r in records):raise SystemExit(1)
