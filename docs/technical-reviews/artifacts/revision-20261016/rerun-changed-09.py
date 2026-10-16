import json,sys,time,hashlib
from pathlib import Path
from datetime import datetime,UTC
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('/workspace/tiny-perceptron-vlm');sys.path.insert(0,str(R))
from scripts.check_notebooks import run_kernel
B=R/'docs/course-revision-20261016';a=json.loads((B/'freeze-08.json').read_bytes());b=json.loads((B/'freeze-09.json').read_bytes());old={x['page_id']:x for x in a['pages']}
pages=[x for x in b['pages'] if x.get('notebook') and x['source_sha256']!=old[x['page_id']]['source_sha256']]
out=R/'outputs/revision-20261016-kernels-03';out.mkdir(exist_ok=False);rows=[];start=datetime.now(UTC).isoformat();t=time.perf_counter()
with ThreadPoolExecutor(max_workers=3) as e:
 fs={e.submit(run_kernel,R/p['notebook'],out,180):p for p in pages}
 for f in as_completed(fs):
  p=fs[f]
  try:target=Path(f.result());rows.append({'id':p['page_id'],'status':'passed','input_sha256':hashlib.sha256((R/p['notebook']).read_bytes()).hexdigest(),'output_path':str(target),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
  except Exception as ex:rows.append({'id':p['page_id'],'status':'failed','error':str(ex)})
  print(rows[-1]['id'],rows[-1]['status'],flush=True)
r={'author_task':'/root','command':'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python /workspace/work/curriculum-20261016/rerun_changed.py','started_at':start,'completed_at':datetime.now(UTC).isoformat(),'seconds':time.perf_counter()-t,'mode':'new independent kernel per notebook','workers':3,'total':len(rows),'passed':sum(x['status']=='passed' for x in rows),'rows':rows,'scope':'Changed freeze09 student notebooks, not formal B/C training.'}
(out/'validation-kernel.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
if r['passed']!=r['total']:raise SystemExit(1)
