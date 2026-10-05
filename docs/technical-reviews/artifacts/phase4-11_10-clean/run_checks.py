"""Save actual bounded checks' commands, stdout/stderr, status, and versions."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parents[3]
env=dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
receipts=[]
def run(name,argv,timeout=40):
    with (out/(name+'.stdout.txt')).open('wb') as stdout,(out/(name+'.stderr.txt')).open('wb') as stderr:
        try:
            result=subprocess.run(argv,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=timeout)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    r={'id':name,'argv':argv,'cwd':str(root),'exit_code':code,'timeout_seconds':timeout,'stdout':name+'.stdout.txt','stderr':name+'.stderr.txt'}
    for key in ['stdout','stderr']:
        r[key+'_sha256']=hashlib.sha256((out/r[key]).read_bytes()).hexdigest()
    receipts.append(r)
    (out/'command-receipts.json').write_text(json.dumps({'python':sys.version,'execution_environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','MPLBACKEND']},'receipts':receipts},ensure_ascii=False,indent=2)+'\n')
    print(name,code,flush=True)
    return code

assert run('cpu-check',[str(root/'.venv/bin/python'),str(out/'verify_cpu.py')],60)==0
assert run('patch-order-render',['/usr/bin/inkscape',str(out/'inputs/course/figures/rewrite-10-patch-order.svg'),'--export-type=png','--export-filename='+str(out/'render/patch-order.png')])==0
for stem in ['attention-v7','vit-v2']:
    assert run(stem+'-text',['pdftotext','-layout',str(out/'sources'/(stem+'.pdf')),str(out/'sources'/(stem+'.txt'))])==0
for name,width,height in [('desktop',1280,1600),('mobile',390,2400)]:
    run('page-'+name,['/usr/bin/chromium','--no-sandbox','--headless','--disable-gpu','--disable-dev-shm-usage','--hide-scrollbars','--no-first-run','--disable-background-networking','--disable-extensions','--disable-sync','--run-all-compositor-stages-before-draw','--virtual-time-budget=3000','--timeout=12000','--window-size='+str(width)+','+str(height),'--screenshot='+str(out/'render'/('page-'+name+'.png')),'http://127.0.0.1:8765/11.10.html'],30)
assert run('vit-formula-page',['pdftoppm','-f','3','-l','3','-scale-to','1600','-png','-singlefile',str(out/'sources/vit-v2.pdf'),str(out/'render/vit-page3')])==0
assert run('attention-formula-page',['pdftoppm','-f','4','-l','4','-scale-to','1600','-png','-singlefile',str(out/'sources/attention-v7.pdf'),str(out/'render/attention-page4')])==0
run('chromium-version',['/usr/bin/chromium','--version'])
run('inkscape-version',['/usr/bin/inkscape','--version'])
