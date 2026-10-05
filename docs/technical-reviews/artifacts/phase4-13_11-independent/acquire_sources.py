import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/workspace/tiny-perceptron-vlm')
BASE=ROOT/'docs/technical-reviews/artifacts/phase4-13_11-independent'
DEST=BASE/'sources'
DEST.mkdir(exist_ok=True)
entries=[
 ('ppo-v2','1707.06347v2','docs/technical-reviews/artifacts/fact_finish_g_4_ppo_original.pdf'),
 ('instructgpt-v1','2203.02155v1','docs/technical-reviews/artifacts/fact_finish_b_6-instructgpt-v1.pdf'),
 ('gae-v6','1506.02438v6',None),
]
records=[]
for name,version,local in entries:
    url='https://arxiv.org/pdf/'+version
    output=DEST/(name+'.pdf')
    record={'id':name,'url':url,'version_requested':version,'accessed_at':datetime.now(timezone.utc).isoformat()}
    if local:
        shutil.copyfile(ROOT/local,output)
        record['original_locator_path']=local
        record['method']='Exact original-paper bytes located in immutable index; independently parsed below.'
    else:
        try:
            with urllib.request.urlopen(url,timeout=30) as response:
                output.write_bytes(response.read())
                record['final_url']=response.url
            record['method']='HTTPS fetch from arXiv versioned original paper.'
        except Exception as exc:
            record['error']=type(exc).__name__+': '+str(exc)
            records.append(record)
            continue
    record['sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    run=subprocess.run(['pdftotext','-layout',str(output),str(DEST/(name+'.txt'))],capture_output=True,text=True)
    record['pdftotext_exit_code']=run.returncode
    record['pdftotext_stderr']=run.stderr
    record['text_sha256']=hashlib.sha256((DEST/(name+'.txt')).read_bytes()).hexdigest()
    records.append(record)
    print(json.dumps(record,ensure_ascii=False))
(BASE/'source-acquisition.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
