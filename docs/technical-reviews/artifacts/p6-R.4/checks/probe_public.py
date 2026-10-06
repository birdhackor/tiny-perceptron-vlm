import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

root=Path('docs/technical-reviews/artifacts/p6-R.4/originals')
rev='979cdfacc588ad0536f1c64fff96f264571cf054'
def get(a):
    prefix=f'selftrained/v2/{a}-joint'
    base=f'https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/{rev}/{prefix}/'
    out={'architecture':a,'revision':rev,'prefix':prefix}
    for name in ['model-config.json','model.safetensors']:
        u=base+name
        r=requests.get(u,timeout=40) if name.endswith('json') else requests.head(u,timeout=40,allow_redirects=False)
        r.raise_for_status()
        out[name]={'url':u,'status':r.status_code,'headers':{k:v for k,v in r.headers.items()
            if k.lower() in ['x-linked-etag','x-linked-size','x-repo-commit','etag','content-length']}}
        if name.endswith('json'):
            p=root/(a+'-'+name);p.write_bytes(r.content)
            out[name]['sha256']=hashlib.sha256(r.content).hexdigest()
            out[name]['topkeys']={k:type(v).__name__ for k,v in r.json().items()}
    return out
rows=list(concurrent.futures.ThreadPoolExecutor(max_workers=2).map(get,['moe','dense']))
(root/'anonymous-public-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
