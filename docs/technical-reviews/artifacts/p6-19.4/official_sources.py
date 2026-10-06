from pathlib import Path
import datetime
import hashlib
import json
import urllib.request

BASE=Path(__file__).resolve().parent
rows=[]
for name,url,version,locator in [
    ('saving_loading_models.py','https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/saving_loading_models.py','pytorch/tutorials main, snapshot retrieved 2026-10-06','lines 187-193 and 256-307'),
    ('autograd-v2.11.0.rst','https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/docs/source/notes/autograd.rst','PyTorch v2.11.0','lines 178-215: Setting requires_grad'),
]:
    data=urllib.request.urlopen(url,timeout=30).read()
    target=BASE/'external'/name;target.write_bytes(data)
    rows.append({'url':url,'version':version,'accessed_on':datetime.datetime.now(datetime.timezone.utc).date().isoformat(),'locator':locator,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'saved':str(target),'transport':'HTTPS through inherited proxy; default TLS verification retained'})
(BASE/'external/source-provenance.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
