from pathlib import Path
import hashlib
import json
import urllib.error
import urllib.request

OUT=Path(__file__).resolve().parent/'sources'
commit='5c4886908584029761b579af026dcfb627c84070'
targets=[('torch-top-level-docs.py',f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/torch/_torch_docs.py'),('torch-autograd.md',f'https://raw.githubusercontent.com/pytorch/pytorch/{commit}/docs/source/notes/autograd.md')]
records=[]
for name,url in targets:
 try:
  with urllib.request.urlopen(url,timeout=30) as response:
   raw=response.read()
   records.append({'name':name,'url':url,'final_url':response.url,'status':response.status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'accessed_on':'2026-10-05'})
  (OUT/name).write_bytes(raw)
 except urllib.error.HTTPError as error:
  records.append({'name':name,'url':url,'status':error.code,'source_not_used':True})
(OUT/'extra-recovery-receipt.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
