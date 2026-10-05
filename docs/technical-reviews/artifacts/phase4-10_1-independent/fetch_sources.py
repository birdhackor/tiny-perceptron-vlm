"""Fetch exact official raw sources with TLS verification; record actual receipts."""
from pathlib import Path
import urllib.request, hashlib, json, datetime, concurrent.futures
OUT=Path(__file__).resolve().parent/'sources';OUT.mkdir(exist_ok=True)
GIT='5c4886908584029761b579af026dcfb627c84070'
TARGETS=[
 ('torch-tensor-docs.py',f'https://raw.githubusercontent.com/pytorch/pytorch/{GIT}/torch/_tensor_docs.py'),
 ('torch-docs.py',f'https://raw.githubusercontent.com/pytorch/pytorch/{GIT}/torch/_torch_docs.py'),
 ('torch-tensor-indexing.h',f'https://raw.githubusercontent.com/pytorch/pytorch/{GIT}/aten/src/ATen/TensorIndexing.h'),
 ('pillow-concepts.rst','https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/docs/handbook/concepts.rst'),
 ('torchvision-functional.py','https://raw.githubusercontent.com/pytorch/vision/v0.24.0/torchvision/transforms/functional.py'),
]
def fetch(item):
 name,url=item
 rec={'file':name,'url':url,'accessed_at':datetime.datetime.now(datetime.UTC).isoformat(),'tls_verification':'urllib default verified HTTPS context'}
 try:
  with urllib.request.urlopen(url,timeout=20) as r:
   data=r.read();rec.update(status=r.status,final_url=r.url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest());(OUT/name).write_bytes(data)
 except Exception as e:rec.update(error=type(e).__name__+': '+str(e))
 return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:recs=list(pool.map(fetch,TARGETS))
(OUT/'retrieval-receipts.json').write_text(json.dumps(recs,indent=2)+'\n')
print(json.dumps(recs,indent=2))
