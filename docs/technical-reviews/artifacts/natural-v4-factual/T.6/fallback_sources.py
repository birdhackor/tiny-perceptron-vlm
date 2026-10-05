import hashlib,json,urllib.request
from pathlib import Path
root=Path('outputs/natural-v4/factual-research/T.6')
items=[]
for name,url in {
 'torch-normalization-source':'https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/normalization.py',
 'torch-functional-source':'https://raw.githubusercontent.com/pytorch/pytorch/main/torch/functional.py',
 'soundfile-source':'https://raw.githubusercontent.com/bastibe/python-soundfile/master/soundfile.py',
}.items():
 try:
  with urllib.request.urlopen(url,timeout=30) as response: raw=response.read()
  (root/(name+'.original')).write_bytes(raw)
  items.append(dict(id=name,url=url,accessed_on='2026-10-04',retrieval_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 except Exception as error:items.append(dict(id=name,url=url,error=str(error)))
print(json.dumps(items,indent=2))
