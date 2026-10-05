import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path
OUT=Path(__file__).parent/'sources'
records=[]
for filename,url in [
 ('pytorch-v2_8_0-sparse.py','https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/modules/sparse.py'),
 ('transformer-v7-direct.pdf','https://arxiv.org/pdf/1706.03762v7')]:
 try:
  with urllib.request.urlopen(url,timeout=20) as r:raw=r.read();final=r.url
  (OUT/filename).write_bytes(raw)
  records.append({'file':filename,'url':url,'final_url':final,'sha256':hashlib.sha256(raw).hexdigest(),
                  'bytes':len(raw),'accessed_at':datetime.now(UTC).isoformat()})
 except Exception as e:records.append({'file':filename,'url':url,'error':repr(e)})
(OUT/'extra-retrieval.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
