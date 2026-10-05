import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

BASE = Path(__file__).parent
records=[]
for name, url in {
    "torch-tolist.html": "https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.tolist.html",
    "torch-ne.html": "https://docs.pytorch.org/docs/2.11/generated/torch.ne.html",
}.items():
    request=urllib.request.Request(url, headers={"User-Agent": "Independent factual review 7.3"})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw=response.read(2_000_001)
        assert len(raw)<=2_000_000
        record={"file":name,"requested_url":url,"final_url":response.url,"status":response.status,"accessed_utc":datetime.now(UTC).isoformat(),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
    (BASE/name).write_bytes(raw);records.append(record)
(BASE/'retrieval-additional.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
