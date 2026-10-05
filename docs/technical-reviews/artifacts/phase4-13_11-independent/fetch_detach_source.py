import hashlib
import json
import urllib.request
from pathlib import Path

BASE=Path(__file__).resolve().parent
url='https://raw.githubusercontent.com/pytorch/pytorch/v2.9.0/torch/_tensor.py'
try:
 with urllib.request.urlopen(url,timeout=30) as response:
  raw=response.read();final_url=response.url
 (BASE/'sources/pytorch-v2.9.0-_tensor.py').write_bytes(raw)
 result={'url':url,'final_url':final_url,'version':'v2.9.0','accessed_on':'2026-10-05',
  'status':'fetched','sha256':hashlib.sha256(raw).hexdigest()}
 lines=raw.decode().splitlines()
 for i,line in enumerate(lines):
  if line.strip()=='detach = _C._add_docstr(':
   result['inspected_line_range']=[i+1,i+23]
   print('\n'.join(f'{j+1}: {lines[j]}' for j in range(i,i+23)))
except Exception as exc:
 result={'url':url,'version':'v2.9.0','accessed_on':'2026-10-05','status':'failed','error':type(exc).__name__+': '+str(exc)}
(BASE/'detach-source-acquisition.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
