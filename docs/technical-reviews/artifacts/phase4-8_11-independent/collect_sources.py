import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sources'
OUT.mkdir(exist_ok=True)
TARGETS={
 'mtbench-arena-v4.pdf':'https://arxiv.org/pdf/2306.05685v4',
 'cpython-v3.13.5-stdtypes.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst',
 'cpython-v3.13.5-simple_stmts.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/simple_stmts.rst',
 'cpython-v3.13.5-functions.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst',
}
def get(item):
 name,url=item
 try:
  with urllib.request.urlopen(url,timeout=30) as response:
   data=response.read(12_000_001)
   if len(data)>12_000_000: raise ValueError('source-size-bound')
   receipt={'file':name,'url':url,'final_url':response.url,'http_status':response.status,'content_type':response.headers.get('Content-Type'),'accessed_at_utc':datetime.now(UTC).isoformat(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  (OUT/name).write_bytes(data)
  return receipt
 except Exception as e:
  return {'file':name,'url':url,'error_type':type(e).__name__,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 receipts=list(pool.map(get,TARGETS.items()))
(ROOT/'source-retrieval-receipt.json').write_text(json.dumps({'command':'python docs/technical-reviews/artifacts/phase4-8_11-independent/collect_sources.py','tls_policy':'default verified HTTPS; no certificate bypass','sources':receipts},ensure_ascii=False,indent=2)+'\n')
for r in receipts: print(json.dumps(r,ensure_ascii=False))
raise SystemExit(1 if any('error' in r for r in receipts) else 0)
