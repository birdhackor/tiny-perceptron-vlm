import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
from datetime import datetime,UTC
out=Path(__file__).resolve().parent/'sources';out.mkdir(exist_ok=True)
requests=[
 ('cpython-functions-v3.13.5.rst','https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst','CPython v3.13.5 tag'),
 ('cpython-stdtypes-v3.13.5.rst','https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst','CPython v3.13.5 tag'),
 ('cpython-expressions-v3.13.5.rst','https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/expressions.rst','CPython v3.13.5 tag'),
 ('mtbench-2306.05685v4.pdf','https://arxiv.org/pdf/2306.05685v4','arXiv:2306.05685v4'),
 ('tiktoken-0.12.0-README.md','https://raw.githubusercontent.com/openai/tiktoken/0.12.0/README.md','tiktoken 0.12.0 tag'),
]
def get(item):
 name,url,version=item; started=time.monotonic()
 receipt={'requested_url':url,'version_locator':version,'accessed_at':datetime.now(UTC).isoformat(),'timeout_seconds':25,'filename':name}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'independent-factual-review/8.12'}),timeout=25) as response:
   raw=response.read(4*1024*1024+1)
   if len(raw)>4*1024*1024:raise ValueError('bounded source exceeds 4 MiB')
   receipt.update(final_url=response.url,http_status=response.status,content_type=response.headers.get('Content-Type'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   (out/name).write_bytes(raw)
 except Exception as e: receipt.update(error=type(e).__name__+': '+str(e))
 receipt['elapsed_seconds']=time.monotonic()-started
 return receipt
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(get,requests))
(out/'fetch-receipts.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for result in results: print(json.dumps(result,ensure_ascii=False))
