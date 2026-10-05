from pathlib import Path
import urllib.request,hashlib,json,datetime
out=Path('docs/technical-reviews/artifacts/phase4-11_7-independent/sources')
urls={'experience-replay-v1.pdf':'https://arxiv.org/pdf/1811.11682v1','experience-replay-abs.html':'https://arxiv.org/abs/1811.11682v1','python-3.13.5-stdtypes.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst','python-3.13.5-builtins.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst','python-3.13.5-expressions.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/expressions.rst'}
receipts=[]
for name,url in urls.items():
 try:
  with urllib.request.urlopen(url,timeout=30) as r: data=r.read();status=r.status;headers={k:v for k,v in r.headers.items() if k.lower() in ['content-type','last-modified','etag']}
  (out/name).write_bytes(data)
  item={'url':url,'path':str(out/name),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'http_status':status,'headers':headers,'accessed_on':'2026-10-05'}
 except Exception as e:
  item={'url':url,'error_type':type(e).__name__,'error':str(e),'accessed_on':'2026-10-05'}
 receipts.append(item)
(out/'fetch-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipts,ensure_ascii=False,indent=2))
