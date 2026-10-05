# Exact Python source used in the initial heredoc retrieval command; retained as a receipt, not rerun.
from pathlib import Path
import shutil,urllib.request,json,hashlib,datetime
base=Path('docs/technical-reviews/artifacts/phase4-9_3-clean');base.mkdir(parents=True,exist_ok=False)
for name in ['section.md','fence-1.py','bootstrap.py','environment.json','extraction.json','execution.json','stdout.txt','stderr.txt']:
 shutil.copy2(Path('outputs/reviewer-tools/phase4-9_3-clean')/name,base/name)
source=base/'sources';source.mkdir()
receipts=[]
requests=[('sycophancy-v4.pdf','https://arxiv.org/pdf/2310.13548v4'),('dpo-v3.pdf','https://arxiv.org/pdf/2305.18290v3'),('cpython-stdtypes-v3.13.5.rst','https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst')]
for filename,url in requests:
 receipt={'url':url,'accessed_on':'2026-10-05'}
 try:
  with urllib.request.urlopen(url,timeout=30) as r:
   raw=r.read();receipt.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'))
  (source/filename).write_bytes(raw);receipt.update(path=str(source/filename),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 except Exception as e:receipt['error']=repr(e)
 receipts.append(receipt)
(base/'source-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipts,ensure_ascii=False,indent=2))
