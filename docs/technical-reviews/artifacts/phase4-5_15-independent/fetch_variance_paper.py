"""Recover original paper from its author's arXiv upload after PMLR 404."""
from pathlib import Path
import hashlib,json,subprocess,urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE=Path(__file__).resolve().parent
sources=BASE/'sources'
items={
 'variance-2103.03098v1.pdf':'https://arxiv.org/pdf/2103.03098v1',
 'variance-2103.03098v1.html':'https://arxiv.org/abs/2103.03098v1',
 'pytorch-2.8-generator.html':'https://docs.pytorch.org/docs/2.8/generated/torch.Generator.html',
 'pytorch-2.8-randperm.html':'https://docs.pytorch.org/docs/2.8/generated/torch.randperm.html',
}
def fetch(item):
 name,url=item
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'technical-review/5.15'}),timeout=25) as r:
   raw=r.read(); resolved=r.url
  (sources/name).write_bytes(raw)
  return {'name':name,'url':url,'resolved_url':resolved,'accessed_on':'2026-10-05','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'saved':True}
 except Exception as e:return {'name':name,'url':url,'saved':False,'error':str(e)}
with ThreadPoolExecutor(max_workers=4) as pool: receipts=list(pool.map(fetch,items.items()))
(BASE/'extra-acquisition.json').write_text(json.dumps(receipts,indent=2)+'\n')
from bs4 import BeautifulSoup
for receipt in receipts:
 if receipt['saved'] and receipt['name'].endswith('.html'):
  name=receipt['name'];soup=BeautifulSoup((sources/name).read_bytes(),'html.parser')
  main=soup.find('article') or soup.find('main') or soup
  for e in main.find_all(['script','style']):e.decompose()
  (sources/name.replace('.html','.txt')).write_text(main.get_text('\n',strip=True)+'\n')
if (sources/'variance-2103.03098v1.pdf').is_file():
 r=subprocess.run(['pdftotext','-layout',str(sources/'variance-2103.03098v1.pdf'),str(sources/'variance-2103.03098v1.txt')],capture_output=True,text=True,timeout=20)
 (BASE/'pdf-conversion.json').write_text(json.dumps({'command':r.args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
print(json.dumps(receipts,indent=2))
