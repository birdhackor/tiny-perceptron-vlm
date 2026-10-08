import urllib.request,pathlib,hashlib,json
b=pathlib.Path('/workspace/work/tutorial-audit-20261008/technical-modalities-work/originals');b.mkdir(exist_ok=True)
urls={
'ppo.pdf':'https://arxiv.org/pdf/1707.06347',
'dpo.pdf':'https://arxiv.org/pdf/2305.18290v3',
'minds-paper.pdf':'https://arxiv.org/pdf/2104.08524',
'minds-card.md':'https://huggingface.co/datasets/PolyAI/minds14/raw/40ce77cb32a384e4d50a568e1ec39ac804019d33/README.md',
'fashion-readme.md':'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/README.md',
'fashion-license.txt':'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE'}
from concurrent.futures import ThreadPoolExecutor

def get(item):
 n,u=item
 try:
  data=urllib.request.urlopen(u,timeout=35).read();(b/n).write_bytes(data);return {'file':n,'url':u,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'downloaded'}
 except Exception as e:return {'file':n,'url':u,'status':'failed','error':str(e)}
with ThreadPoolExecutor(max_workers=6) as pool:r=list(pool.map(get,urls.items()))
(b/'manifest.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
