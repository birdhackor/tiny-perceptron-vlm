"""Fetch primary sources outside Git; retain only bounded excerpts and receipts."""
import hashlib
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

WORK = Path('/workspace/work/textbook-repair-20261017/training-technical/originals')
OUT = Path(__file__).parent
WORK.mkdir(parents=True, exist_ok=True)
SOURCES = {
 'venv': ('https://docs.python.org/3.12/library/venv.html', ['Scripts', 'bin/python']),
 'powershell': ('https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/new-item?view=powershell-7.5', ['-Path', '-Force']),
 'autograd': ('https://docs.pytorch.org/docs/stable/notes/autograd.html', ['accumulated', 'backward']),
 'optimizer': ('https://docs.pytorch.org/docs/stable/optim.html', ['optimizer.step()', 'zero_grad']),
 'sdpa': ('https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html', ['three supported implementations', 'FlashAttention']),
 'memory': ('https://docs.pytorch.org/docs/stable/notes/cuda.html', ['memory_allocated', 'memory_reserved']),
 'distill': ('https://arxiv.org/html/1503.02531v1', ['temperature', 'cross entropy']),
 'dpo': ('https://arxiv.org/html/2305.18290v3', ['fixed reference', 'classification', 'increase the relative']),
 'flash': ('https://arxiv.org/html/2205.14135v2', ['tiling', 'HBM']),
 'splits': ('https://scikit-learn.org/stable/modules/cross_validation.html', ['independent', 'GroupKFold']),
 'qat': ('https://arxiv.org/html/1712.05877v1', ['simulated quantization', 'training']),
 'pku': ('https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/raw/9421ffafec3fa40a1f1a7d567b4d525079477ecb/README.md', ['license:', 'safer_response_id']),
}

class Plain(HTMLParser):
 def __init__(self):
  super().__init__(); self.items=[]; self.skip=0
 def handle_starttag(self,t,a):
  if t in ('script','style'): self.skip+=1
 def handle_endtag(self,t):
  if t in ('script','style'): self.skip=max(0,self.skip-1)
 def handle_data(self,d):
  if not self.skip: self.items.append(d)

receipts=[]
for name,(url,terms) in SOURCES.items():
 try:
  request=urllib.request.Request(url, headers={'User-Agent':'Technical review source verification'})
  with urllib.request.urlopen(request, timeout=40) as r: data=r.read(); resolved=r.url
  redirects=[]
  for _ in range(3):
   redirect=re.search(br'<link rel="canonical" href="([^"]+)"', data)
   if len(data)>2000 or not redirect: break
   from urllib.parse import urljoin
   target=urljoin(resolved,redirect.group(1).decode())
   if target==resolved: break
   redirects.append({'url':resolved,'sha256':hashlib.sha256(data).hexdigest()})
   with urllib.request.urlopen(urllib.request.Request(target,headers={'User-Agent':'Technical review source verification'}),timeout=40) as r: data=r.read(); resolved=r.url
  path=WORK/f'{name}.html'; path.write_bytes(data)
  p=Plain(); p.feed(data.decode('utf-8','replace'))
  plain=re.sub(r'\s+',' ',' '.join(p.items)); (WORK/f'{name}.txt').write_text(plain)
  excerpts=[]
  for term in terms:
   hits=[m.start() for m in re.finditer(re.escape(term),plain,re.I)]
   excerpts.append({'term':term,'hits':len(hits),'excerpts':[plain[max(0,i-100):i+300] for i in hits[:1]]})
  receipts.append({'id':name,'url':url,'resolved_url':resolved,'meta_redirect_receipts':redirects,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'full_text_external':str(path),'excerpts':excerpts})
 except Exception as e:
  receipts.append({'id':name,'url':url,'error':str(e)})
(OUT/'primary-source-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k not in ('excerpts',)} for r in receipts],indent=2))
