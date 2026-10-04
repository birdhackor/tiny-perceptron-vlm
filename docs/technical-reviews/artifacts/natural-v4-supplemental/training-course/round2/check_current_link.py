from pathlib import Path
import urllib.request,json,hashlib,sys
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[6];D=Path(__file__).parent;url='https://docs.pytorch.org/docs/2.14/generated/torch.cuda.memory.max_memory_allocated.html';attempts=[{'attempt':1,'request':'plain urllib.urlopen without explicit User-Agent','status':403,'error':'HTTP Error403 Forbidden; Python traceback displayed in actual tool call; no HTML saved'}]
try:
 req=urllib.request.Request(url,headers={'User-Agent':'Factual-review/1.0'})
 with urllib.request.urlopen(req,timeout=30) as r:b=r.read();status=r.status;resolved=r.url
 P=ROOT/'outputs/natural-v4/factual-research/training-course/round2-memory-canonical.html';P.write_bytes(b);soup=BeautifulSoup(b,'html.parser');main=soup.select_one('article') or soup.select_one('main');P.with_suffix('.txt').write_text(main.get_text('\n',strip=True))
 attempts.append({'attempt':2,'request':'User-AgentFactual-review/1.0 (same as original round1 retrieval)','status':status,'url':url,'resolved_url':resolved,'raw_sha256':hashlib.sha256(b).hexdigest(),'ignored_path':str(P.relative_to(ROOT))})
except Exception as e:attempts.append({'attempt':2,'error':repr(e),'status':getattr(e,'code',None),'url':url})
(D/'current-link-retrieval.json').write_text(json.dumps({'version':'PyTorch2.14','accessed_on':'2026-10-04','attempts':attempts},indent=2)+'\n');print(json.dumps(attempts,indent=2))
