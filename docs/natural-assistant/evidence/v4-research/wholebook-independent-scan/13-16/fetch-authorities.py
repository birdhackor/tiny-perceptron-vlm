from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,subprocess,requests
from bs4 import BeautifulSoup
out=Path('/workspace/tiny-perceptron-vlm/outputs/natural-v4/wholebook-review/13-16/sources'); out.mkdir(exist_ok=True)
items=[('dpo','https://arxiv.org/pdf/2305.18290v3'),('ppo','https://arxiv.org/pdf/1707.06347'),('switch','https://arxiv.org/pdf/2101.03961v3'),('roformer','https://arxiv.org/pdf/2104.09864v5'),('gqa','https://arxiv.org/pdf/2305.13245v3'),('flash','https://arxiv.org/pdf/2205.14135v2'),('pytorch-sdpa','https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html'),('pytorch-amp','https://docs.pytorch.org/docs/2.14/amp.html')]
def fetch(item):
 name,url=item
 r=requests.get(url,timeout=40); r.raise_for_status(); kind='pdf' if r.content.startswith(b'%PDF') else 'html'; p=out/f'{name}.{kind}';p.write_bytes(r.content)
 if kind=='pdf':
  result=subprocess.run(['pdftotext','-layout',str(p),str(out/f'{name}.txt')],capture_output=True,text=True); assert result.returncode==0,result.stderr
 else: (out/f'{name}.txt').write_text(BeautifulSoup(r.text,'html.parser').get_text(' ',strip=True))
 t=out/f'{name}.txt'
 return {'name':name,'requested_url':url,'final_url':r.url,'status':r.status_code,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_path':str(p),'source_sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'text_path':str(t),'text_sha256':hashlib.sha256(t.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(fetch,items))
(out/'receipts.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for r in results: print(r['name'],r['status'],r['bytes'],r['source_sha256'])
