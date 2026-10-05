import hashlib,json,urllib.request,subprocess
from pathlib import Path
from html.parser import HTMLParser

OUT=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-16_8-independent-fresh/sources')
items=[
 ('sdpa-2.14.html','https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html'),
 ('allclose-2.14.html','https://docs.pytorch.org/docs/2.14/generated/torch.allclose.html'),
 ('matmul-2.14.html','https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html'),
 ('attention-1706.03762v7.pdf','https://arxiv.org/pdf/1706.03762v7'),
 ('flashattention-2205.14135v2.pdf','https://arxiv.org/pdf/2205.14135v2'),
 ('functional-5c488690.py','https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/torch/nn/functional.py'),
]
class TextParser(HTMLParser):
 def __init__(self):super().__init__();self.data=[];self.skip=0
 def handle_starttag(self,tag,attrs):
  if tag in ('script','style'):self.skip+=1
  if tag in ('p','div','pre','li','section','h1','h2','h3'):self.data.append('\n')
 def handle_endtag(self,tag):
  if tag in ('script','style'):self.skip-=1
 def handle_data(self,data):
  if not self.skip:self.data.append(data)
records=[]
for name,url in items:
 try:
  with urllib.request.urlopen(url,timeout=30) as r:raw=r.read();status=r.status;final_url=r.url
  p=OUT/name;p.write_bytes(raw)
  item=dict(url=url,final_url=final_url,http_status=status,path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),accessed_on='2026-10-05')
  if name.endswith('.html'):
   parser=TextParser();parser.feed(raw.decode());text=''.join(parser.data)
   (OUT/(name+'.txt')).write_text(text)
  elif name.endswith('.pdf'):
   subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  records.append(item)
 except Exception as e:records.append(dict(url=url,error=type(e).__name__,message=str(e)))
 (OUT/'fetch_log.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
