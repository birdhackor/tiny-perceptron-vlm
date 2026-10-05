import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, UTC
from html.parser import HTMLParser
from pathlib import Path

OUT = Path(__file__).parent / 'sources'
URLS = {
 'python-unicode.html':'https://docs.python.org/3.13/howto/unicode.html',
 'python-stdtypes.html':'https://docs.python.org/3.13/library/stdtypes.html',
 'python-codecs.html':'https://docs.python.org/3.13/library/codecs.html',
 'unicode-uax29.html':'https://www.unicode.org/reports/tr29/tr29-47.html',
 'rfc3629.txt':'https://www.rfc-editor.org/rfc/rfc3629.txt',
 'transformer-abs.html':'https://arxiv.org/abs/1706.03762v7',
 'gpt2-report.pdf':'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf',
 'torch-embedding.html':'https://docs.pytorch.org/docs/2.8/generated/torch.nn.Embedding.html',
 'hf-tokenizer-summary.html':'https://huggingface.co/docs/transformers/v4.57.1/tokenizer_summary',
}
class PlainText(HTMLParser):
 def __init__(self):
  super().__init__(); self.parts=[]; self.hidden=0
 def handle_starttag(self, tag, attrs):
  if tag in ('script','style'): self.hidden+=1
  if tag in ('p','li','h1','h2','h3','h4','div','tr','br','dt','dd','pre'):self.parts.append('\n')
 def handle_endtag(self,tag):
  if tag in ('script','style'):self.hidden-=1
 def handle_data(self,data):
  if not self.hidden:self.parts.append(data)
def fetch(item):
 name,url=item
 try:
  with urllib.request.urlopen(url,timeout=25) as response:
   raw=response.read(); final=response.url; status=response.status
  (OUT/name).write_bytes(raw)
  if name.endswith('.html'):
   parser=PlainText(); parser.feed(raw.decode('utf-8')); (OUT/name.replace('.html','.txt')).write_text(''.join(parser.parts))
  return {'file':name,'url':url,'final_url':final,'status':status,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'accessed_at':datetime.now(UTC).isoformat()}
 except Exception as e:return {'file':name,'url':url,'error':repr(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 results=list(pool.map(fetch,URLS.items()))
(OUT/'retrieval.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
