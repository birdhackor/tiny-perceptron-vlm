import urllib.request,hashlib,json
from pathlib import Path
from html.parser import HTMLParser
class Text(HTMLParser):
 def __init__(self): super().__init__(); self.out=[]; self.skip=0
 def handle_starttag(self,t,a):
  if t in ('script','style'): self.skip+=1
  if t in ('p','div','li','h1','h2','h3','pre','section'): self.out.append('\n')
 def handle_endtag(self,t):
  if t in ('script','style'): self.skip=max(0,self.skip-1)
 def handle_data(self,d):
  if not self.skip: self.out.append(d)
root=Path.cwd(); out=root/'outputs/natural-v4/factual-research/site-validation-publishing'; receipt=[]
items={
 'ml-splits':'https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets',
 'pages':'https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages',
 'uv-sync':'https://docs.astral.sh/uv/concepts/projects/sync/',
 'git-clone':'https://git-scm.com/docs/git-clone',
 'mdn-origin':'https://developer.mozilla.org/en-US/docs/Web/Security/Same-origin_policy',
 'zensical-cli':'https://zensical.org/docs/usage/build/'
}
for ident,url in items.items():
 try:
  response=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'FactualReview/1.0'}),timeout=25); raw=response.read(); p=out/(ident+'.html'); p.write_bytes(raw)
  parser=Text(); parser.feed(raw.decode('utf-8')); txt=''.join(parser.out); (out/(ident+'.txt')).write_text(txt)
  receipt.append({'id':ident,'url':url,'final_url':response.url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'status':response.status,'text_path':str((out/(ident+'.txt')).relative_to(root))})
 except Exception as exc: receipt.append({'id':ident,'url':url,'error':repr(exc)})
a=root/'docs/technical-reviews/artifacts/natural-v4-supplemental/site-validation-publishing'; (a/'authority-retrieval.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
