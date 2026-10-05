import hashlib
import json
import urllib.request
from pathlib import Path
from html.parser import HTMLParser

OUT=Path('/workspace/tiny-perceptron-vlm/docs/technical-reviews/artifacts/phase4-5_12-independent/sources')
sources={
 'python-3.13.5-stdtypes.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst',
 'python-3.13.5-expressions.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/reference/expressions.rst',
 'python-3.13.5-functions.rst':'https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/functions.rst',
 'stanford-shingling.html':'https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html',
 'stanford-kgram.html':'https://nlp.stanford.edu/IR-book/html/htmledition/k-gram-indexes-for-wildcard-queries-1.html',
 'sklearn-GroupShuffleSplit.html':'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html',
 'lee-acl-2022.245-record.html':'https://aclanthology.org/2022.acl-long.577/',
 'lee-2022.acl-long.577.pdf':'https://aclanthology.org/2022.acl-long.577.pdf',
}
class Text(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.ignore=0
 def handle_starttag(self,tag,attrs):
  if tag in {'script','style'}:self.ignore+=1
  if tag in {'p','h1','h2','h3','li','div','br','tr'}:self.parts.append('\n')
 def handle_endtag(self,tag):
  if tag in {'script','style'}:self.ignore-=1
 def handle_data(self,data):
  if not self.ignore:self.parts.append(data)
records=[]
for name,url in sources.items():
 req=urllib.request.Request(url,headers={'User-Agent':'independent-factual-review/1.0'})
 with urllib.request.urlopen(req,timeout=20) as r:raw=r.read();final=r.url
 (OUT/name).write_bytes(raw)
 if name.endswith('.html'):
  parser=Text();parser.feed(raw.decode());(OUT/(name+'.txt')).write_text(''.join(parser.parts))
 records.append({'name':name,'url':url,'final_url':final,'accessed_on':'2026-10-05','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 print(name,len(raw),hashlib.sha256(raw).hexdigest())
(OUT/'fetch-receipt.json').write_text(json.dumps(records,indent=2)+'\n')
