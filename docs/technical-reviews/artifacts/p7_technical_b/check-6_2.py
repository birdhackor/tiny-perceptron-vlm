from collections import Counter
import inspect,json,platform
from pathlib import Path

def count(corpus):
 pairs=Counter()
 for row in corpus:
  for left,right in zip(row,row[1:]):pairs[(left,right)]+=1
 return pairs

def merge(row,pair):
 out=[];i=0
 while i<len(row):
  if i+1<len(row) and tuple(row[i:i+2])==pair:
   out.append(''.join(pair));i+=2
  else:out.append(row[i]);i+=1
 return out

out=[]
for texts in [['abab','abac'],['abab','abac','acacacac']]:
 corpus=[list(t) for t in texts];pairs=count(corpus);pair=pairs.most_common(1)[0][0];rows=[merge(row,pair) for row in corpus]
 assert [''.join(r) for r in rows]==texts
 out.append({'texts':texts,'pairs':[(list(k),v) for k,v in pairs.items()],'selected':pair,'merged':rows})
assert out[0]['selected']==('a','b') and out[0]['merged']==[['ab','ab'],['ab','a','c']]
assert out[1]['selected']==('a','c') and out[1]['merged']==[['a','b','a','b'],['a','b','ac'],['ac']*4]
missing=Counter()['new'];assert missing==0
p=Path(__file__).resolve().parent/'sources/python-counter.py';p.write_text(inspect.getsource(Counter.__missing__)+'\n'+inspect.getsource(Counter.most_common))
print(json.dumps({'python':platform.python_version(),'device':'cpu','counter_missing':missing,'runs':out,'boundary_scope':'count and merge separately per document','updates':'none'},ensure_ascii=False,indent=2))

