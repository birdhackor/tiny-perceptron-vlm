import pathlib,ast
# Owner independent one-row Wagner-Fischer calculation.
def distance(reference,prediction):
 row=list(range(len(prediction)+1))
 for i,a in enumerate(reference,1):
  prev=row[0];row[0]=i
  for j,b in enumerate(prediction,1):
   old=row[j];row[j]=min(row[j]+1,row[j-1]+1,prev+(a!=b));prev=old
 return row[-1]
for p in ['1000','10100','010','101','1010']:print('edit-example',p,distance('1010',p),distance('1010',p)/4)
p=pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/torchmetrics-edit-helper-v1.8.2.py');s=p.read_text();n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_edit_distance');ns={};exec(compile(ast.get_source_segment(s,n),str(p),'exec'),ns)
x=json.loads(pathlib.Path('docs/technical-reviews/artifacts/p7_technical_c/originals/ocr-raw.json').read_text())['results']['test']['samples'];edits=[]
for v in x:
 d=distance(v['target'],v['generated']);assert d==ns['_edit_distance'](list(v['generated']),list(v['target']));edits.append(d)
print('per item edits',edits);print('history edits/ref/CER',sum(edits),sum(len(v['target']) for v in x),sum(edits)/sum(len(v['target']) for v in x),'EOS',sum(v['eos'] for v in x),'exact',sum(v['generated']==v['target'] for v in x))
try:
 assert len('1010')==len('101')
except AssertionError:print('unequal length rejected')
