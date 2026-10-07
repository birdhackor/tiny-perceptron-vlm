import json,platform,torch
from tiny_perceptron.data import render_chat,pad_batch
example=render_chat([{'role':'user','content':'很長的問題'},{'role':'assistant','content':'是'}]);error=''
try:pad_batch([example],max_length=2)
except ValueError as e:error=str(e)
else:raise AssertionError('應該偵測空監督')
x,y=example;first=int((y!=-100).nonzero()[0]);cases=[]
for n in [first+1,len(x)-1,len(x)]:
 a,b,v=pad_batch([example],max_length=n);cases.append({'max_length':n,'active':int((b!=-100).sum()),'targets':b[b!=-100].tolist(),'has_EOS':bool((b==2).any())})
assert first==18 and len(x)==22 and [z['active'] for z in cases]==[1,3,4]
print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','expected_error':error,'first':first,'minimum_nonempty_length':first+1,'full_length':len(x),'question_utf8bytes':len('很長的問題'.encode()),'answer_ids':[239,160,183],'cases':cases},ensure_ascii=False,indent=2))
