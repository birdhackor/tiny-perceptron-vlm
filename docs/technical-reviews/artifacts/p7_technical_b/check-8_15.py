import json
from tiny_perceptron.data import render_chat,IGNORE
from tiny_perceptron.tokenization import ByteTokenizer
rows=[(1,'中文','今日休館'),(2,'英文','Closed today'),(3,'中文','明日開放'),(4,'英文','Open tomorrow')]
material='\n'.join(x[2] for x in rows);tok=ByteTokenizer();out={}
targets={'全部照抄':'\n'.join(x[2] for x in rows),'只抄中文':'\n'.join(x[2] for x in rows if x[1]=='中文'),'中文JSON':json.dumps({'text':[x[2] for x in rows if x[1]=='中文']},ensure_ascii=False)}
for req,target in targets.items():
 x,y=render_chat([{'role':'user','content':material+'\n要求：'+req},{'role':'assistant','content':target}],mode='sft')
 supervised=[n for n in y if n!=IGNORE];assert supervised==tok.encode(target)+[tok.eos_id]
 assert y[-1]==tok.eos_id and len(x)==len(y)
 out[req]={'target':target,'supervised_targets':len(supervised),'EOS_supervised':True,'not_training':True}
english=[x[2] for x in rows if x[1]=='英文'];assert english==['Closed today','Open tomorrow']
out['variation_English']={'plain':'\n'.join(english),'json':{'text':english}}
print(json.dumps(out,ensure_ascii=False,indent=2))
